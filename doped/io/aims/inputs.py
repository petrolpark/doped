"""
Code to generate FHI-aims defect calculation input files.
"""

import contextlib
import copy
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn
from pymatgen.core import SETTINGS
from pymatgen.core.structure import Structure
from pymatgen.io.aims.sets import \
    AimsInputSet  # TODO remove dependency on pre-alpha package
from pymatgen.util.typing import PathLike

from doped import _doped_obj_properties_methods, get_mp_context, pool_manager
from doped.core import DefectEntry
from doped.generation import (DefectsGenerator, get_defect_name_from_entry,
                              name_defect_entries)
from doped.utils.parsing import _get_bulk_supercell, _get_defect_supercell
from doped.utils.symmetry import _frac_coords_sort_func

_SPECIES_DEFAULTS_SHORTHANDS = ("light", "tight", "really_tight")
AIMS_PATH: str | None = None
AIMS_PATH_SEARCHED = False

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
default_kpoints_set = loadfn(os.path.join(PACKAGE_DIR, "AIMS_sets", "KpointsSet.yaml"))
GAMMA_KPOINTS_SETTINGS = {"k_grid": (1, 1, 1)}

r"""
Lazily infer the AIMS species-defaults directory if AIMS_SPECIES_DIR is not set.
This allows fallback discovery from common AIMS binary locations on the host.
"""
def _infer_species_dir_from_exe_path(aims_exe: Path) -> str | None:
    candidate_dirs = [
        aims_exe.parent.parent / "species_defaults",
        aims_exe.parent.parent / "share" / "aims" / "species_defaults",
        aims_exe.parent.parent / "aims" / "species_defaults",
        aims_exe.parent / "species_defaults",
    ]
    for candidate in candidate_dirs:
        if (candidate / "defaults_2020" / "light").is_dir():
            return str(candidate)
    return None


def _discover_aims_species_dir() -> str | None:
    global AIMS_PATH, AIMS_PATH_SEARCHED
    if AIMS_PATH is not None:
        return AIMS_PATH
    if AIMS_PATH_SEARCHED:
        return None
    AIMS_PATH_SEARCHED = True

    aims_species_dir = SETTINGS.get("AIMS_SPECIES_DIR")
    if aims_species_dir:
        AIMS_PATH = aims_species_dir
        return AIMS_PATH

    search_paths = []
    binary_path = shutil.which("aims")
    if binary_path:
        search_paths.append(Path(binary_path))
    for path in (
        "/opt/aims/bin/aims",
        "/opt/fhi-aims/bin/aims",
        "/usr/local/bin/aims",
        "/usr/bin/aims",
        "/snap/bin/aims",
    ):
        exe = Path(path)
        if exe.exists():
            search_paths.append(exe)

    for exe in search_paths:
        species_dir = _infer_species_dir_from_exe_path(exe)
        if species_dir:
            warnings.warn(
                f"Found FHI-aims executable at {exe}. "
                "Set AIMS_SPECIES_DIR in pymatgen settings to the directory containing "
                "defaults_2020 for reliable species-default resolution.",
                UserWarning,
                stacklevel=3,
            )
            AIMS_PATH = species_dir
            return AIMS_PATH

    warnings.warn(
        "AIMS_SPECIES_DIR is not configured in pymatgen settings and a valid "
        "FHI-aims species defaults directory could not be inferred from a common "
        "binary location. Please set AIMS_SPECIES_DIR to the directory containing "
        "defaults_2020.",
        UserWarning,
        stacklevel=3,
    )
    return None


def _resolve_species_defaults(species_defaults: PathLike | str) -> str:
    """Return an existing directory containing FHI-aims species-default files.

    ``"light"``, ``"tight"``, and ``"really_tight"`` are reserved
    shorthands for ``<AIMS_SPECIES_DIR>/defaults_2020/<shorthand>``, where
    ``AIMS_SPECIES_DIR`` is pymatgen's configured FHI-aims species-defaults
    directory.

    If the input is a string that is not one of the three shorthand values,
    it is first interpreted as a literal path. If that literal path is not
    absolute and ``AIMS_SPECIES_DIR`` is configured, the path is also
    interpreted relative to ``AIMS_SPECIES_DIR``.

    If ``AIMS_SPECIES_DIR`` is not configured, this module will attempt to
    infer it lazily from a known AIMS executable location. It will warn if
    the executable is found and a warning if it cannot infer a valid path.
    """
    if isinstance(species_defaults, str) and species_defaults in _SPECIES_DEFAULTS_SHORTHANDS:
        aims_species_dir = SETTINGS.get("AIMS_SPECIES_DIR") or _discover_aims_species_dir()
        if not aims_species_dir:
            raise ValueError(
                "AIMS_SPECIES_DIR must be configured in pymatgen settings to use the "
                f"{species_defaults!r} species_defaults shorthand. Alternatively, pass the full "
                "path to the directory containing the element default files."
            )
        species_defaults_path = Path(aims_species_dir) / "defaults_2020" / species_defaults
    else:
        species_defaults_path = Path(species_defaults).expanduser()
        if not species_defaults_path.is_absolute():
            aims_species_dir = SETTINGS.get("AIMS_SPECIES_DIR") or _discover_aims_species_dir()
            if aims_species_dir:
                relative_path = Path(aims_species_dir) / species_defaults
                if relative_path.is_dir():
                    species_defaults_path = relative_path

    if not species_defaults_path.is_dir():
        raise FileNotFoundError(
            "species_defaults must be an existing directory containing FHI-aims element default "
            f"files, got: {species_defaults_path}"
        )
    return str(species_defaults_path.resolve())


# pyfhiaims has a bug where newlines are not inserted between consecutive species' basis set definitions, so the last line of one species' basis set definition runs directly into the next species' comment-banner header (e.g. ``...hydro 5 f 16.0################...``). This regex finds such cases and inserts a newline before the banner.
_MISSING_NEWLINE_BEFORE_BANNER_RE = re.compile(r"(?<=[^\n#])(#{30,})")


class DopedAimsInputSet(AimsInputSet):
    """
    Extension to ``pymatgen-io-aims`` ``AimsInputSet`` object for ``FHI-aims`` defect calculations,
    including SnB rattling functionality
    """

    def get_input_files(self) -> tuple[str, str]:
        r"""
        Get the ``control.in``/``geometry.in`` contents, working around a ``pyfhiaims``
        bug (as of v1.1.1) where ``SpeciesDefaults.content`` blocks for consecutive
        species are concatenated with no separating newline, so the last line of one
        species' basis set definition runs directly into the next species' comment-
        banner header (e.g. ``...hydro 5 f 16.0################...``).
        """
        control_in, geometry_in = super().get_input_files()
        control_in = _MISSING_NEWLINE_BEFORE_BANNER_RE.sub(r"\n\1", control_in)
        return control_in, geometry_in

    def _structure_for_aims(self, structure: Structure) -> Structure:
        r"""
        Return a copy with element-only species and explicit per-site charge.

        The FHI-supplied ``pyfhiaims`` module makes certain assumptions about ``pymatgen`` ``Structure`` objects that are not always true for ``Structure`` objects generated by ``doped``.
        Specifically, that ``species_name`` does not include the charge. This method strips the charge from the species name and adds the explicit ``charge`` site 
        property to the ``Structure`` that is used by ``pyfhiaims`` to write the ``geometry.in`` file.
        
        """
        aims_structure = structure.copy()
        charges = []
        needs_charge_property = False

        for site in aims_structure:
            explicit_charge = site.properties.get("charge")
            specie_charge = getattr(site.specie, "oxi_state", None)

            if explicit_charge is not None:
                charges.append(explicit_charge)
                if specie_charge is not None and explicit_charge != specie_charge:
                    raise ValueError(
                        "site.properties['charge'] and species oxidation state do not agree."
                    )
                needs_charge_property = True
            elif specie_charge is not None:
                charges.append(specie_charge)
                if specie_charge != 0:
                    needs_charge_property = True
            else:
                charges.append(0.0)

        if charges:
            charge_delta = float(self.charge_state) - float(sum(charges))
            if abs(charge_delta) > 1e-12:
                charges[0] += charge_delta
                needs_charge_property = True

        if needs_charge_property:
            aims_structure.add_site_property("charge", charges)

        aims_structure.remove_oxidation_states()
        return aims_structure

    def __init__(
        self,
        parameters: dict[str, Any],
        structure: Structure,
        properties: Sequence[str] = ("energy", "free_energy"),
    ):
        r"""
        TODO docs
        """
        self.charge_state = parameters.get("charge", 0)
        self._structure = self._structure_for_aims(structure)
        super().__init__(parameters, self._structure, properties)

    def write_input(
        self,
        output_path,
        rattle: bool = False,
        make_dir_if_not_present = True,
        zip_output = False,
        stdev: float | None = None,
        d_min: float | None = None,
    ):
        r"""
        Writes out all input to a directory.

        Refactored slightly from ``pymatgen-io-aims`` ``AimsInputSet.write_input()`` to
        allow generation of rattled structures.

        Args:
            output_path (PathLike):
                Directory to output the ``FHI-aims`` input files.
            rattle (bool):
                Apply random displacements to all atomic
                positions in the structure using the ``ShakeNBreak`` algorithm;
                i.e. with the displacement distances randomly drawn from a
                Gaussian distribution of standard deviation equal to 10% of the
                nearest neighbour distance and using a Monte Carlo algorithm to
                penalise displacements that bring atoms closer than 80% of the
                nearest neighbour distance.
                ``stdev`` and ``d_min`` can also be given as input kwargs.
                This is intended to be used as a fallback option for breaking
                symmetry to partially aid location of global minimum defect
                geometries, if ``ShakeNBreak`` structure-searching is being
                skipped. However, rattling still only finds the ground-state
                structure for <~30% of known cases of energy-lowering
                reconstructions relative to an unperturbed defect structure.
                (default: False)
            make_dir_if_not_present (bool):
                Set to ``True`` if you want the directory (and the whole path)
                to be created if it is not present. (default: True)
            zip_output (bool):
                Whether to zip each ``FHI-aims`` input file written to the output
                directory. (default: False)
            stdev (float):
                Standard deviation for the Gaussian distribution of
                displacements for the ``ShakeNBreak`` rattling algorithm. If
                ``None`` (default) this is set to 10% of the nearest neighbour
                distance in the structure.
            d_min (float):
                Minimum interatomic distance (in Angstroms) in the rattled
                structure. Monte Carlo rattle moves that put atoms at distances
                less than this will be heavily penalised. Default is to set
                this to 80% of the nearest neighbour distance in the structure.
        """

        # apply SnB algo
        if rattle:
            try:
                from shakenbreak.distortions import rattle as SnB_rattle
            except ImportError as e:
                raise ImportError(
                    "ShakeNBreak must be installed (pip install shakenbreak) to use the rattle option!"
                ) from e
            self._structure: Structure = SnB_rattle(self._structure, stdev=stdev, d_min=d_min)
            control_in, geometry_in = self.get_input_files()
            self.inputs["control.in"] = control_in
            self.inputs["geometry.in"] = geometry_in

        super().write_input(output_path, make_dir_if_not_present, overwrite=True, zip_inputs=zip_output)

class AimsDefectRelaxSet(MSONable):
    """
    Class for generating input files for ``FHI-aims`` defect relaxation
    calculations for a single ``pymatgen`` ``DefectEntry`` or ``Structure``
    object.
    """

    defect_entry: DefectEntry | Structure
    charge_state: int
    defect_supercell: Structure
    bulk_supercell: Structure | None

    user_parameters: dict[str, Any]
    user_properties: Sequence[str]

    def __init__(
        self,
        defect_entry: DefectEntry | Structure,
        charge_state: int | None = None,
        user_parameters: dict[str, Any] | None = None,
        user_properties: Sequence[str] | None = None,
        user_kpoints_settings: dict[str, Any] | None = None,
        species_defaults: PathLike | None = None,
        **kwargs,
    ):
        r"""

        Args:
            defect_entry (DefectEntry, Structure):
                ``doped``/``pymatgen`` ``DefectEntry`` or ``Structure`` (defect
                supercell) for which to generate ``DefectDictSet``\s for.
            charge_state (int):
                Charge state of the defect. Overrides
                ``DefectEntry.charge_state`` if ``DefectEntry`` is input.
            user_parameters (dict[str, Any]):

            user_properties (Sequence[str]):

            user_kpoints_settings (dict[str, Any]):
                Dictionary of FHI-aims ``control.in`` k-point keywords (e.g.
                ``{"k_grid_density": 5.0}`` or ``{"k_grid": (4, 4, 4)}``), used
                for the ``aims_std`` (non-Γ-only) ``DefectDictSet``\s.
                Default (see ``doped/AIMS_sets/KpointsSet.yaml``) is
                ``{"k_grid_density": 5.0}``. ``aims_gam`` always uses an
                explicit Γ-only ``k_grid 1 1 1``, regardless of this setting.
            species_defaults (PathLike):
                Full path to the FHI-aims directory containing the element
                default files (e.g. ``.../defaults_2020/tight``). Alternatively,
                ``"light"``, ``"tight"``, or ``"really_tight"`` uses the
                corresponding ``defaults_2020`` subdirectory of pymatgen's
                configured ``AIMS_SPECIES_DIR``.
                A path relative to ``AIMS_SPECIES_DIR`` is also accepted if
                ``AIMS_SPECIES_DIR`` is configured.

        """

        self.defect_entry = defect_entry
        self.charge_state = (
            charge_state if charge_state is not None else getattr(defect_entry, "charge_state", 0)
        )
        self.user_parameters = copy.deepcopy(user_parameters) if user_parameters else {}
        self.user_properties = user_properties or ("energy", "free_energy")
        self.user_kpoints_settings = (
            copy.deepcopy(user_kpoints_settings) if user_kpoints_settings else dict(default_kpoints_set)
        )
        self.species_defaults = species_defaults
        if species_defaults is not None:
            if "species_dir" in self.user_parameters:
                raise ValueError(
                    "Specify either species_defaults or user_parameters['species_dir'], not both."
                )
            self.user_parameters["species_dir"] = _resolve_species_defaults(species_defaults)
        self.kwargs = kwargs

        if isinstance(self.defect_entry, Structure):
            self.defect_supercell = self.defect_entry
            self.bulk_supercell = None
        elif isinstance(self.defect_entry, DefectEntry):
            self.defect_supercell = _get_defect_supercell(self.defect_entry)
            self.bulk_supercell = _get_bulk_supercell(self.defect_entry)
        else:
            raise TypeError("defect_entry must be a doped/pymatgen DefectEntry or Structure object.")

        # Use the defect entry / provided `charge_state` as canonical.
        # If the user supplied `user_parameters['charge']` and it differs,
        # warn the user and override with `charge_state`.
        if "charge" in self.user_parameters and int(self.user_parameters["charge"]) != int(self.charge_state):
            warnings.warn(
                "user_parameters['charge'] differs from defect `charge_state`; preferring defect `charge_state`.",
                UserWarning,
                stacklevel=3,
            )

        try:
            self.user_parameters["charge"] = int(self.charge_state)
        except Exception:
            self.user_parameters["charge"] = self.charge_state
        #TODO probably need some required properties in user_properties in order to get the info needed for parsing

    @property
    def aims_gam(self) -> DopedAimsInputSet:
        """``DefectDictSet``-equivalent for a Γ-point-only (``aims_gam``) defect
        supercell relaxation."""
        return DopedAimsInputSet(
            parameters={**self.user_parameters, **GAMMA_KPOINTS_SETTINGS},
            structure=self.defect_supercell,
            properties=self.user_properties,
        )

    @property
    def aims_std(self) -> DopedAimsInputSet:
        """``DefectDictSet``-equivalent for a defect supercell relaxation using
        ``aims_std`` (i.e. with a non-Γ-only kpoint mesh, per
        ``self.user_kpoints_settings``)."""
        return DopedAimsInputSet(
            parameters={**self.user_parameters, **self.user_kpoints_settings},
            structure=self.defect_supercell,
            properties=self.user_properties,
        )

    def _bulk_input_set(self, kpoints_settings: dict[str, Any]) -> DopedAimsInputSet | None:
        if self.bulk_supercell is None:
            return None
        # bulk reference is always neutral, and shouldn't inherit the formal oxidation
        # states doped decorates onto the bulk supercell for internal charge-state guessing
        bulk_parameters = copy.deepcopy(self.user_parameters)
        bulk_parameters["charge"] = 0
        bulk_parameters.update(kpoints_settings)
        bulk_structure = self.bulk_supercell.copy()
        bulk_structure.remove_oxidation_states()
        return DopedAimsInputSet(
            parameters=bulk_parameters,
            structure=bulk_structure,
            properties=self.user_properties,
        )

    @property
    def bulk_aims_gam(self) -> DopedAimsInputSet | None:
        """Γ-point-only (``aims_gam``) input set for the pristine bulk supercell
        reference calculation."""
        return self._bulk_input_set(GAMMA_KPOINTS_SETTINGS)

    @property
    def bulk_aims_std(self) -> DopedAimsInputSet | None:
        """``aims_std`` (non-Γ-only) input set for the pristine bulk supercell
        reference calculation."""
        return self._bulk_input_set(self.user_kpoints_settings)

    #TODO currently copied from VASP implementation. would like to move this to ABC for both VASP, aims and others
    def _get_output_path(self, defect_dir: PathLike | None = None, subfolder: PathLike | None = None):
        if defect_dir is None:
            defect_name = getattr(self.defect_entry, "name", None)
            if defect_name is None:
                if isinstance(self.defect_entry, DefectEntry):
                    self.defect_entry.name = get_defect_name_from_entry(self.defect_entry, relaxed=False)
                    defect_name = self.defect_entry.name
                else:
                    defect_name = self.defect_entry.composition.reduced_formula

            defect_dir = defect_name

        return f"{defect_dir}/{subfolder}" if subfolder is not None else defect_dir

    def _write_aims_xxx_files(
        self,
        defect_dir: PathLike | None,
        subfolder: PathLike | None,
        rattle: bool,
        aims_xxx_attribute: AimsInputSet,
        **kwargs,
    ):
        output_path = self._get_output_path(defect_dir, subfolder)

        # also all copied from VASP equivalent, would like to be superclass
        stdev = d_min = None
        if rattle and isinstance(self.defect_entry, DefectEntry):
            for trial_structure in [
                self.defect_entry.defect.structure,
                self.defect_entry.bulk_supercell,
                self.defect_entry.bulk_supercell * 2,
            ]:
                if trial_structure is None:
                    continue
                distance_matrix = trial_structure.distance_matrix
                sorted_distances = np.sort(distance_matrix[distance_matrix > 0.8].flatten())
                if len(sorted_distances) > 0:
                    stdev = 0.1 * sorted_distances[0]
                    d_min = 0.8 * sorted_distances[0]
                    break

        aims_xxx_attribute.write_input(
            output_path,
            rattle=rattle,
            stdev=stdev,
            d_min=d_min,
            **kwargs,
        )

        if isinstance(self.defect_entry, DefectEntry) and "bulk" not in defect_dir:  # not a bulk supercell
            self.defect_entry.to_json(f"{output_path}/{self.defect_entry.name}.json.gz")

    def write_all(
        self,
        defect_dir: PathLike | None = None,
        rattle: bool = False,
        **kwargs,
    ):
        r"""
        Write FHI-aims input files to ``aims_gam`` and ``aims_std`` subfolders in
        the ``defect_dir`` folder.

        - ``aims_gam``:
            Γ-point-only defect supercell relaxation (explicit ``k_grid 1 1 1``).
        - ``aims_std``:
            Defect supercell relaxation with a non-Γ-only kpoint mesh, per
            ``self.user_kpoints_settings`` (default: ``k_grid_density = 5.0``,
            see ``doped/AIMS_sets/KpointsSet.yaml``).

        Args:
            defect_dir (PathLike):
                Folder in which to create the ``FHI-aims`` defect calculation
                inputs. Default is to use the ``DefectEntry`` name (e.g.
                ``"Y_i_C4v_O1.92_+2"`` etc.), from ``self.defect_entry.name``.
                If this attribute is not set, it is automatically generated
                according to the doped convention (using
                ``get_defect_name_from_entry()``).
            rattle (bool):
                If writing ``POSCAR``, apply random displacements to all atomic
                positions in the structure using the ``ShakeNBreak`` algorithm;
                i.e. with the displacement distances randomly drawn from a
                Gaussian distribution of standard deviation equal to 10% of the
                nearest neighbour distance and using a Monte Carlo algorithm to
                penalise displacements that bring atoms closer than 80% of the
                nearest neighbour distance.
                ``stdev`` and ``d_min`` can also be given as input kwargs.
                This is intended to be used as a fallback option for breaking
                symmetry to partially aid location of global minimum defect
                geometries, if ``ShakeNBreak`` structure-searching is being
                skipped. However, rattling still only finds the ground-state
                structure for <~30% of known cases of energy-lowering
                reconstructions relative to an unperturbed defect structure.
                (default: False)
            # stdev (float):
            #     Standard deviation for the Gaussian distribution of
            #     displacements for the ``ShakeNBreak`` rattling algorithm. If
            #     ``None`` (default) this is set to 10% of the nearest neighbour
            #     distance in the structure.
            # d_min (float):
            #     Minimum interatomic distance (in Angstroms) in the rattled
            #     structure. Monte Carlo rattle moves that put atoms at distances
            #     less than this will be heavily penalised. Default is to set
            #     this to 80% of the nearest neighbour distance in the structure.
        """
        if defect_dir is None:
            defect_dir = self.defect_entry.name

        self._write_aims_xxx_files(
            defect_dir,
            "aims_gam",
            rattle,
            self.aims_gam,
            **kwargs,
        )
        self._write_aims_xxx_files(
            defect_dir,
            "aims_std",
            rattle,
            self.aims_std,
            **kwargs,
        )

class AimsDefectsSet(MSONable):
    """
    Class for generating input files for ``FHI-aims`` defect calculations for a set
    of ``doped``/``pymatgen`` ``DefectEntry`` objects.
    """

    def __init__(
        self,
        defect_entries: DefectsGenerator | dict[str, DefectEntry] | list[DefectEntry] | DefectEntry,
        user_parameters: dict[str, Any] | None = None,
        user_properties: Sequence[str] | None = None,
        user_kpoints_settings: dict[str, Any] | None = None,
        species_defaults: PathLike | None = None,
        **kwargs,
    ):
        r"""
        Creates a dictionary of ``{defect name: AimsDefectRelaxSet}``.

        Input files are written separately with :meth:`write_files`, matching
        the lifecycle of :class:`doped.vasp.DefectsSet`.
        """
        self.user_parameters = user_parameters
        self.user_properties = user_properties
        self.user_kpoints_settings = user_kpoints_settings
        self.species_defaults = species_defaults
        self.kwargs = kwargs
        self.defect_entries, self.json_name, self.json_obj = self._format_defect_entries_input(
            defect_entries
        )
        self.defect_sets: dict[str, AimsDefectRelaxSet] = {
            defect_species: AimsDefectRelaxSet(
                defect_entry=defect_entry,
                charge_state=defect_entry.charge_state,
                user_parameters=self.user_parameters,
                user_properties=self.user_properties,
                user_kpoints_settings=self.user_kpoints_settings,
                species_defaults=self.species_defaults,
                **self.kwargs,
            )
            for defect_species, defect_entry in self.defect_entries.items()
        }

        if not self.defect_sets:
            raise ValueError(
                "No `AimsDefectRelaxSet` objects created, indicating problems with the "
                "`AimsDefectsSet` input/creation!"
            )

        # All entries in a DefectsGenerator share the same bulk supercell. As
        # with DefectsSet, expose the final set's bulk reference at set level.
        defect_relax_set = list(self.defect_sets.values())[-1]
        self.bulk_supercell = defect_relax_set.bulk_supercell

    @property
    def bulk_aims_gam(self) -> DopedAimsInputSet | None:
        """Γ-point-only (``aims_gam``) input set for the pristine bulk supercell
        reference calculation."""
        return list(self.defect_sets.values())[-1].bulk_aims_gam

    @property
    def bulk_aims_std(self) -> DopedAimsInputSet | None:
        """``aims_std`` (non-Γ-only) input set for the pristine bulk supercell
        reference calculation."""
        return list(self.defect_sets.values())[-1].bulk_aims_std

    def _format_defect_entries_input(
        self,
        defect_entries: DefectsGenerator | dict[str, DefectEntry] | list[DefectEntry] | DefectEntry,
    ) -> tuple[dict[str, DefectEntry], str, dict[str, DefectEntry] | DefectsGenerator]:
        """Normalise supported defect-entry inputs and prepare provenance data."""
        json_name = "defect_entries.json.gz"
        json_obj = defect_entries
        if type(defect_entries).__name__ == "DefectsGenerator":
            defect_entries = defect_entries  # type narrowing for static type checkers
            formula = defect_entries.primitive_structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
            json_name = f"{formula}_defects_generator.json.gz"
            defect_entries = defect_entries.defect_entries
        elif isinstance(defect_entries, DefectEntry):
            defect_entries = [defect_entries]

        if isinstance(defect_entries, list):
            defect_entry_list = copy.deepcopy(defect_entries)
            with contextlib.suppress(AttributeError, TypeError):
                defect_entry_list.sort(key=lambda entry: _frac_coords_sort_func(entry.conv_cell_frac_coords))

            unnamed_entries = [entry for entry in defect_entry_list if not hasattr(entry, "name")]
            for name, entry in name_defect_entries(unnamed_entries).items():
                entry.name = f"{name}_{'+' if entry.charge_state > 0 else ''}{entry.charge_state}"

            if len({entry.name for entry in defect_entry_list}) != len(defect_entry_list):
                raise ValueError(
                    "Some defect entries have the same name, due to mixing of named and unnamed input "
                    "`DefectEntry`s! This would cause defect folders to be overwritten. Please check "
                    "your DefectEntry names and/or generate your defects using DefectsGenerator instead."
                )
            defect_entries = {entry.name: entry for entry in defect_entry_list}
            formula = defect_entry_list[0].defect.structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
            json_name = f"{formula}_defect_entries.json.gz"
            json_obj = defect_entries

        if isinstance(defect_entries, dict) and not all(
            isinstance(entry, DefectEntry) for entry in defect_entries.values()
        ):
            raise TypeError(
                "Input defect_entries dict must be of the form {defect_name: DefectEntry}, got dict "
                f"with values of type {[type(value) for value in defect_entries.values()]} instead"
            )
        if not isinstance(defect_entries, dict):
            raise TypeError(
                "Input defect_entries must be of type DefectsGenerator, dict, list or DefectEntry, got "
                f"type {type(defect_entries)} instead."
            )
        return defect_entries, json_name, json_obj

    @staticmethod
    def _write_defect(args):
        defect_species, defect_relax_set, output_path, rattle, kwargs = args
        defect_relax_set.write_all(
            defect_dir=os.path.join(output_path, defect_species), rattle=rattle, **kwargs
        )

    def write_files(
        self,
        output_path: PathLike = ".",
        rattle: bool = True,
        bulk: bool = True,
        processes: int | None = None,
        **kwargs,
    ):
        """Write defect inputs and, optionally, one bulk reference input folder.

        Defect inputs are written to ``<output_path>/<defect name>/aims_gam``
        and ``<output_path>/<defect name>/aims_std``. When ``bulk`` is true, a
        pristine reference calculation is written once to
        ``<output_path>/<formula>_bulk/aims_gam`` and
        ``<output_path>/<formula>_bulk/aims_std``.
        """
        args_list = [
            (
                defect_species,
                defect_relax_set,
                output_path,
                rattle,
                kwargs,
            )
            for defect_species, defect_relax_set in self.defect_sets.items()
        ]

        if processes is None:
            mp = get_mp_context()
            processes = min(round(len(args_list) / 30), mp.cpu_count() - 1)

        if processes <= 1:
            for args in args_list:
                self._write_defect(args)
        else:
            mp = get_mp_context()
            processes = min(processes, max(1, mp.cpu_count() - 1))
            with pool_manager(processes) as pool:
                pool.map(self._write_defect, args_list)

        if bulk and self.bulk_aims_gam is not None:
            formula = self.bulk_supercell.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
            bulk_dir = os.path.join(output_path, f"{formula}_bulk")
            self.bulk_aims_gam.write_input(os.path.join(bulk_dir, "aims_gam"), **kwargs)
            self.bulk_aims_std.write_input(os.path.join(bulk_dir, "aims_std"), **kwargs)

        dumpfn(self.json_obj, os.path.join(output_path, self.json_name))

    def __repr__(self):
        """
        Returns a string representation of the ``AimsDefectsSet`` object.
        """
        first_entry = next(iter(self.defect_entries.values()))
        formula = (
            first_entry.defect.structure.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
            if isinstance(first_entry, DefectEntry)
            else "structure"
        )
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped AimsDefectsSet for bulk composition {formula}, with {len(self.defect_entries)} "
            f"defect entries in self.defect_entries. Available attributes:\n{properties}\n\n"
            f"Available methods:\n{methods}"
        )

    def __getattr__(self, attr):
        """
        Redirect unknown attribute access to the ``defect_sets`` dictionary.
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        if hasattr(self.defect_sets, attr):
            return getattr(self.defect_sets, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __getitem__(self, key):
        return self.defect_sets[key]

    def __setitem__(self, key, value):
        if not isinstance(value, AimsDefectRelaxSet):
            raise TypeError(f"Value must be a AimsDefectRelaxSet object, not {type(value).__name__}")
        if value.bulk_supercell != self.bulk_supercell:
            warnings.warn(
                "Note that the bulk supercell of the input AimsDefectRelaxSet differs from that of "
                "the AimsDefectsSet. This could lead to inaccuracies in parsing/predictions if the "
                "bulk supercells are not the same!\n"
                f"AimsDefectRelaxSet bulk supercell:\n{value.bulk_supercell}\n"
                f"AimsDefectsSet bulk supercell:\n{self.bulk_supercell}"
            )
        self.defect_sets[key] = value

    def __delitem__(self, key):
        del self.defect_sets[key]

    def __contains__(self, key):
        return key in self.defect_sets

    def __len__(self):
        return len(self.defect_sets)

    def __iter__(self):
        return iter(self.defect_sets)
