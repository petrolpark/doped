"""
Code to generate FHI-aims defect calculation input files.
"""

import copy
import json
import os
import re
import time
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from monty.json import MontyEncoder, MSONable
from monty.serialization import loadfn
from pyfhiaims.control.control import AimsControl
from pyfhiaims.control.cube import AimsCube
from pyfhiaims.geometry.geometry import AimsGeometry
from pymatgen.core.structure import Structure
from pymatgen.io.core import InputSet
from pymatgen.util.typing import PathLike

from doped.core import DefectEntry, _get_bulk_supercell, _get_defect_supercell
from doped.generation import DefectsGenerator, get_defect_name_from_entry
from doped.io.aims.utils import _resolve_species_defaults
from doped.io.inputs import DefectsSetBase

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_kpoints_set = loadfn(os.path.join(MODULE_DIR, "AIMS_sets", "KpointsSet.yaml"))
default_defect_set = loadfn(os.path.join(MODULE_DIR, "AIMS_sets", "AIMS_DefectSet.yaml"))
default_ncl_set = loadfn(os.path.join(MODULE_DIR, "AIMS_sets", "AIMS_NCLDefectSet.yaml"))
default_competing_phases_relax_set = loadfn(
    os.path.join(MODULE_DIR, "AIMS_sets", "AIMS_CompetingPhasesSet.yaml")
)
GAMMA_KPOINTS_SETTINGS = {"k_grid": (1, 1, 1)}
SOC_MIN_ATOMIC_NUMBER = 31  # matches doped.io.vasp.inputs' `DefectRelaxSet.soc` auto-detection

aims_competing_phases_singlepoint_settings = {
    "relax_geometry": "none",  # single-point calculation, mirroring VASP's `IBRION = -1`, `NSW = 0`
    "sc_accuracy_etot": 1.0e-07,  # eV; tighter than `AIMS_CompetingPhasesSet.yaml`'s 1e-6, for the
    # final total energies used in the phase diagram / chemical potential limit determination
}
"""
Overrides applied on top of ``AIMS_CompetingPhasesSet.yaml`` for the final single-point energy
calculation of competing phases (see
``doped.chemical_potentials.CompetingPhases.get_aims_singlepoint_sets``), mirroring VASP's
``singlepoint_incar_settings`` in ``doped/io/vasp/inputs.py``.
"""

PLANAR_POTENTIAL_CUBE_TYPE = "hartree_potential"
"""
FHI-aims ``output cube`` type giving the full (short- + long-range) electrostatic potential --
the FHI-aims analog of VASP's ``LOCPOT``. Requested by default in generated ``control.in`` files
(see ``_planar_potential_cube``), for the FNV (Freysoldt) charge correction (see
``doped.io.aims.outputs.get_planar_averaged_potentials``).
"""

PLANAR_POTENTIAL_CUBE_FILENAME = "hartree_potential"
"""``cube filename`` requested for the ``PLANAR_POTENTIAL_CUBE_TYPE`` output (FHI-aims appends
the format-appropriate file extension itself; see ``doped.io.aims.outputs.
PLANAR_POTENTIAL_CUBE_FILE``, which locates the resulting file for parsing)."""

PLANAR_POTENTIAL_TARGET_GRID_SPACING = 0.2  # Angstrom
"""
Target ``cube edge`` grid spacing along each lattice vector, for the default
``PLANAR_POTENTIAL_CUBE_TYPE`` output. Only the planar average of this grid is needed downstream
(not the full 3D resolution), so a coarser spacing than typical density-visualisation cubes is
used here by default, to keep cube file sizes reasonable for large defect supercells.
"""


def _planar_potential_cube(
    structure: Structure, target_spacing: float = PLANAR_POTENTIAL_TARGET_GRID_SPACING
) -> AimsCube:
    r"""
    Build the ``AimsCube`` requesting the FHI-aims ``hartree_potential`` cube output for
    ``structure``, needed to compute planar-averaged electrostatic potentials for the FNV
    (Freysoldt) charge correction (see ``doped.io.aims.outputs.get_planar_averaged_potentials``).

    The grid step (``cube edge``) along each axis is set explicitly, as ``lattice_vector /
    round(|lattice_vector| / target_spacing)``, rather than relying on FHI-aims' internal default
    spacing (which is not guaranteed to tile the cell exactly, per the FHI-aims manual). This
    ensures that: (1) the cube grid exactly tiles the periodic cell (no partial voxel at the far
    boundary), so grid axis ``i`` corresponds directly to lattice vector ``i``; and (2) -- since
    this only depends on the lattice and ``target_spacing`` -- the resulting grid is guaranteed to
    be identical between a defect supercell and its corresponding bulk supercell, which share the
    same lattice in ``doped``.
    """
    lattice_matrix = structure.lattice.matrix
    n_points = tuple(max(2, round(length / target_spacing)) for length in structure.lattice.abc)
    edges = tuple(tuple(float(x) for x in lattice_matrix[i] / n_points[i]) for i in range(3))
    return AimsCube(
        type=PLANAR_POTENTIAL_CUBE_TYPE,
        origin=(0.0, 0.0, 0.0),
        edges=edges,
        points=n_points,
        filename=PLANAR_POTENTIAL_CUBE_FILENAME,
    )


def _with_default_defect_set(user_parameters: dict[str, Any] | None) -> dict[str, Any]:
    r"""
    Merge ``user_parameters`` with the default ``control.in`` parameters in
    ``doped/io/aims/AIMS_sets/AIMS_DefectSet.yaml``.

    This requests the FHI-aims ``output`` flags required by the parsing functions in
    ``doped.io.aims.outputs`` (e.g. ``output mulliken``, ``output hirshfeld``,
    ``output atom_proj_dos ...``; see ``AIMS_DefectSet.yaml`` for full details/rationale),
    so that the necessary raw outputs are available for later parsing/charge corrections,
    without the user having to set these manually.

    ``user_parameters`` takes priority over the defaults for any overlapping keys, except
    ``output``, for which the user-specified and default ``output`` lines are combined (so
    that any user-requested outputs supplement, rather than replace, the required defaults).
    """
    user_parameters = user_parameters or {}
    merged_parameters = copy.deepcopy(default_defect_set)
    merged_parameters.update(user_parameters)
    merged_parameters["output"] = list(
        dict.fromkeys([*default_defect_set.get("output", []), *user_parameters.get("output", [])])
    )
    return merged_parameters


def _with_default_ncl_set(user_parameters: dict[str, Any] | None) -> dict[str, Any]:
    r"""
    Merge ``user_parameters`` with the default ``control.in`` parameters for the
    ``aims_ncl`` (single-point, spin-orbit-coupled) defect calculation stage, i.e.
    ``doped/io/aims/AIMS_sets/AIMS_DefectSet.yaml`` overridden by
    ``doped/io/aims/AIMS_sets/AIMS_NCLDefectSet.yaml`` (see that file for the rationale
    of each override; e.g. disabling geometry relaxation and including spin-orbit
    coupling), then by ``user_parameters``.
    """
    user_parameters = user_parameters or {}
    merged_parameters = copy.deepcopy(default_defect_set)
    merged_parameters.update(default_ncl_set)
    merged_parameters.update(user_parameters)
    merged_parameters["output"] = list(
        dict.fromkeys([*default_defect_set.get("output", []), *user_parameters.get("output", [])])
    )
    return merged_parameters


# pyfhiaims has a bug where newlines are not inserted between consecutive species' basis set definitions, so the last line of one species' basis set definition runs directly into the next species' comment-banner header (e.g. ``...hydro 5 f 16.0################...``). This regex finds such cases and inserts a newline before the banner.
_MISSING_NEWLINE_BEFORE_BANNER_RE = re.compile(r"(?<=[^\n#])(#{30,})")


_FILE_HEADER_TEMPLATE = (
    f"#{'=' * 79}\n"
    "# FHI-aims {filename}\n"
    "# File generated by doped\n"
    "# {date}\n"
    f"#{'=' * 79}\n\n"
)


class DopedAimsInputSet(InputSet):
    """
    ``FHI-aims`` input set (``control.in``/``geometry.in``) for ``doped`` defect
    calculations, including ``ShakeNBreak`` rattling functionality.

    Generates ``control.in``/``geometry.in`` file contents directly using
    ``pyfhiaims`` (the underlying, official FHI-aims Python package).
    """

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
        self._parameters = parameters
        self._structure = self._structure_for_aims(structure)
        self._properties = properties

        control_in, geometry_in = self.get_input_files()
        super().__init__(
            inputs={
                "control.in": control_in,
                "geometry.in": geometry_in,
                "parameters.json": json.dumps(self._parameters, cls=MontyEncoder),
            }
        )

    def get_input_files(self) -> tuple[str, str]:
        r"""
        Get the ``control.in``/``geometry.in`` contents for this input set.

        ``compute_forces``/``compute_analytical_stress``/``compute_heat_flux`` flags
        are set according to ``self._properties`` (mirroring ``pymatgen``'s
        ``VaspInputSet`` handling of calculated properties), and a workaround is
        applied for a ``pyfhiaims`` bug (as of v1.1.1) where ``SpeciesDefaults.content``
        blocks for consecutive species are concatenated with no separating newline, so
        the last line of one species' basis set definition runs directly into the next
        species' comment-banner header (e.g. ``...hydro 5 f 16.0################...``).

        A ``hartree_potential`` cube output (see ``_planar_potential_cube``) is requested by
        default, for the FNV (Freysoldt) charge correction; this can be overridden (or disabled,
        with an empty list) by setting ``user_parameters["cubes"]`` explicitly.
        """
        property_flags = {
            "forces": "compute_forces",
            "stress": "compute_analytical_stress",
            "stresses": "compute_heat_flux",
        }
        parameters = dict(self._parameters)
        for prop in self._properties:
            aims_name = property_flags.get(prop)
            if aims_name is not None:
                parameters[aims_name] = True

        outputs = parameters.pop("output", [])
        parameters.setdefault("cubes", [_planar_potential_cube(self._structure)])
        control_content = AimsControl(parameters=parameters, outputs=outputs).get_content(self._structure)
        control_content = _MISSING_NEWLINE_BEFORE_BANNER_RE.sub(r"\n\1", control_content)

        date = time.asctime()
        control_in = _FILE_HEADER_TEMPLATE.format(filename="control file: control.in", date=date) + (
            control_content
        )
        geometry_in = _FILE_HEADER_TEMPLATE.format(
            filename="geometry file: geometry.in", date=date
        ) + AimsGeometry.from_structure(self._structure).to_string()

        return control_in, geometry_in

    def _structure_for_aims(self, structure: Structure) -> Structure:
        r"""
        Return a copy with element-only species (no oxidation-state suffix).

        The FHI-supplied ``pyfhiaims`` module makes certain assumptions about
        ``pymatgen`` ``Structure`` objects that are not always true for
        ``Structure`` objects generated by ``doped``. Specifically, that
        ``species_name`` does not include the charge, so any oxidation-state
        decoration (e.g. from ``DefectsGenerator``'s guessed oxidation states)
        is stripped here.

        Per-atom ``initial_charge`` values are deliberately NOT set in
        ``geometry.in``: doped's guessed formal oxidation states rarely sum to
        exactly the defect's ``charge_state`` (particularly at the extremes of
        the considered charge range), so any residual delta would have to be
        concentrated onto a single atom, producing physically extreme formal
        ions (e.g. Se\ :sup:`6-`) whose free-atom reference density FHI-aims
        cannot initialise (``check_occupation_dimensions`` in
        ``free_atoms.f90``: "may be due to a negative ion which would occupy a
        shell that does not exist"). The overall defect charge is still set
        correctly via the ``charge`` ``control.in`` keyword; FHI-aims
        renormalises its neutral free-atom superposition density to match this
        total charge itself (see the "Renormalizing the ... density to the
        exact electron count" ``aims.out`` output).
        """
        aims_structure = structure.copy()
        aims_structure.remove_oxidation_states()
        return aims_structure

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

        Refactored slightly from ``pymatgen``'s generic ``InputSet.write_input()`` to
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

class DefectRelaxSet(MSONable):
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
        soc: bool | None = None,
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
                Dictionary of FHI-aims ``control.in`` parameters (in the
                ``pyfhiaims``/``ase``-style ``AimsControl`` format, e.g.
                ``{"xc": "pbe", "output": ["mulliken"]}``), to override the
                ``doped`` defaults. See ``doped/io/aims/AIMS_sets/AIMS_DefectSet.yaml``
                for the default ``output`` flags requested (needed for the parsing
                functions in ``doped.io.aims.outputs``); any ``output`` entries given
                here are added to (rather than replacing) these required defaults.
                A ``hartree_potential`` cube output is also requested by default (see
                ``_planar_potential_cube``), for the FNV (Freysoldt) charge correction;
                unlike ``output``, this is fully replaced (not merged) by a ``"cubes"``
                entry here (e.g. ``{"cubes": []}`` to disable it).
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
            soc (bool):
                Whether to generate an ``aims_ncl`` (single-point,
                spin-orbit-coupled) ``DopedAimsInputSet``/subfolder. If ``None``
                (default), this is set to ``True`` for defect supercells with a
                max atomic number (Z) >= 31 (i.e. further down the periodic
                table than Zn), otherwise ``False`` -- mirroring
                ``doped.io.vasp.inputs.DefectRelaxSet.soc``.

        """

        self.defect_entry = defect_entry
        self.charge_state = (
            charge_state if charge_state is not None else getattr(defect_entry, "charge_state", 0)
        )
        self.user_parameters = _with_default_defect_set(user_parameters)
        self.user_ncl_parameters = _with_default_ncl_set(user_parameters)
        self.user_properties = user_properties or ("energy", "free_energy")
        self.user_kpoints_settings = (
            copy.deepcopy(user_kpoints_settings) if user_kpoints_settings else dict(default_kpoints_set)
        )
        self.species_defaults = species_defaults
        if species_defaults is not None:
            for parameters in (self.user_parameters, self.user_ncl_parameters):
                if "species_dir" in parameters:
                    raise ValueError(
                        "Specify either species_defaults or user_parameters['species_dir'], not both."
                    )
                parameters["species_dir"] = _resolve_species_defaults(species_defaults)
        self.kwargs = kwargs

        if isinstance(self.defect_entry, Structure):
            self.defect_supercell = self.defect_entry
            self.bulk_supercell = None
        elif isinstance(self.defect_entry, DefectEntry):
            self.defect_supercell = _get_defect_supercell(self.defect_entry)
            self.bulk_supercell = _get_bulk_supercell(self.defect_entry)
        else:
            raise TypeError("defect_entry must be a doped/pymatgen DefectEntry or Structure object.")

        self.soc = soc if soc is not None else max(self.defect_supercell.atomic_numbers) >= SOC_MIN_ATOMIC_NUMBER

        # Use the defect entry / provided `charge_state` as canonical.
        # If the user supplied `user_parameters['charge']` and it differs,
        # warn the user and override with `charge_state`.
        for parameters in (self.user_parameters, self.user_ncl_parameters):
            if "charge" in parameters and int(parameters["charge"]) != int(self.charge_state):
                warnings.warn(
                    "user_parameters['charge'] differs from defect `charge_state`; preferring defect "
                    "`charge_state`.",
                    UserWarning,
                    stacklevel=3,
                )
            try:
                parameters["charge"] = int(self.charge_state)
            except Exception:
                parameters["charge"] = self.charge_state

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

    @property
    def aims_ncl(self) -> DopedAimsInputSet | None:
        """``DefectDictSet``-equivalent for a defect supercell single-point
        calculation with spin-orbit coupling (SOC) included, using ``aims_ncl``
        (mirroring VASP's ``vasp_ncl``). Returns ``None`` if ``self.soc`` is
        ``False``. Uses the same (non-Γ-only) kpoint mesh as ``aims_std``, per
        ``self.user_kpoints_settings``."""
        if not self.soc:
            return None
        return DopedAimsInputSet(
            parameters={**self.user_ncl_parameters, **self.user_kpoints_settings},
            structure=self.defect_supercell,
            properties=self.user_properties,
        )

    def _bulk_input_set(
        self, parameters: dict[str, Any], kpoints_settings: dict[str, Any]
    ) -> DopedAimsInputSet | None:
        if self.bulk_supercell is None:
            return None
        # bulk reference is always neutral, and shouldn't inherit the formal oxidation
        # states doped decorates onto the bulk supercell for internal charge-state guessing
        bulk_parameters = copy.deepcopy(parameters)
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
        return self._bulk_input_set(self.user_parameters, GAMMA_KPOINTS_SETTINGS)

    @property
    def bulk_aims_std(self) -> DopedAimsInputSet | None:
        """``aims_std`` (non-Γ-only) input set for the pristine bulk supercell
        reference calculation."""
        return self._bulk_input_set(self.user_parameters, self.user_kpoints_settings)

    @property
    def bulk_aims_ncl(self) -> DopedAimsInputSet | None:
        """``aims_ncl`` (single-point, spin-orbit-coupled) input set for the
        pristine bulk supercell reference calculation. Returns ``None`` if
        ``self.soc`` is ``False``."""
        if not self.soc:
            return None
        return self._bulk_input_set(self.user_ncl_parameters, self.user_kpoints_settings)

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
        aims_xxx_attribute: DopedAimsInputSet,
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
        bulk: bool = False,
        **kwargs,
    ):
        r"""
        Write FHI-aims input files to ``aims_gam`` and ``aims_std`` (and, if
        ``self.soc`` is ``True``, ``aims_ncl``) subfolders in the ``defect_dir``
        folder.

        - ``aims_gam``:
            Γ-point-only defect supercell relaxation (explicit ``k_grid 1 1 1``).
        - ``aims_std``:
            Defect supercell relaxation with a non-Γ-only kpoint mesh, per
            ``self.user_kpoints_settings`` (default: ``k_grid_density = 5.0``,
            see ``doped/AIMS_sets/KpointsSet.yaml``).
        - ``aims_ncl``:
            Single-point (static) energy calculation with spin-orbit coupling
            (SOC) included, using the same kpoint mesh as ``aims_std``. Only
            written if ``self.soc`` is ``True`` (by default, set to ``True``
            for defect supercells with a max atomic number (Z) >= 31, i.e.
            further down the periodic table than Zn; see ``DefectRelaxSet``
            docstring).

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
            bulk (bool):
                If ``True``, the input files for a single-point calculation of
                the bulk supercell are also written to
                ``"{output_path}/{formula}_bulk/{aims_gam,aims_std}"``, where
                ``output_path`` is the parent directory of ``defect_dir``.
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
        if self.soc:
            self._write_aims_xxx_files(
                defect_dir,
                "aims_ncl",
                rattle,
                self.aims_ncl,
                **kwargs,
            )

        if bulk and self.bulk_aims_gam is not None:
            output_path = os.path.dirname(str(defect_dir)) if "/" in str(defect_dir) else "."
            formula = self.bulk_supercell.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
            bulk_dir = os.path.join(output_path, f"{formula}_bulk")
            self.bulk_aims_gam.write_input(os.path.join(bulk_dir, "aims_gam"), **kwargs)
            self.bulk_aims_std.write_input(os.path.join(bulk_dir, "aims_std"), **kwargs)
            if self.soc:
                self.bulk_aims_ncl.write_input(os.path.join(bulk_dir, "aims_ncl"), **kwargs)


class DefectsSet(DefectsSetBase):
    """
    Class for generating input files for ``FHI-aims`` defect calculations for a set
    of ``doped``/``pymatgen`` ``DefectEntry`` objects.
    """

    _input_set_name = "DefectRelaxSet"

    def __init__(
        self,
        defect_entries: DefectsGenerator | dict[str, DefectEntry] | list[DefectEntry] | DefectEntry,
        user_parameters: dict[str, Any] | None = None,
        user_properties: Sequence[str] | None = None,
        user_kpoints_settings: dict[str, Any] | None = None,
        species_defaults: PathLike | None = None,
        soc: bool | None = None,
        **kwargs,
    ):
        r"""
        Creates a dictionary of ``{defect name: DefectRelaxSet}``.

        Input files are written separately with :meth:`write_files`, matching
        the lifecycle of :class:`doped.io.vasp.inputs.DefectsSet`.

        ``soc`` (bool):
            Whether to generate ``aims_ncl`` (single-point, spin-orbit-coupled)
            input sets. If ``None`` (default), this is determined once for the
            whole set (mirroring ``doped.io.vasp.inputs.DefectsSet``), set to
            ``True`` if the max atomic number (Z) across all defect supercells
            in the set is >= 31 (i.e. further down the periodic table than
            Zn), otherwise ``False``.
        """
        self.user_parameters = user_parameters
        self.user_properties = user_properties
        self.user_kpoints_settings = user_kpoints_settings
        self.species_defaults = species_defaults
        self._input_soc = soc
        super().__init__(defect_entries, **kwargs)  # format entries & build ``DefectRelaxSet``s

        # All entries in a DefectsGenerator share the same bulk supercell. As with
        # DefectsSet (VASP), expose the final set's bulk reference at set level.
        defect_relax_set = list(self.defect_sets.values())[-1]
        self.bulk_supercell = defect_relax_set.bulk_supercell
        self.bulk_aims_gam = defect_relax_set.bulk_aims_gam
        self.bulk_aims_std = defect_relax_set.bulk_aims_std
        self.bulk_aims_ncl = defect_relax_set.bulk_aims_ncl

    def _setup(self):
        """
        Determine whether SOC (``aims_ncl``) input sets should be generated for
        the whole set, if ``soc`` was not explicitly set, mirroring
        ``doped.io.vasp.inputs.DefectsSet._setup``.
        """
        if self._input_soc is not None:
            self.soc = self._input_soc
            return

        max_atomic_num = max(
            max(_get_defect_supercell(defect_entry).atomic_numbers)
            for defect_entry in self.defect_entries.values()
        )
        self.soc = max_atomic_num >= SOC_MIN_ATOMIC_NUMBER

    def _defect_input_set(self, defect_entry: DefectEntry) -> DefectRelaxSet:
        """
        Build the ``DefectRelaxSet`` for a single defect entry.
        """
        return DefectRelaxSet(
            defect_entry=defect_entry,
            charge_state=defect_entry.charge_state,
            soc=self.soc,
            user_parameters=self.user_parameters,
            user_properties=self.user_properties,
            user_kpoints_settings=self.user_kpoints_settings,
            species_defaults=self.species_defaults,
            **self.kwargs,
        )

    @staticmethod
    def _write_defect(args):
        defect_species, defect_relax_set, output_path, bulk, write_kwargs = args
        kwargs = dict(write_kwargs)
        defect_relax_set.write_all(
            defect_dir=os.path.join(output_path, defect_species),
            rattle=kwargs.pop("rattle", False),
            bulk=bulk,
            **kwargs,
        )

    def write_files(  # type: ignore[override]  # deliberately extends the base signature
        self,
        output_path: PathLike = ".",
        rattle: bool = True,
        bulk: bool | str = True,
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
        super().write_files(
            output_path=output_path,
            bulk=bulk,
            processes=processes,
            rattle=rattle,
            **kwargs,
        )

    def __setitem__(self, key, value):
        """
        Set the value of a specific key (defect name) in the ``defect_sets``
        dictionary.

        Note that other |DefectsSet| attributes, like ``self.bulk_aims_...``
        are not changed.
        """
        if not isinstance(value, DefectRelaxSet):
            raise TypeError(f"Value must be a DefectRelaxSet object, not {type(value).__name__}")
        if value.bulk_supercell != self.bulk_supercell:
            warnings.warn(
                "Note that the bulk supercell of the input DefectRelaxSet differs from that of "
                "the DefectsSet. This could lead to inaccuracies in parsing/predictions if the "
                "bulk supercells are not the same!\n"
                f"DefectRelaxSet bulk supercell:\n{value.bulk_supercell}\n"
                f"DefectsSet bulk supercell:\n{self.bulk_supercell}"
            )
        self.defect_sets[key] = value
