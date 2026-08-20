"""
``FHI-aims`` input generation for competing-phase (chemical potential limit)
calculations.

This provides :class:`AimsCompetingPhasesMixin`, mixed into
:class:`~doped.chemical_potentials.CompetingPhases`, so that the ``FHI-aims``-
specific input-generation logic lives alongside the other ``FHI-aims`` input-
generation machinery in this ``io.aims`` subpackage (mirroring
:mod:`doped.io.aims.inputs`), rather than in the (nominally calculator-
agnostic) :mod:`doped.chemical_potentials` module.
"""

import copy
import os
import warnings
from typing import Any

from pymatgen.core import Element
from pymatgen.core.entries import ComputedEntry
from pymatgen.core.structure import Structure
from pymatgen.util.typing import PathLike

from doped.io.aims.inputs import (
    GAMMA_KPOINTS_SETTINGS,
    SOC_MIN_ATOMIC_NUMBER,
    DopedAimsInputSet,
    aims_competing_phases_singlepoint_settings,
    default_competing_phases_relax_set,
)
from doped.io.aims.utils import _resolve_species_defaults


class AimsCompetingPhasesMixin:
    r"""
    Mixin providing ``FHI-aims`` input-generation methods for
    :class:`~doped.chemical_potentials.CompetingPhases`.
    """

    def get_aims_relaxation_sets(
        self,
        k_grid_density_metals: float = 6.0,
        k_grid_density_nonmetals: float = 3.0,
        user_parameters: dict[str, Any] | None = None,
        species_defaults: PathLike | None = None,
        extrinsic_only: bool = False,
        output_path: PathLike = "CompetingPhases",
        subfolder: PathLike = "Relax",
    ) -> dict[str, DopedAimsInputSet]:
        r"""
        Generates ``DopedAimsInputSet``\s for relaxations of the competing
        phases, using ``FHI-aims``.

        This is the ``FHI-aims`` analogue of :meth:`get_relaxation_sets`, and
        mirrors its default behaviour (automatically switching to
        Methfessel-Paxton smearing for metallic entries, and Γ-only k-points
        with a fixed unit cell for molecule-in-a-box entries) as closely as
        possible given the differences between ``VASP`` and ``FHI-aims``. Note
        that any changes to the default settings should be consistent with
        those used for the defect supercell calculations (see
        :mod:`doped.io.aims.inputs`).

        Args:
            k_grid_density_metals (float):
                ``k_grid_density`` (see the ``FHI-aims`` manual) to use for
                metallic entries (those with zero band gap). Default is 6.0.
            k_grid_density_nonmetals (float):
                ``k_grid_density`` to use for non-metallic entries (those with
                non-zero band gap). Default is 3.0.
            user_parameters (dict):
                Override the default ``control.in`` parameters e.g.
                ``{"xc": "pbesol", "sc_accuracy_etot": 1e-6}``. See
                ``doped/io/aims/AIMS_sets/AIMS_CompetingPhasesSet.yaml`` for
                the default settings. Note that ``xc`` (functional) is not set
                by default and must be specified here.
            species_defaults (PathLike):
                Full path to the ``FHI-aims`` directory containing the element
                default files (e.g. ``.../defaults_2020/tight``).
                Alternatively, ``"light"``, ``"tight"``, or ``"really_tight"``
                uses the corresponding ``defaults_2020`` subdirectory of
                ``pymatgen``'s configured ``AIMS_SPECIES_DIR``. See
                :class:`doped.io.aims.inputs.DefectRelaxSet` for further
                details.
            extrinsic_only (bool):
                If ``True``, only generate inputs for
                ``self.extrinsic_entries`` (useful when adding dopants to an
                existing intrinsic competing-phases set). Default is ``False``
                (generate inputs for all entries).
            output_path (PathLike):
                Top-level output directory name (used as a key prefix).
                Default is ``"CompetingPhases"``.
            subfolder (PathLike):
                Output folder structure is
                ``<output_path>/<competing_phase_dir>/<subfolder>``.
                Default is ``"Relax"``. Set to ``"."`` to write input files
                directly to ``<output_path>/<competing_phase_dir>``, with no
                subfolders created.

        Returns:
            dict[str, DopedAimsInputSet]:
                Mapping of output folder paths to generated
                ``DopedAimsInputSet``\s.
        """
        from doped.chemical_potentials import _get_competing_phase_folder_name

        base_parameters = copy.deepcopy(default_competing_phases_relax_set)
        base_parameters.update(user_parameters or {})
        dict_sets: dict[str, DopedAimsInputSet] = {}
        extrinsic_entries = getattr(self, "extrinsic_entries", [])

        for entry, category, structure in self._iter_entries_with_categories():
            if extrinsic_only and entry not in extrinsic_entries:
                continue
            if structure.properties.get("_is_nominal_structure", False):
                warnings.warn(
                    f"No structure is available for '{entry.name}', so FHI-aims input files "
                    f"cannot be generated for this entry (skipping). Generate/provide a structure "
                    f"for this composition to include it."
                )
                continue

            parameters = copy.deepcopy(base_parameters)
            if category == "molecules":
                kpoints_settings = dict(GAMMA_KPOINTS_SETTINGS)
            else:
                kpoints_settings = {
                    "k_grid_density": k_grid_density_metals
                    if category == "metals"
                    else k_grid_density_nonmetals
                }
                if parameters.get("relax_geometry") != "none":  # not a singlepoint calculation
                    parameters["relax_unit_cell"] = "full"

            self._set_aims_spin_polarisation(parameters, user_parameters or {}, entry, structure)
            if category == "metals":
                self._set_default_aims_metal_smearing(parameters, user_parameters or {})

            parameters.update(kpoints_settings)
            if species_defaults is not None:
                if "species_dir" in parameters:
                    raise ValueError(
                        "Specify either species_defaults or user_parameters['species_dir'], not both."
                    )
                parameters["species_dir"] = _resolve_species_defaults(species_defaults)

            dict_set = DopedAimsInputSet(parameters=parameters, structure=structure)
            fname = f"{output_path}/{_get_competing_phase_folder_name(entry)}/{subfolder}"
            dict_sets[fname] = dict_set

        return dict_sets

    def write_aims_relaxation_files(
        self,
        k_grid_density_metals: float = 6.0,
        k_grid_density_nonmetals: float = 3.0,
        user_parameters: dict[str, Any] | None = None,
        species_defaults: PathLike | None = None,
        extrinsic_only: bool = False,
        output_path: PathLike = "CompetingPhases",
        subfolder: PathLike = "Relax",
        **kwargs,
    ) -> dict[str, DopedAimsInputSet]:
        r"""
        Generates and writes ``FHI-aims`` input files for relaxations of the
        competing phases (see :meth:`get_aims_relaxation_sets` for details of
        the default settings used).

        Args:
            k_grid_density_metals (float):
                ``k_grid_density`` to use for metallic entries (those with
                zero band gap). Default is 6.0.
            k_grid_density_nonmetals (float):
                ``k_grid_density`` to use for non-metallic entries (those with
                non-zero band gap). Default is 3.0.
            user_parameters (dict):
                Override the default ``control.in`` parameters. See
                ``doped/io/aims/AIMS_sets/AIMS_CompetingPhasesSet.yaml`` for
                the default settings. Note that ``xc`` (functional) is not set
                by default and must be specified here.
            species_defaults (PathLike):
                Full path to (or shorthand for) the ``FHI-aims`` species
                defaults directory. See :meth:`get_aims_relaxation_sets`.
            extrinsic_only (bool):
                If ``True``, only generate/write inputs for
                ``self.extrinsic_entries``. Default is ``False``.
            output_path (PathLike):
                Top-level output directory name. Default is
                ``"CompetingPhases"``.
            subfolder (PathLike):
                Output folder structure is
                ``<output_path>/<competing_phase_dir>/<subfolder>``.
                Default is ``"Relax"``.
            **kwargs:
                Additional kwargs to pass to ``DopedAimsInputSet.write_input()``
                (e.g. ``rattle``).

        Returns:
            dict[str, DopedAimsInputSet]:
                Mapping of output folder paths to generated
                ``DopedAimsInputSet``\s.
        """
        dict_sets = self.get_aims_relaxation_sets(
            k_grid_density_metals=k_grid_density_metals,
            k_grid_density_nonmetals=k_grid_density_nonmetals,
            user_parameters=user_parameters,
            species_defaults=species_defaults,
            extrinsic_only=extrinsic_only,
            output_path=output_path,
            subfolder=subfolder,
        )
        return self._write_competing_phase_aims_input_sets(dict_sets, **kwargs)

    def get_aims_singlepoint_sets(
        self,
        k_grid_density_metals: float = 6.0,
        k_grid_density_nonmetals: float = 3.0,
        soc: bool | None = None,
        user_parameters: dict[str, Any] | None = None,
        species_defaults: PathLike | None = None,
        extrinsic_only: bool = False,
        output_path: PathLike = "CompetingPhases",
        subfolder: PathLike | None = None,
    ) -> dict[str, DopedAimsInputSet]:
        r"""
        Generates ``DopedAimsInputSet``\s for single-point (static) energy
        calculations of the competing phases (i.e. no geometry relaxation),
        using ``FHI-aims``. This is the ``FHI-aims`` analogue of
        :meth:`get_singlepoint_sets`.

        These are expected to be used as the final energy calculation after
        geometry relaxation (from :meth:`write_aims_relaxation_files`), to
        obtain accurate total energies with tight convergence settings.

        If ``soc=True``, spin-orbit coupling (SOC) is included (by setting
        ``include_spin_orbit = pauli`` for ``FHI-aims``) and the output
        subfolder name will default to ``"aims_ncl"`` (if not set otherwise).
        If ``soc`` is not explicitly set (i.e. is ``None``), it defaults to
        ``True`` for systems where the max atomic number across all species
        (host and extrinsic) is Z >= 31 (heavier than Zn), matching the
        convention in :mod:`doped.io.aims.inputs`.

        Args:
            k_grid_density_metals (float):
                ``k_grid_density`` to use for metallic entries (those with
                zero band gap). Default is 6.0.
            k_grid_density_nonmetals (float):
                ``k_grid_density`` to use for non-metallic entries (those with
                non-zero band gap). Default is 3.0.
            soc (bool):
                Whether to include spin-orbit coupling (SOC), by setting
                ``include_spin_orbit = pauli`` in the ``FHI-aims``
                ``control.in``. If not set (when ``soc = None``; default), SOC
                is enabled when the max atomic number across all species
                (host and extrinsic) is Z >= 31. The default ``subfolder``
                name is set to ``"aims_ncl"`` when ``soc`` is ``True``.
            user_parameters (dict):
                Override the default ``control.in`` parameters. See
                ``doped/io/aims/AIMS_sets/AIMS_CompetingPhasesSet.yaml`` for
                the default settings. Note that ``xc`` (functional) is not set
                by default and must be specified here.
            species_defaults (PathLike):
                Full path to (or shorthand for) the ``FHI-aims`` species
                defaults directory. See :meth:`get_aims_relaxation_sets`.
            extrinsic_only (bool):
                If ``True``, only generate inputs for
                ``self.extrinsic_entries``. Default is ``False``.
            output_path (PathLike):
                Top-level output directory name (used as a key prefix).
                Default is ``"CompetingPhases"``.
            subfolder (PathLike):
                Output folder structure is
                ``<output_path>/<competing_phase_dir>/<subfolder>`` where
                ``subfolder`` = ``"SinglePoint"`` by default if ``soc`` is
                ``False``, or ``"aims_ncl"`` if ``soc`` is ``True``.

        Returns:
            dict[str, DopedAimsInputSet]:
                Mapping of output folder paths to generated
                ``DopedAimsInputSet``\s.
        """
        if soc is None:
            all_elements = self.intrinsic_elements + getattr(self, "extrinsic_elements", [])
            max_Z = max(Element(el).Z for el in all_elements)
            soc = max_Z >= SOC_MIN_ATOMIC_NUMBER

            if soc:  # if SOC being automatically determined, print an info message
                print(
                    "Spin-orbit coupling (SOC) is being used by default for competing phase "
                    "single-point calculations, as the heaviest element present (across intrinsic "
                    "and extrinsic species) has an atomic number Z >= 31 -- consistent with the "
                    "convention in `DefectsSet`. Set `soc` explicitly to control this behaviour "
                    "(and suppress this message). As always, consistent settings with the defect "
                    "supercell calculations should be used for the final single-point energy "
                    "calculations.",
                )

        sp_parameters = copy.deepcopy(aims_competing_phases_singlepoint_settings)
        if soc:
            sp_parameters["include_spin_orbit"] = "pauli"
        sp_parameters.update(user_parameters or {})

        if subfolder is None:
            subfolder = "aims_ncl" if soc else "SinglePoint"

        # reuse relaxation set generation with the singlepoint parameter overrides
        return self.get_aims_relaxation_sets(
            k_grid_density_metals=k_grid_density_metals,
            k_grid_density_nonmetals=k_grid_density_nonmetals,
            user_parameters=sp_parameters,
            species_defaults=species_defaults,
            extrinsic_only=extrinsic_only,
            output_path=output_path,
            subfolder=subfolder,
        )

    def write_aims_singlepoint_files(
        self,
        k_grid_density_metals: float = 6.0,
        k_grid_density_nonmetals: float = 3.0,
        soc: bool | None = None,
        user_parameters: dict[str, Any] | None = None,
        species_defaults: PathLike | None = None,
        extrinsic_only: bool = False,
        output_path: PathLike = "CompetingPhases",
        subfolder: PathLike | None = None,
        geometry: bool = False,
        **kwargs,
    ) -> dict[str, DopedAimsInputSet]:
        r"""
        Generates and writes ``FHI-aims`` input files for single-point
        (static) energy calculations of the competing phases (see
        :meth:`get_aims_singlepoint_sets` for details of the default settings
        used).

        Args:
            k_grid_density_metals (float):
                ``k_grid_density`` to use for metallic entries. Default is
                6.0.
            k_grid_density_nonmetals (float):
                ``k_grid_density`` to use for non-metallic entries. Default is
                3.0.
            soc (bool):
                Whether to include spin-orbit coupling (SOC). See
                :meth:`get_aims_singlepoint_sets`.
            user_parameters (dict):
                Override the default ``control.in`` parameters. Note that
                ``xc`` (functional) is not set by default and must be
                specified here.
            species_defaults (PathLike):
                Full path to (or shorthand for) the ``FHI-aims`` species
                defaults directory. See :meth:`get_aims_relaxation_sets`.
            extrinsic_only (bool):
                If ``True``, only generate/write inputs for
                ``self.extrinsic_entries``. Default is ``False``.
            output_path (PathLike):
                Top-level output directory name. Default is
                ``"CompetingPhases"``.
            subfolder (PathLike):
                See :meth:`get_aims_singlepoint_sets`.
            geometry (bool):
                Whether to write ``geometry.in`` files. Defaults to ``False``,
                as single-point (static) calculations are intended to be run
                on `relaxed` structures (e.g. the final ``geometry.in.next_step``,
                or the last "Updated atomic structure" in ``aims.out``, from
                the corresponding :meth:`write_aims_relaxation_files`
                calculation), so using the (unrelaxed) input structure is
                typically undesirable; copy the relaxed geometry in manually
                as ``geometry.in`` in each output folder before running.
            **kwargs:
                Additional kwargs to pass to ``DopedAimsInputSet.write_input()``
                (e.g. ``rattle``).

        Returns:
            dict[str, DopedAimsInputSet]:
                Mapping of output folder paths to generated
                ``DopedAimsInputSet``\s.
        """
        dict_sets = self.get_aims_singlepoint_sets(
            k_grid_density_metals=k_grid_density_metals,
            k_grid_density_nonmetals=k_grid_density_nonmetals,
            soc=soc,
            user_parameters=user_parameters,
            species_defaults=species_defaults,
            extrinsic_only=extrinsic_only,
            output_path=output_path,
            subfolder=subfolder,
        )
        return self._write_competing_phase_aims_input_sets(dict_sets, geometry=geometry, **kwargs)

    def _write_competing_phase_aims_input_sets(
        self, dict_sets: dict[str, DopedAimsInputSet], geometry: bool = True, **kwargs
    ) -> dict[str, DopedAimsInputSet]:
        r"""
        Write a dictionary of ``DopedAimsInputSet``\s to their corresponding
        output folders, warning if any already exist.

        ``geometry.in`` writing is controlled by the ``geometry`` argument
        (default ``True``); set to ``False`` (as done by default in
        :meth:`write_aims_singlepoint_files`) to skip writing ``geometry.in``,
        e.g. when a relaxed geometry must be manually copied in instead.
        """
        for fname, dict_set in dict_sets.items():
            if os.path.exists(fname):
                warnings.warn(f"Output folder {fname} already exists. Overwriting files.")
            geometry_in = dict_set.inputs.pop("geometry.in", None) if not geometry else None
            try:
                dict_set.write_input(fname, **kwargs)
            finally:
                if geometry_in is not None:
                    dict_set.inputs["geometry.in"] = geometry_in

        return dict_sets

    def _set_aims_spin_polarisation(
        self,
        parameters: dict,
        user_parameters: dict,
        entry: ComputedEntry,
        structure: Structure,
    ) -> None:
        """
        ``FHI-aims`` analogue of :meth:`_set_spin_polarisation`.

        If the entry has a non-zero total magnetization (greater than the
        default tolerance of 0.1), set ``spin`` to ``"collinear"`` and
        ``default_initial_moment`` to the total magnetization divided evenly
        over the number of sites in ``structure`` (as a rough starting guess;
        see the ``FHI-aims`` manual's ``default_initial_moment`` entry for
        caveats of this blanket approach).

        Otherwise ``spin`` is not set, so spin polarisation is not allowed (as
        typically desired for non-magnetic phases, for efficiency).
        """
        magnetization = entry.data.get("summary", {}).get("total_magnetization")
        if magnetization is not None and magnetization > 0.1:  # account for magnetic moment
            parameters["spin"] = user_parameters.get("spin", "collinear")
            if "default_initial_moment" not in parameters and magnetization > 0:
                parameters["default_initial_moment"] = magnetization / max(len(structure), 1)

        # otherwise spin not set, so no spin polarisation

    def _set_default_aims_metal_smearing(self, parameters: dict, user_parameters: dict) -> None:
        """
        ``FHI-aims`` analogue of :meth:`_set_default_metal_smearing`.

        Set the smearing parameters to the ``doped`` defaults for metallic
        phases (i.e. Methfessel-Paxton smearing with a width of 0.2 eV).
        """
        parameters["occupation_type"] = user_parameters.get(
            "occupation_type", ["methfessel-paxton", 0.2, 1]
        )
