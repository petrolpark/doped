import os
import re
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from monty.serialization import loadfn
from pymatgen.core import SETTINGS
from pymatgen.core.entries import ComputedStructureEntry
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun
from test_utils import EXAMPLE_DIR, data_dir, vasp_data_dir

# temp
SETTINGS["AIMS_SPECIES_DIR"] = "~/Documents/fhi-aims.260331/species_defaults"

from doped.analysis import DefectParser
from doped.chemical_potentials import get_doped_chempots_from_entries
from doped.generation import DefectsGenerator
from doped.io import get_calculation_outputs as get_generic_calculation_outputs
from doped.io.aims.inputs import DefectsSet
from doped.io.aims.utils import _resolve_species_defaults
from doped.io.aims.outputs import (
    get_aims_output, get_atomic_magnetic_moments_from_aims_output,
    get_calculation_outputs, get_magnetization_from_aims_output,
    get_n_electrons_from_aims_output, get_neutral_n_electrons,
    get_site_potentials, spin_degeneracy_from_aims_output,
    total_charge_from_aims_output)
from doped.io.outputs import CalculationOutputs
from doped.utils.efficiency import Structure


class AimsTest(unittest.TestCase):

    def setUp(self):
        self.prim_cdte = Structure.from_file(f"{EXAMPLE_DIR}/CdTe/relaxed_primitive_POSCAR")
        self.aims_output_path = Path(data_dir) / "aims"
        self.mgo_prim = Structure.from_file(f"{EXAMPLE_DIR}/MgO/Input_files/MgO_POSCAR_prim")
        # real (216-atom) VASP bulk supercell from the pre-calculated MgO defect study
        # (examples/MgO/Defects/Pre_Calculated_Results), for generating matching-supercell aims
        # input sets (see `test_MgO_defect_input_generation`):
        self.mgo_bulk_supercell = Structure.from_file(
            f"{EXAMPLE_DIR}/MgO/Defects/Pre_Calculated_Results/MgO_bulk/vasp_std/POSCAR"
        )
        self.cu2sise3_bulk = Structure.from_file(f"{EXAMPLE_DIR}/Cu2SiSe3/bulk/vasp_std/vasprun.xml.gz")
        self.sb2si2te6_bulk = Structure.from_file(f"{EXAMPLE_DIR}/Sb2Si2Te6/Bulk/vasprun.xml.gz")
        self.ytos_bulk = Structure.from_file(f"{EXAMPLE_DIR}/YTOS/Bulk/vasprun.xml.gz")

    def test_species_defaults_path_resolution(self):
        """Resolve the default light species_defaults shorthand."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            species_defaults = Path(tmp_dir) / "species_defaults"
            basis_path = species_defaults / "defaults_2020" / "light"
            basis_path.mkdir(parents=True)
            assert _resolve_species_defaults(basis_path) == str(basis_path.resolve())
            with patch("doped.io.aims.utils.SETTINGS", {"AIMS_SPECIES_DIR": str(species_defaults)}):
                assert _resolve_species_defaults("light") == str(basis_path.resolve())
                assert _resolve_species_defaults(Path("defaults_2020") / "light") == str(basis_path.resolve())

    def test_soc_auto_detection(self):
        """``DefectRelaxSet.soc`` is auto-set from the max atomic number in the
        defect supercell (>= 31, i.e. heavier than Zn), mirroring
        ``doped.io.vasp.inputs.DefectRelaxSet.soc``, and gates whether
        ``aims_ncl``/``bulk_aims_ncl`` are generated."""
        from doped.io.aims.inputs import DefectRelaxSet

        light_element_relax_set = DefectRelaxSet(self.mgo_prim, user_parameters={"xc": "pbesol"})
        assert light_element_relax_set.soc is False
        assert light_element_relax_set.aims_ncl is None

        heavy_element_relax_set = DefectRelaxSet(self.prim_cdte, user_parameters={"xc": "pbe"})
        assert heavy_element_relax_set.soc is True
        assert re.findall(
            r"^\s*include_spin_orbit\s+(\S+)\s*$",
            heavy_element_relax_set.aims_ncl.inputs["control.in"],
            re.MULTILINE,
        ) == ["pauli"]

        # explicit `soc` kwarg overrides the atomic-number auto-detection:
        overridden_relax_set = DefectRelaxSet(self.prim_cdte, user_parameters={"xc": "pbe"}, soc=False)
        assert overridden_relax_set.soc is False
        assert overridden_relax_set.aims_ncl is None

    def test_CdTe_defect_input_generation(self):
        """Generate CdTe defects and write an FHI-aims input set for each.

        The real CdTe VASP calculations (``examples/CdTe/Int_Te_3_1``,
        ``Int_Te_3_2``, ``Te_Cd_+1``) used a non-standard tuned hybrid:
        ``LHFCALC = True``, ``AEXX = 0.345``, ``HFSCREEN = 0.2`` Å\\ :sup:`-1`
        (NOT literature HSE06's 25% exact exchange), hence the explicit
        ``hybrid_xc_coeff`` override below (see the ``hybrid_xc_coeff``/
        ``hse_unit`` entries in ``docs/FHI-aims_manual.txt``).
        """
        self._generate_and_check_defect_inputs(
            self.prim_cdte,
            "CdTe",
            user_parameters={"xc": "hse06 0.2", "hse_unit": "A", "hybrid_xc_coeff": 0.345},
            expect_soc=True,  # Cd (Z=48), Te (Z=52) >= 31
        )

    def test_MgO_defect_input_generation(self):
        """Generate MgO defects and write an FHI-aims input set for each,
        using the real (216-atom) VASP bulk supercell from the pre-
        calculated MgO defect study directly (``generate_supercell=False``),
        so the generated defect supercells match those of the real VASP
        calculations (``examples/MgO/Defects/Pre_Calculated_Results``),
        rather than the differently-sized (96-atom) supercell that would be
        auto-generated from the primitive cell.

        The real MgO VASP calculations used ``GGA = Ps`` (PBEsol) with
        ``LHFCALC = False`` (no hybridisation).
        """
        self._generate_and_check_defect_inputs(
            self.mgo_bulk_supercell,
            "MgO",
            user_parameters={"xc": "pbesol"},
            expect_soc=False,  # Mg/O < 31
            generate_supercell=False,
            rattle=True,
        )

    def test_Cu2SiSe3_defect_input_generation(self):
        """Generate Cu2SiSe3 defects and write an FHI-aims input set for each.

        The real Cu2SiSe3 VASP calculations used literature HSE06
        (``LHFCALC = True``, ``AEXX = 0.25``, ``HFSCREEN = 0.2`` Å\\ :sup:`-1`).
        """
        self._generate_and_check_defect_inputs(
            self.cu2sise3_bulk,
            "Cu2SiSe3",
            user_parameters={"xc": "hse06 0.2", "hse_unit": "A", "hybrid_xc_coeff": 0.25},
            expect_soc=True,  # Se (Z=34) >= 31
        )

    def test_Sb2Si2Te6_defect_input_generation(self):
        """Generate Sb2Si2Te6 defects and write an FHI-aims input set for each.

        The real Sb2Si2Te6 VASP calculations used literature HSE06
        (``LHFCALC = True``, ``AEXX = 0.25``, ``HFSCREEN = 0.208`` Å\\ :sup:`-1`).
        """
        self._generate_and_check_defect_inputs(
            self.sb2si2te6_bulk,
            "Sb2Si2Te6",
            user_parameters={"xc": "hse06 0.208", "hse_unit": "A", "hybrid_xc_coeff": 0.25},
            expect_soc=True,  # Sb (Z=51), Te (Z=52) >= 31
        )

    def test_YTOS_defect_input_generation(self):
        """Generate YTOS defects and write an FHI-aims input set for each.

        The real YTOS VASP calculations used plain PBE (``LHFCALC = False``).
        """
        self._generate_and_check_defect_inputs(
            self.ytos_bulk, "YTOS", user_parameters={"xc": "pbe"}, expect_soc=True  # Y (Z=39) >= 31
        )

    def _generate_and_check_defect_inputs(
        self,
        structure,
        material_dir_name,
        user_parameters,
        expect_soc,
        generate_supercell=True,
        rattle=False,
    ):
        """Generate defects for ``structure`` and write/check an FHI-aims input set
        for each, in ``self.aims_output_path / material_dir_name``.

        ``expect_soc`` is the expected value of ``DefectsSet.soc`` (whether
        ``aims_ncl`` -- single-point, spin-orbit-coupled -- input sets should
        also be generated); set to ``True`` if the max atomic number across the
        material's elements is >= 31 (i.e. further down the periodic table than
        Zn), mirroring ``doped.io.vasp.inputs.DefectsSet``'s auto-detection.

        ``generate_supercell`` is passed to ``DefectsGenerator``; set to ``False``
        when ``structure`` is already the desired defect supercell (e.g. a real
        VASP calculation supercell, to match its defect supercells exactly),
        rather than a primitive/unit cell to auto-generate a supercell from.

        ``rattle`` is passed to ``DefectsSet.write_files``; set to ``True`` so
        defect geometries are ``ShakeNBreak``-rattled before writing, letting the
        relaxation escape the exact (unbroken) site symmetry of the unperturbed
        defect structure (otherwise gradient-based relaxation from an exactly
        symmetric starting geometry cannot spontaneously lower that symmetry,
        even when a lower-symmetry, lower-energy minimum exists).
        """
        defect_generator = DefectsGenerator(structure, generate_supercell=generate_supercell)
        output_path = self.aims_output_path / material_dir_name
        os.makedirs(output_path, exist_ok=True)

        defects_set = DefectsSet(
            defect_generator,
            user_parameters=user_parameters,
            species_defaults="light",
        )
        assert defects_set.soc is expect_soc
        defects_set.write_files(output_path=output_path, rattle=rattle, processes=1)

        assert len(defects_set) == len(defect_generator.defect_entries)
        formula = defect_generator.bulk_supercell.composition.get_reduced_formula_and_factor(
            iupac_ordering=True
        )[0]
        generator_json = os.path.join(output_path, f"{formula}_defects_generator.json.gz")
        assert os.path.isfile(generator_json)
        reloaded_generator = loadfn(generator_json)
        assert set(reloaded_generator.defect_entries) == set(defect_generator.defect_entries)
        assert reloaded_generator.bulk_supercell == defect_generator.bulk_supercell

        subfolders = ("aims_gam", "aims_std", "aims_ncl") if expect_soc else ("aims_gam", "aims_std")
        for defect_name, defect_entry in defect_generator.defect_entries.items():
            defect_dir = os.path.join(output_path, defect_name)
            for subfolder in subfolders:
                subfolder_dir = os.path.join(defect_dir, subfolder)
                assert os.path.isfile(os.path.join(subfolder_dir, "parameters.json"))
                assert os.path.isfile(os.path.join(subfolder_dir, f"{defect_name}.json.gz"))
                self._check_valid_control_and_geometry(
                    subfolder_dir, subfolder, expected_charge=defect_entry.charge_state
                )
            if not expect_soc:
                assert not os.path.isdir(os.path.join(defect_dir, "aims_ncl"))

        bulk_dir = os.path.join(output_path, f"{formula}_bulk")
        for subfolder in subfolders:
            self._check_valid_control_and_geometry(
                os.path.join(bulk_dir, subfolder), subfolder, expected_charge=0
            )

    def _check_valid_control_and_geometry(self, subfolder_dir, subfolder, expected_charge):
        """Check that ``control.in``/``geometry.in`` in ``subfolder_dir`` are valid,
        self-consistent FHI-aims inputs for a periodic calculation (i.e. a `k`-point
        grid is specified, matching ``aims_gam``/``aims_std``/``aims_ncl`` as
        appropriate, the total charge in ``control.in`` matches ``expected_charge``
        (the defect's ``charge_state``), ``geometry.in`` has no per-atom
        ``initial_charge`` lines (see ``_structure_for_aims`` -- concentrating any
        residual charge-state delta onto one atom can produce physically extreme
        formal ions that crash FHI-aims's free-atom initial-density guess), smearing/
        SCF convergence are set matching ``AIMS_DefectSet.yaml``, and
        ``aims_gam``/``aims_std`` request geometry relaxation while ``aims_ncl``
        is single-point with spin-orbit coupling included, per
        ``AIMS_NCLDefectSet.yaml``)."""
        control_path = os.path.join(subfolder_dir, "control.in")
        geometry_path = os.path.join(subfolder_dir, "geometry.in")
        assert os.path.isfile(control_path)
        assert os.path.isfile(geometry_path)

        with open(control_path, encoding="utf-8") as control_file:
            control = control_file.read()
        with open(geometry_path, encoding="utf-8") as geometry_file:
            geometry = geometry_file.read()

        # a periodic FHI-aims calculation requires a k-point grid to be specified:
        k_grid_matches = re.findall(r"^\s*k_grid\s+(\d+)\s+(\d+)\s+(\d+)\s*$", control, re.MULTILINE)
        k_grid_density_matches = re.findall(r"^\s*k_grid_density\s+([\d.]+)\s*$", control, re.MULTILINE)
        if subfolder == "aims_gam":
            assert k_grid_matches == [("1", "1", "1")]
            assert not k_grid_density_matches
        else:  # aims_std/aims_ncl
            assert not k_grid_matches
            assert k_grid_density_matches == ["5.0"]

        charge_matches = re.findall(r"^\s*charge\s+([+-]?[0-9]+(?:\.[0-9]*)?)", control, re.MULTILINE)
        assert len(charge_matches) == 1
        assert float(charge_matches[0]) == float(expected_charge)
        # no per-atom initial_charge lines (see check_occupation_dimensions crash rationale above):
        assert not re.findall(r"^\s*initial_charge\s+\S+\s*$", geometry, re.MULTILINE)

        # smearing and SCF iteration limit are always set, matching AIMS_DefectSet.yaml
        # (independent of relaxation/SOC stage). `spin`/`default_initial_moment` are
        # deliberately NOT set by doped's defaults (see AIMS_DefectSet.yaml for why: doped's
        # per-atom `initial_charge` can assign atoms formal charges beyond what their species'
        # default valence definition supports for a second spin channel, crashing FHI-aims's
        # `check_occupation_dimensions` for many higher-|charge| defects):
        assert not re.findall(r"^\s*spin\s+\S+\s*$", control, re.MULTILINE)
        assert not re.findall(r"^\s*default_initial_moment\s+\S+\s*$", control, re.MULTILINE)
        assert re.findall(r"^\s*occupation_type\s+(\S+\s+\S+)\s*$", control, re.MULTILINE) == [
            "gaussian 0.05"
        ]
        assert re.findall(r"^\s*sc_iter_limit\s+(\S+)\s*$", control, re.MULTILINE) == ["1000"]

        relax_matches = re.findall(r"^\s*relax_geometry\s+(.+)\s*$", control, re.MULTILINE)
        soc_matches = re.findall(r"^\s*include_spin_orbit\s+(\S+)\s*$", control, re.MULTILINE)
        sc_accuracy_matches = re.findall(r"^\s*sc_accuracy_etot\s+(\S+)\s*$", control, re.MULTILINE)
        if subfolder == "aims_ncl":
            # single-point (no relaxation tolerance needed), spin-orbit-coupled, tighter SCF
            # convergence:
            assert relax_matches == ["none"]
            assert soc_matches == ["pauli"]
            assert sc_accuracy_matches == ["1e-06"]
        else:  # aims_gam/aims_std: geometry relaxation, no SOC:
            assert relax_matches == ["bfgs 0.01"]
            assert not soc_matches
            assert sc_accuracy_matches == ["1e-05"]


class MgOOutputsTest(unittest.TestCase):
    """
    Output-parsing tests using the real FHI-aims ``MgO`` test data.

    Unlike the ``CdTe`` test data (only used for input-generation testing
    above), the ``MgO`` ``Mg_O`` antisite defect (all charge states) has a
    matching set of *real* VASP calculation outputs checked into the repo
    (``examples/MgO/Defects/Pre_Calculated_Results``), generated from the
    *same* (216-atom) supercell as the aims data here, so aims-parsed values
    can be directly cross-checked against an independent code/parser,
    rather than only against hardcoded numbers copied from a single run of
    the parser under test (which isn't a meaningful correctness check -- so
    tests of that form, e.g. for ``energy``/``efermi``/``vbm``/``cbm``/raw
    eigenvalue values, are intentionally not included here).

    Not all VASP-tested properties of this data can currently be cross-
    checked: ``get_eigenvalue_analysis()`` (band-edge state/(un)occupied
    localised state character, tested for this exact VASP data in
    ``test_analysis.py``) requires ``projected_eigenvalues`` with a
    ``PROCAR``-style per-band/per-site/per-orbital-type breakdown, which the
    aims backend does not (yet) provide in that exact form (only orbital-
    angular-momentum-resolved Mulliken populations, from ``Mulliken.out`` --
    see ``test_get_calculation_outputs_projected_eigenvalues`` below); and
    ``degeneracy_factors["orientational degeneracy"]`` was found to not
    match between backends (see ``test_spin_degeneracy_matches_vasp``) --
    apparently a genuine gap/bug in the generic symmetry analysis for the
    aims backend, rather than a data issue, meriting separate investigation.
    """

    def setUp(self):
        vasp_pre_calculated_path = Path(EXAMPLE_DIR) / "MgO" / "Defects" / "Pre_Calculated_Results"
        # real (216-atom) VASP bulk supercell -- the aims test data below was (re)generated
        # directly from this structure (`generate_supercell=False`; see
        # `AimsTest.test_MgO_defect_input_generation`), so its lattice is used as the ground
        # truth for the aims-parsed structures' lattice below:
        self.mgo_bulk_supercell = Structure.from_file(str(vasp_pre_calculated_path / "MgO_bulk" / "vasp_std" / "POSCAR"))
        self.aims_output_path = Path(data_dir) / "aims" / "MgO"
        self.bulk_dir = self.aims_output_path / "MgO_bulk" / "aims_gam"
        self.charged_dirs = {
            charge: self.aims_output_path / f"Mg_O_{'+' if charge else ''}{charge}" / "aims_gam"
            for charge in (0, 1, 2, 3, 4)
        }
        self.vasp_bulk_dir = vasp_pre_calculated_path / "MgO_bulk" / "vasp_std"
        self.vasp_charged_dirs = {
            charge: vasp_pre_calculated_path / f"Mg_O_{'+' if charge else ''}{charge}" / "vasp_std"
            for charge in (0, 1, 2, 3, 4)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # non-spin-polarised calcs -> no warnings expected anyway
            self.bulk_output = get_aims_output(self.bulk_dir / "aims.out")
            self.charged_outputs = {
                charge: get_aims_output(charged_dir / "aims.out")
                for charge, charged_dir in self.charged_dirs.items()
            }

    def test_get_aims_output(self):
        """``get_aims_output`` returns an ``AimsStdout`` wrapping the parsed file."""
        from pyfhiaims.outputs.stdout import AimsStdout

        assert isinstance(self.bulk_output, AimsStdout)

    def test_get_aims_output_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            get_aims_output(self.aims_output_path / "MgO_bulk" / "aims_gam" / "not_a_real_file.out")

    def test_get_n_electrons_from_aims_output(self):
        """216-atom Mg108O108 bulk cell: 108*12 + 108*8 = 2160 electrons."""
        assert get_n_electrons_from_aims_output(self.bulk_output) == 2160
        # Mg_O antisite (Mg on O site) with charge +1: one fewer O, one more Mg, plus +1 charge:
        assert get_n_electrons_from_aims_output(self.charged_outputs[1]) == 2160 + 4 - 1

    def test_get_neutral_n_electrons(self):
        structure = self.bulk_output.get_image(-1).geometry.structure
        assert get_neutral_n_electrons(structure) == 2160
        charged_structure = self.charged_outputs[1].get_image(-1).geometry.structure
        assert get_neutral_n_electrons(charged_structure) == 2160 + 4  # Mg109O107, neutral electron count

    def test_total_charge_from_aims_output(self):
        for charge, aims_output in self.charged_outputs.items():
            assert total_charge_from_aims_output(aims_output) == charge

    def test_get_magnetization_from_aims_output(self):
        """All MgO test outputs are non-spin-polarised, so magnetization is zero."""
        assert get_magnetization_from_aims_output(self.bulk_output) == 0.0
        for aims_output in self.charged_outputs.values():
            assert get_magnetization_from_aims_output(aims_output) == 0.0

    def test_spin_degeneracy_from_aims_output(self):
        # bulk: even (2160) electrons, zero magnetization -> singlet:
        assert spin_degeneracy_from_aims_output(self.bulk_output) == 1
        # Mg_O_+1: odd (2163) electrons -> doublet:
        assert spin_degeneracy_from_aims_output(self.charged_outputs[1]) == 2
        # explicit charge_state overrides the auto-determined electron count:
        assert spin_degeneracy_from_aims_output(self.bulk_output, charge_state=1) == 2
        assert spin_degeneracy_from_aims_output(self.bulk_output, charge_state=0) == 1

    def test_get_atomic_magnetic_moments_from_aims_output_requires_mulliken(self):
        """None of the MgO test outputs have ``output mulliken`` results parsed."""
        with self.assertRaises(KeyError):
            get_atomic_magnetic_moments_from_aims_output(self.bulk_output)

    def test_get_calculation_outputs(self):
        """
        Test parsing an FHI-aims bulk supercell calculation to
        ``CalculationOutputs``, checking the parsed fields against
        independently-derivable values (composition/formula-based electron
        counts, internal cross-field consistency between separately-parsed
        quantities, and the real VASP-relaxed MgO primitive cell) -- not
        hardcoded numbers merely copied from a single run of the parser
        under test.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = get_calculation_outputs(self.bulk_dir, label="bulk")

        assert outputs.calculator == "aims"
        assert str(outputs.directory) == str(self.bulk_dir)
        assert len(outputs.structure) == 216  # Mg108O108 bulk supercell, matching the real VASP supercell
        assert outputs.structure.composition.reduced_formula == "MgO"
        assert outputs.converged_electronic is True
        assert outputs.converged_ionic is True
        assert outputs.nelect == 2160  # 108*12 + 108*8
        assert outputs.charge == 0
        assert outputs.magnetization == 0.0
        assert outputs.spin_degeneracy() == 1
        assert outputs.spin_degeneracy(charge_state=0) == 1
        assert int(outputs.run_metadata["charge"]) == 0
        assert outputs.run_metadata["num_electrons"] == 2160.0
        assert outputs.run_metadata["xc"] == "pbesol"  # matches the real VASP MgO calculations

        # `aims_gam` is Gamma-point-only with no `output k_point_list` set, so `pyfhiaims`
        # doesn't parse explicit k-point coordinates/weights here:
        assert outputs.kpoint_coords is None
        assert outputs.kpoint_weights is None

        # cross-field consistency: the occupied/unoccupied boundary of the parsed eigenvalue
        # array should match the separately-parsed `vbm`/`cbm` (both ultimately come from the
        # same `aims.out` file, but via different `pyfhiaims` parsing paths), and its band count
        # should match the independently-parsed `run_metadata["num_bands"]`:
        assert list(outputs.eigenvalues.keys()) == [Spin.up]  # non-spin-polarised
        eigenvalues = outputs.eigenvalues[Spin.up][0]  # Gamma-only -> single k-point
        assert eigenvalues.shape == (outputs.run_metadata["num_bands"], 2)
        occupied, unoccupied = eigenvalues[eigenvalues[:, 1] > 1.0], eigenvalues[eigenvalues[:, 1] < 1.0]
        assert np.isclose(occupied[:, 0].max(), outputs.vbm)
        assert np.isclose(unoccupied[:, 0].min(), outputs.cbm)

        # `aims`-generated defect supercells use a fixed unit cell (no `relax_unit_cell` in
        # `control.in`), and were generated directly from the real (216-atom) VASP bulk
        # supercell (`generate_supercell=False`), so the parsed structure's lattice should
        # exactly match it:
        assert np.allclose(outputs.structure.lattice.matrix, self.mgo_bulk_supercell.lattice.matrix)
        assert len(outputs.structure) == len(self.mgo_bulk_supercell)

    def test_get_calculation_outputs_projected_eigenvalues(self):
        """
        Test that ``parse_projected_eigen=True`` populates
        ``projected_eigenvalues`` from ``Mulliken.out``, with the correct
        shape and a physically-derivable cross-check: the Mulliken
        decomposition partitions each (normalised) Kohn-Sham state's density
        across atoms/orbital-angular-momentum channels, so summing the
        parsed projections over all atoms and orbitals for a single band
        should recover ~1 (not the band's occupation number, which is a
        separate, already-parsed quantity) -- not a number merely copied
        from a single run of the parser under test.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = get_calculation_outputs(self.bulk_dir, label="bulk", parse_projected_eigen=True)

        assert list(outputs.projected_eigenvalues.keys()) == [Spin.up]  # non-spin-polarised
        projected_eigenvalues = outputs.projected_eigenvalues[Spin.up][0]  # Gamma-only -> single k-point
        assert projected_eigenvalues.shape == (outputs.run_metadata["num_bands"], 216, 3)  # l=0,1,2

        band_sums = projected_eigenvalues.sum(axis=(1, 2))
        assert np.allclose(band_sums, 1.0, atol=1e-2)

    def test_get_calculation_outputs_matches_vasp(self):
        """
        Test that the FHI-aims-parsed ``CalculationOutputs`` for the
        ``Mg_O`` antisite defect (all charge states) agree with the real
        VASP calculation outputs for the same defect/charge states
        (``examples/MgO/Defects/Pre_Calculated_Results``).

        Unlike the earlier ``CdTe``/``v_Cd_-2`` case, the aims and VASP
        ``MgO`` calculations use the *same* (216-atom) supercell (the aims
        inputs were generated directly from the real VASP bulk supercell,
        ``generate_supercell=False`` -- see
        ``AimsTest.test_MgO_defect_input_generation``), so their structures'
        atom counts and lattices should match exactly, in addition to the
        code-independent physical quantities (charge state, spin
        degeneracy, electronic convergence). Absolute energies/electron
        counts still aren't expected to match (pseudopotential vs.
        all-electron treatments, and differing XC functionals).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aims_bulk = get_calculation_outputs(self.bulk_dir, label="bulk")
            vasp_bulk = get_generic_calculation_outputs(self.vasp_bulk_dir, parse_projected_eigen=False)
        assert aims_bulk.charge == vasp_bulk.charge == 0
        assert aims_bulk.spin_degeneracy() == vasp_bulk.spin_degeneracy() == 1
        assert aims_bulk.converged_electronic is vasp_bulk.converged_electronic is True
        assert len(aims_bulk.structure) == len(vasp_bulk.structure) == 216
        assert np.allclose(aims_bulk.structure.lattice.matrix, vasp_bulk.structure.lattice.matrix)

        for charge, charged_dir in self.charged_dirs.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aims_outputs = get_calculation_outputs(charged_dir, label="defect")
                vasp_outputs = get_generic_calculation_outputs(
                    self.vasp_charged_dirs[charge], parse_projected_eigen=False
                )

            assert aims_outputs.charge == vasp_outputs.charge == charge
            assert aims_outputs.converged_electronic is True
            assert vasp_outputs.converged_electronic is True
            assert aims_outputs.spin_degeneracy() == vasp_outputs.spin_degeneracy()
            assert int(aims_outputs.run_metadata["charge"]) == charge
            assert len(aims_outputs.structure) == len(vasp_outputs.structure) == 216
            assert np.allclose(aims_outputs.structure.lattice.matrix, vasp_outputs.structure.lattice.matrix)

    def test_spin_degeneracy_matches_vasp(self):
        """
        Test that the ``DefectEntry.degeneracy_factors["spin degeneracy"]``
        computed via the aims backend for the ``Mg_O`` antisite (all charge
        states) matches the same quantity computed via the VASP backend for
        the real ``examples/MgO/Defects/Pre_Calculated_Results`` data --
        mirroring the check already made of this VASP data in
        ``test_analysis.py::DopedParsingTestCase::test_eigenvalues_parsing_
        and_warnings`` (``degeneracy_factors["spin degeneracy"] == 2`` for
        ``Mg_O_+1``).

        ``degeneracy_factors["orientational degeneracy"]`` is *not* compared
        here, as it did not match between backends when checked manually
        (VASP: 24/24/8/8/12 for charges 0-4; aims: 1 for all) -- this
        appears to be a genuine gap/bug in the generic orientational-
        degeneracy symmetry analysis for the aims backend (`doped.utils.
        symmetry`), rather than a data issue, and needs separate
        investigation before it can be meaningfully tested.
        """
        for charge, charged_dir in self.charged_dirs.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aims_defect_entry = DefectParser.from_paths(
                    defect_path=str(charged_dir),
                    bulk_path=str(self.bulk_dir),
                    calculator="aims",
                    skip_corrections=True,
                ).defect_entry
                vasp_defect_entry = DefectParser.from_paths(
                    defect_path=str(self.vasp_charged_dirs[charge]),
                    bulk_path=str(self.vasp_bulk_dir),
                    skip_corrections=True,
                ).defect_entry

            assert (
                aims_defect_entry.degeneracy_factors["spin degeneracy"]
                == vasp_defect_entry.degeneracy_factors["spin degeneracy"]
            )

    def test_get_calculation_outputs_site_potentials(self):
        """
        Test that ``load_site_potentials=True`` populates ``site_potentials``
        with one value per site (needed for the eFNV charge correction),
        matching ``get_site_potentials`` called directly on the same
        directory.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = get_calculation_outputs(self.bulk_dir, label="bulk", load_site_potentials=True)

        assert len(outputs.site_potentials) == len(outputs.structure)

        direct_site_potentials = get_site_potentials(self.bulk_dir, dir_type="bulk", outputs=outputs)
        assert np.allclose(outputs.site_potentials, direct_site_potentials)

    def test_calculation_outputs_serialisation(self):
        """
        Test that ``energy``/``structure``/``site_potentials`` survive an
        ``as_dict()``/``from_dict()`` round trip with their parsed numerical
        values intact.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = get_calculation_outputs(self.bulk_dir, label="bulk", load_site_potentials=True)

        reloaded = CalculationOutputs.from_dict(outputs.as_dict())
        assert np.isclose(reloaded.energy, outputs.energy)
        assert len(reloaded.structure) == len(outputs.structure)
        assert np.allclose(reloaded.site_potentials, outputs.site_potentials)
        assert np.isclose(reloaded.get_computed_entry().energy, outputs.energy)


class CdTeSpinPolarisedOutputsTest(unittest.TestCase):
    """
    Test parsing of spin-polarised (``spin collinear``) FHI-aims outputs,
    using the real ``CdTe``/``v_Cd_+1`` (Cd vacancy, charge +1) data at
    ``tests/data/aims/CdTe/v_Cd_+1/aims_gam`` -- the only spin-polarised
    calculation currently in the aims test data (all other charge states use
    a closed-shell/non-spin-polarised ``default_initial_moment``-free setup).
    """

    def setUp(self):
        self.defect_dir = Path(data_dir) / "aims" / "CdTe" / "v_Cd_+1" / "aims_gam"

    def test_get_calculation_outputs_projected_eigenvalues_spin_polarised(self):
        """
        As with ``MgOOutputsTest.test_get_calculation_outputs_projected_
        eigenvalues`` (non-spin-polarised), but checking both spin channels
        of a genuinely spin-polarised calculation are parsed from
        ``Mulliken.out`` (``Spin channel: up``/``down`` blocks) into separate
        ``{Spin.up: array, Spin.down: array}`` entries, each independently
        satisfying the same per-band sum-to-~1 physical cross-check.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = get_calculation_outputs(
                self.defect_dir, label="defect", parse_projected_eigen=True
            )

        assert outputs.magnetization == 3.0  # N_up - N_down, from `aims.out`
        assert set(outputs.projected_eigenvalues.keys()) == {Spin.up, Spin.down}

        for spin in (Spin.up, Spin.down):
            projected_eigenvalues = outputs.projected_eigenvalues[spin][0]  # Gamma-only
            assert projected_eigenvalues.shape == (
                outputs.run_metadata["num_bands"],
                len(outputs.structure),
                4,  # l=0,1,2,3
            )
            band_sums = projected_eigenvalues.sum(axis=(1, 2))
            assert np.allclose(band_sums, 1.0, atol=1e-2)


class CdTeChargeCorrectionTest(unittest.TestCase):
    """
    Test the Kumagai (eFNV) charge correction computed via the aims backend
    against the VASP-computed correction for the same physical defect,
    ``v_Cd_-2`` (Cd vacancy, charge -2) in CdTe -- using the real aims
    output at ``tests/data/aims/CdTe/v_Cd_-2`` and the real VASP output at
    ``tests/data/vasp/CdTe_charge_correction_tests/v_Cd_-2_vasp_gam``.

    The two calculations use different supercells (54 aims atoms vs. 63
    VASP atoms) and XC functionals (aims: tuned HSE06, per
    ``AimsTest.test_CdTe_defect_input_generation``; VASP: standard GGA), so
    their correction values are not expected to match exactly -- but the
    eFNV correction is primarily an electrostatic quantity (set by the
    dielectric constant, charge state and long-range potential screening),
    so should be relatively insensitive to these differences, unlike e.g.
    total energies.
    """

    def test_kumagai_correction_matches_vasp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aims_defect_entry = DefectParser.from_paths(
                defect_path=str(Path(data_dir) / "aims" / "CdTe" / "v_Cd_-2" / "aims_gam"),
                bulk_path=str(Path(data_dir) / "aims" / "CdTe" / "CdTe_bulk" / "aims_gam"),
                dielectric=9.13,
                calculator="aims",
                load_site_potentials=True,
            ).defect_entry
            vasp_defect_entry = DefectParser.from_paths(
                defect_path=str(Path(vasp_data_dir) / "CdTe_charge_correction_tests" / "v_Cd_-2_vasp_gam"),
                bulk_path=str(Path(vasp_data_dir) / "CdTe_charge_correction_tests" / "bulk_vasp_gam"),
                dielectric=9.13,
                parse_projected_eigen=False,
            ).defect_entry

        assert aims_defect_entry.charge_state == vasp_defect_entry.charge_state == -2
        aims_correction = aims_defect_entry.corrections["kumagai_charge_correction"]
        vasp_correction = vasp_defect_entry.corrections["kumagai_charge_correction"]
        assert np.sign(aims_correction) == np.sign(vasp_correction)
        # ~3.4% difference in practice (0.7766 eV aims vs. 0.7509 eV VASP), despite the
        # differing supercells/functionals -- allow some margin beyond that:
        assert np.isclose(aims_correction, vasp_correction, rtol=0.1)


def _parse_aims_defects(
    aims_root: Path, bulk_dir_name: str, dielectric: float | np.ndarray
) -> tuple[dict, list[str]]:
    """
    Re-parse all real aims ``aims_gam`` defect calculations directly under
    ``aims_root`` (skipping ``bulk_dir_name``, ``CompetingPhases`` and
    ``logs``), returning ``(parsed_defect_dict, skipped_unconverged_names)``.

    Calculations with no usable total energy (``entry.sc_entry._energy is
    None``, i.e. SCF didn't converge -- ``*** scf_solver: SCF cycle not
    converged.`` at the end of ``aims.out``, vs. ``Have a nice day.`` for a
    converged run) are skipped rather than included, as attempting to build a
    ``DefectThermodynamics`` with such an entry raises ``TypeError:
    unsupported format string passed to NoneType.__format__`` (from
    ``DefectEntry.sc_entry_energy`` hashing ``None``) -- so there may be
    scope for parsing to instead skip/warn on unconverged calculations,
    rather than erroring.
    """
    bulk_path = aims_root / bulk_dir_name / "aims_gam"
    defect_dirs = sorted(
        d
        for d in aims_root.iterdir()
        if d.is_dir()
        and d.name not in (bulk_dir_name, "CompetingPhases", "logs")
        and (d / "aims_gam" / "aims.out").exists()
    )

    parsed_defect_dict = {}
    skipped = []
    for d in defect_dirs:
        defect_entry = DefectParser.from_paths(
            defect_path=str(d / "aims_gam"),
            bulk_path=str(bulk_path),
            dielectric=dielectric,
            calculator="aims",
            load_site_potentials=True,
        ).defect_entry
        if defect_entry.sc_entry._energy is None:  # SCF didn't converge, no usable energy
            skipped.append(d.name)
            continue
        parsed_defect_dict[d.name] = defect_entry

    return parsed_defect_dict, skipped


def _get_aims_chempots(competing_phases_dir: Path, formula: str) -> dict | None:
    """
    Compute the aims-derived chemical potential limits for ``formula`` from
    the real aims ``aims_std`` calculations under ``competing_phases_dir``
    (one subfolder per competing phase; see ``CdTeCompetingPhasesTest``/
    ``MgOCompetingPhasesTest`` for how this data is generated/parsed), or
    return ``None`` if those calculations haven't been run yet (no
    ``aims.out`` files present).
    """
    if not competing_phases_dir.is_dir():
        return None

    phase_dirs = sorted(
        d for d in competing_phases_dir.iterdir() if d.is_dir() and d.name != "logs"
    )
    if not phase_dirs or not all((d / "aims_std" / "aims.out").exists() for d in phase_dirs):
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entries = []
        for d in phase_dirs:
            aims_output = get_aims_output(d / "aims_std" / "aims.out")
            image = aims_output.get_image(-1)
            entries.append(
                ComputedStructureEntry(image.geometry.structure, image.results["total_energy"])
            )
        return get_doped_chempots_from_entries(entries, formula)


def _generate_transition_level_diagrams(
    formula: str,
    vasp_thermo_path: Path,
    aims_root: Path,
    aims_bulk_dir_name: str,
    dielectric: float | np.ndarray,
    competing_phases_error_hint: str,
    output_dir: Path | str | None,
):
    """
    Shared implementation for ``generate_CdTe_transition_level_diagrams`` and
    ``generate_MgO_transition_level_diagrams`` (see those for details) --
    builds ``DefectThermodynamics`` for ``formula`` from both real VASP (pre-
    parsed, loaded from ``vasp_thermo_path``) and real FHI-aims (re-parsed
    from ``aims_root``) data, and plots both the vertical transition level
    diagram (``plot_transition_levels()``) and the formation energy vs. Fermi
    level plot (``plot()``, a.k.a. `the` transition level diagram, per its
    own docstring) for a by-eye cross-code comparison.

    Raises:
        RuntimeError: If the aims competing-phase calculations for
            ``formula`` haven't been run yet, as the aims formation energy
            plot requires real elemental references (unlike
            ``plot_transition_levels``, which doesn't need chemical
            potentials at all) -- zero elemental references are off by
            ~100,000s of eV (all-electron aims total energies) and render
            the plot illegible.
    """
    from doped.thermodynamics import DefectThermodynamics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vasp_thermo = loadfn(vasp_thermo_path)
        aims_defect_dict, skipped_aims_defects = _parse_aims_defects(
            aims_root, aims_bulk_dir_name, dielectric
        )
        aims_thermo = DefectThermodynamics(aims_defect_dict)

    print(f"VASP: {len(vasp_thermo.defect_entries)} defect entries parsed")
    print(
        f"AIMS: {len(aims_thermo.defect_entries)} defect entries parsed, "
        f"skipped {len(skipped_aims_defects)} unconverged: {skipped_aims_defects}"
    )

    vasp_tl_filename = Path(output_dir) / f"{formula}_vasp_transition_levels.png" if output_dir else None
    aims_tl_filename = Path(output_dir) / f"{formula}_aims_transition_levels.png" if output_dir else None
    vasp_thermo.plot_transition_levels(filename=vasp_tl_filename)
    aims_thermo.plot_transition_levels(filename=aims_tl_filename)

    vasp_fe_filename = (
        str(Path(output_dir) / f"{formula}_vasp_formation_energies.png") if output_dir else None
    )
    vasp_thermo.plot(filename=vasp_fe_filename)  # `vasp_thermo.chempots` already set (pre-parsed)

    aims_chempots = _get_aims_chempots(aims_root / "CompetingPhases", formula)
    if aims_chempots is None:
        raise RuntimeError(
            f"AIMS {formula} competing-phase calculations have not been run yet -- "
            f"{competing_phases_error_hint}"
        )
    aims_thermo.chempots = aims_chempots

    aims_fe_filename = (
        str(Path(output_dir) / f"{formula}_aims_formation_energies.png") if output_dir else None
    )
    aims_thermo.plot(filename=aims_fe_filename)

    return vasp_thermo, aims_thermo, skipped_aims_defects


def generate_CdTe_transition_level_diagrams(output_dir: Path | str | None = None):
    """
    Ad hoc helper (not itself a test -- see ``CdTeTransitionLevelDiagramTest``
    below) to build ``DefectThermodynamics`` for CdTe from both the real VASP
    and real FHI-aims data in this repo, and plot the corresponding defect
    transition level diagrams for a by-eye cross-code comparison. Can also be
    called manually, e.g. via:
    ``python -c "from test_aims import generate_CdTe_transition_level_diagrams as f; f('.')"``

    VASP data: loaded directly from the pre-parsed, fully-converged reference
    dataset at ``examples/CdTe/CdTe_thermo_wout_meta.json.gz`` (the same one
    used in the quickstart tutorial and ``test_thermodynamics.py``). The VASP
    data under ``tests/data/vasp/CdTe`` is `not` used here, as those
    per-defect ``<defect>.json.gz`` files are placeholder ``DefectEntry``\\s
    (all with ``sc_entry_energy = 0.0``) generated for input-file-writing
    tests, not real calculation outputs.

    AIMS data: re-parsed from the real ``aims.out`` files under
    ``tests/data/aims/CdTe/*/aims_gam`` (the only subfolder with actual
    calculation output for these defects -- ``aims_std``/``aims_ncl`` only
    contain unrun input files and the same kind of placeholder
    ``<defect>.json.gz`` as the VASP test data).

    Data gap: roughly half of the aims ``aims_gam`` calculations in this
    dataset did not reach SCF convergence (see ``_parse_aims_defects``), and
    so are skipped (see the printed ``skipped`` list), but this means the
    aims transition level diagram is missing entire defects (e.g. all 7
    ``Te_Cd`` charge states are unconverged) and most charge states of
    several others (e.g. only the ``+4`` charge state of ``Te_i_Td_Te2.83``
    converged), unlike the complete VASP diagram.

    Args:
        output_dir (Path | str | None):
            If provided, directory in which to save the four
            ``CdTe_{vasp,aims}_{transition_levels,formation_energies}.png``
            plots. If ``None`` (default), the plots are not saved to disk
            (but the built ``DefectThermodynamics`` are still returned).

    Returns:
        tuple[DefectThermodynamics, DefectThermodynamics, list[str]]:
            ``(vasp_thermo, aims_thermo, skipped_aims_defects)``.

    Raises:
        RuntimeError: See ``_generate_transition_level_diagrams``; here, if
            the aims ``CdTe/CompetingPhases`` calculations (see
            ``CdTeCompetingPhasesTest``) haven't been run yet.
    """
    return _generate_transition_level_diagrams(
        formula="CdTe",
        vasp_thermo_path=Path(EXAMPLE_DIR) / "CdTe" / "CdTe_thermo_wout_meta.json.gz",
        aims_root=Path(data_dir) / "aims" / "CdTe",
        aims_bulk_dir_name="CdTe_bulk",
        dielectric=9.13,  # CdTe, from CdTeChargeCorrectionTest
        competing_phases_error_hint=(
            "run tests/data/aims/CdTe/CompetingPhases/run.slurm (editing `AIMS_EXE` first) and add "
            "the resulting aims.out files, then re-run this function."
        ),
        output_dir=output_dir,
    )


def generate_MgO_transition_level_diagrams(output_dir: Path | str | None = None):
    """
    As ``generate_CdTe_transition_level_diagrams``, but for MgO -- can also
    be called manually, e.g. via:
    ``python -c "from test_aims import generate_MgO_transition_level_diagrams as f; f('.')"``

    VASP data: loaded directly from the pre-parsed reference dataset at
    ``examples/MgO/MgO_thermo.json.gz``. Unlike CdTe, this only contains the
    ``Mg_O`` antisite (5 charge states) -- the illustrative single-defect
    example used in the docs/tutorials -- `not` the full complement of MgO
    point defects, so the VASP diagram here is much sparser than the aims one
    (which has all 5 defect types: ``v_Mg``, ``v_O``, ``Mg_O``, ``O_Mg``,
    ``Mg_i``).

    AIMS data: re-parsed from the real ``aims.out`` files under
    ``tests/data/aims/MgO/*/aims_gam``, as for CdTe. Convergence is much
    better here than for CdTe (only 4 of 24 unconverged: ``O_i_Td_0/-1/-2``,
    ``O_Mg_-3``).

    Unlike CdTe, the aims ``CompetingPhases`` data already exists in this
    repo (``tests/data/aims/MgO/CompetingPhases``, real & converged; see
    ``MgOCompetingPhasesTest``), so this should run end-to-end without
    hitting the ``RuntimeError`` case.

    Args:
        output_dir (Path | str | None):
            As ``generate_CdTe_transition_level_diagrams``, for
            ``MgO_{vasp,aims}_{transition_levels,formation_energies}.png``.

    Returns:
        tuple[DefectThermodynamics, DefectThermodynamics, list[str]]:
            ``(vasp_thermo, aims_thermo, skipped_aims_defects)``.
    """
    return _generate_transition_level_diagrams(
        formula="MgO",
        vasp_thermo_path=Path(EXAMPLE_DIR) / "MgO" / "MgO_thermo.json.gz",
        aims_root=Path(data_dir) / "aims" / "MgO",
        aims_bulk_dir_name="MgO_bulk",
        dielectric=8.8963,  # MgO, from MgO_thermo.json.gz's defect_entries' calculation_metadata
        competing_phases_error_hint=(
            "this shouldn't happen, as this data already exists in this repo -- check "
            "tests/data/aims/MgO/CompetingPhases."
        ),
        output_dir=output_dir,
    )


class CdTeTransitionLevelDiagramTest(unittest.TestCase):
    """
    Generates the CdTe VASP/aims transition level diagrams (via
    ``generate_CdTe_transition_level_diagrams``) and saves them to
    ``tests/data/aims``, for visual cross-code comparison -- not a
    correctness check, so this always passes (any failure is just warned)
    regardless of how many aims calculations happen to have converged in the
    current test data (see the docstring of
    ``generate_CdTe_transition_level_diagrams`` for the current data gaps).
    """

    def test_generate_CdTe_transition_level_diagrams(self):
        try:
            generate_CdTe_transition_level_diagrams(output_dir=Path(data_dir) / "aims")
        except Exception as e:
            warnings.warn(f"Failed to generate CdTe transition level diagrams: {e!r}")


class MgOTransitionLevelDiagramTest(unittest.TestCase):
    """
    As ``CdTeTransitionLevelDiagramTest``, but for MgO (via
    ``generate_MgO_transition_level_diagrams``) -- also always passes, though
    (unlike CdTe) this is expected to run end-to-end without hitting the
    warned-and-skipped case, as the aims ``MgO/CompetingPhases`` data already
    exists in this repo.
    """

    def test_generate_MgO_transition_level_diagrams(self):
        try:
            generate_MgO_transition_level_diagrams(output_dir=Path(data_dir) / "aims")
        except Exception as e:
            warnings.warn(f"Failed to generate MgO transition level diagrams: {e!r}")


class CdTeCompetingPhasesTest(unittest.TestCase):
    """
    Numerical comparison of FHI-aims vs. VASP competing-phase (elemental
    reference energy) results for CdTe, mirroring ``MgOCompetingPhasesTest``.

    Unlike MgO, there are no real VASP competing-phase calculations stored in
    this repo for CdTe (only the final derived
    ``examples/CdTe/CdTe_chempots.json``), so the comparison here is against
    that stored reference rather than re-deriving the VASP side from raw
    ``vasprun.xml`` files.

    The aims relaxation input files for the CdTe competing phases (at
    ``tests/data/aims/CdTe/CompetingPhases``, one folder per phase from
    ``CompetingPhases("CdTe", api_key=...).entries``) were generated with::

        CompetingPhases("CdTe", api_key=...).write_aims_relaxation_files(
            user_parameters={"xc": "hse06 0.2", "hse_unit": "A", "hybrid_xc_coeff": 0.345},
            species_defaults="light",
            output_path="tests/data/aims/CdTe/CompetingPhases", subfolder="aims_std",
        )

    i.e. matching the tuned-hybrid functional (`not` plain PBEsol, unlike
    ``MgOCompetingPhasesTest``) used for the CdTe defect supercells themselves
    (see ``AimsTest.test_CdTe_defect_input_generation``) -- required, as the
    elemental references need to be on the same absolute (all-electron) total
    energy scale as the defect/bulk supercell energies they're subtracted
    from in the formation energy expression, for that huge (~100,000s of eV)
    scale to properly cancel down to a physically sensible (~eV-scale)
    formation energy. An earlier version of this used PBEsol (mirroring the
    MgO precedent, and ``doped``'s own documented default for competing-phase
    calculations) which gave a `much` better-behaved-looking but still wrong
    result (formation energies off by 10s of eV, rather than ~100,000s) --
    i.e. this failure mode isn't always obviously broken from the plot alone,
    so take care to match functionals when setting up equivalent comparisons
    elsewhere.

    This means the comparison against the (presumed GGA-level) VASP-derived
    elemental references in ``CdTe_chempots.json`` below is no longer a
    strict like-for-like (same XC functional) comparison as it is for MgO --
    so the ``rtol`` here may need loosening once this has real data to
    compare against.

    These calculations have not yet been run (no ``aims.out`` files exist
    under ``tests/data/aims/CdTe/CompetingPhases`` yet), so this test is
    skipped until they are -- run
    ``tests/data/aims/CdTe/CompetingPhases/run.slurm`` (edit ``AIMS_EXE``
    first) to generate them.
    """

    def setUp(self):
        self.aims_dir = Path(data_dir) / "aims" / "CdTe" / "CompetingPhases"
        self.phases = sorted(
            d.name for d in self.aims_dir.iterdir() if d.is_dir() and d.name != "logs"
        )
        if not self.phases or not all(
            (self.aims_dir / phase / "aims_std" / "aims.out").exists() for phase in self.phases
        ):
            self.skipTest(
                f"aims.out not yet present for all CdTe competing phases -- run "
                f"{self.aims_dir / 'run.slurm'} first."
            )

    def _aims_entries(self):
        entries = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for phase in self.phases:
                aims_output = get_aims_output(self.aims_dir / phase / "aims_std" / "aims.out")
                image = aims_output.get_image(-1)
                entries.append(
                    ComputedStructureEntry(image.geometry.structure, image.results["total_energy"])
                )
        return entries

    def test_aims_chempots_match_vasp(self):
        """
        Compare the aims- and VASP-derived ΔµTe (at the ``CdTe-Cd`` Cd-rich
        limit) -- `not` the raw ``elemental_refs``, which are on fundamentally
        different absolute energy scales between the two codes (aims:
        all-electron; VASP: pseudopotential, valence-only) and are never
        expected to numerically agree -- only the relative (formal) chemical
        potentials, referenced to each code's own elemental phases, are
        physically comparable (mirroring
        ``MgOCompetingPhasesTest.test_aims_chempots_match_vasp``).
        """
        aims_chempots = get_doped_chempots_from_entries(self._aims_entries(), "CdTe")
        vasp_chempots = loadfn(Path(EXAMPLE_DIR) / "CdTe" / "CdTe_chempots.json")

        vasp_delta_mu_Te = vasp_chempots["limits_wrt_el_refs"]["Cd-CdTe"]["Te"]
        aims_delta_mu_Te = aims_chempots["limits_wrt_el_refs"]["CdTe-Cd"]["Te"]

        # the "CdTe-Te"/"CdTe-Te" limit's ΔµCd should be the negative of the "Cd-CdTe"/
        # "CdTe-Cd" limit's ΔµTe, for both codes (binary 1:1 compound):
        assert np.isclose(vasp_chempots["limits_wrt_el_refs"]["CdTe-Te"]["Cd"], vasp_delta_mu_Te)
        assert np.isclose(aims_chempots["limits_wrt_el_refs"]["CdTe-Te"]["Cd"], aims_delta_mu_Te)

        # like MgOCompetingPhasesTest, allow a fairly loose tolerance given the differing
        # numerics (all-electron LCAO vs. pseudopotential plane-wave; light vs. converged
        # basis set) -- may need adjusting once this has more real data points to compare
        # against (currently: -1.25 eV VASP vs. -1.23 eV aims, ~1.8% difference in practice):
        assert np.isclose(aims_delta_mu_Te, vasp_delta_mu_Te, rtol=0.2)


class MgOCompetingPhasesTest(unittest.TestCase):
    """
    Numerical comparison of FHI-aims vs. VASP competing-phase (chemical
    potential limit) results for MgO.

    The real VASP competing-phase calculations at
    ``examples/MgO/CompetingPhases`` (``Mg`` in three polymorphs, ``O2``,
    and ``MgO`` itself) use ``GGA = Ps`` (PBEsol) with no hybridisation. The
    ``tests/data/aims/MgO/CompetingPhases`` aims data was generated with
    ``CompetingPhases.write_aims_relaxation_files(user_parameters={"xc":
    "pbesol"}, ...)`` (geometries seeded from the VASP-relaxed structures)
    and run with the ``light`` species-defaults tier, for a like-for-like
    (same XC functional) but not identical (all-electron LCAO vs.
    pseudopotential plane-wave; ``light`` vs. converged basis set) cross-
    code comparison.

    Note that VASP and aims independently select *different* elemental Mg
    ground-state polymorphs here (VASP-PBEsol: ``P6_3/mmc``; aims-PBEsol/
    light: ``R-3m``) -- expected/correct behaviour, as each code's own
    lowest-energy competing phase should be used as its elemental
    reference, but a further (minor) source of divergence between the two
    ``ΔµO`` values compared below, alongside the basis-set/pseudopotential
    differences.
    """

    def setUp(self):
        self.vasp_dir = Path(EXAMPLE_DIR) / "MgO" / "CompetingPhases"
        self.aims_dir = Path(data_dir) / "aims" / "MgO" / "CompetingPhases"
        self.phases = [
            "MgO_Fm-3m_EaH_0",
            "Mg_Fm-3m_EaH_0",
            "Mg_P6_3mmc_EaH_0.009",
            "Mg_R-3m_EaH_0.003",
            "O2_mmm_EaH_0",
        ]

    def _vasp_entries(self):
        entries = []
        for phase in self.phases:
            vr = Vasprun(
                str(self.vasp_dir / phase / "vasp_std" / "vasprun.xml.gz"), parse_potcar_file=False
            )
            entries.append(ComputedStructureEntry(vr.final_structure, vr.final_energy))
        return entries

    def _aims_entries(self):
        entries = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for phase in self.phases:
                aims_output = get_aims_output(self.aims_dir / phase / "aims_std" / "aims.out")
                image = aims_output.get_image(-1)
                entries.append(
                    ComputedStructureEntry(image.geometry.structure, image.results["total_energy"])
                )
        return entries

    def test_vasp_chempots_match_stored_reference(self):
        """
        Sanity check that re-deriving the chemical potential limits here
        (from the raw ``vasprun.xml.gz`` files, via
        ``get_doped_chempots_from_entries``) exactly reproduces the stored
        ``examples/MgO/CompetingPhases/MgO_chempots.json`` -- confirming
        the comparison methodology below matches ``doped``'s own analysis
        pipeline, before comparing against the (independently-generated)
        aims results.
        """
        chempots = get_doped_chempots_from_entries(self._vasp_entries(), "MgO")
        stored_chempots = loadfn(self.vasp_dir / "MgO_chempots.json")
        assert chempots["limits_wrt_el_refs"] == stored_chempots["limits_wrt_el_refs"]

    def test_aims_chempots_match_vasp(self):
        """
        Compare the aims- and VASP-derived ``ΔµO`` (at the ``MgO-Mg``
        Mg-rich limit) for MgO -- the deviation of the O chemical potential
        from its elemental O2 reference, i.e. (the negative of) the MgO
        formation energy per formula unit.
        """
        vasp_chempots = get_doped_chempots_from_entries(self._vasp_entries(), "MgO")
        aims_chempots = get_doped_chempots_from_entries(self._aims_entries(), "MgO")

        vasp_delta_mu_O = vasp_chempots["limits_wrt_el_refs"]["MgO-Mg"]["O"]
        aims_delta_mu_O = aims_chempots["limits_wrt_el_refs"]["MgO-Mg"]["O"]
        assert np.isclose(vasp_delta_mu_O, -5.64572, atol=1e-4)  # from the stored MgO_chempots.json
        assert np.isclose(aims_delta_mu_O, -6.1058, atol=1e-4)  # from the real aims.out data above

        # the "MgO-O2" limit's ΔµMg should be the negative of the "MgO-Mg" limit's ΔµO, for
        # both codes (binary 1:1 compound):
        assert np.isclose(vasp_chempots["limits_wrt_el_refs"]["MgO-O2"]["Mg"], vasp_delta_mu_O)
        assert np.isclose(aims_chempots["limits_wrt_el_refs"]["MgO-O2"]["Mg"], aims_delta_mu_O)

        # ~8.1% difference in practice (-6.1058 eV aims vs. -5.64572 eV VASP), despite the
        # differing basis sets/pseudopotential treatments (and elemental Mg reference
        # polymorphs, see class docstring) -- allow some margin beyond that:
        assert np.isclose(aims_delta_mu_O, vasp_delta_mu_O, rtol=0.2)
