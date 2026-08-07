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
from pymatgen.electronic_structure.core import Spin
from test_utils import EXAMPLE_DIR, data_dir

# temp
SETTINGS["AIMS_SPECIES_DIR"] = "~/Documents/fhi-aims.260331/species_defaults"

from doped.generation import DefectsGenerator
from doped.io.aims.inputs import DefectsSet
from doped.io.aims.utils import _resolve_species_defaults
from doped.io.aims.outputs import (
    get_aims_output, get_atomic_magnetic_moments_from_aims_output,
    get_band_edge_eigenvalues_from_aims_output, get_calculation_outputs,
    get_magnetization_from_aims_output, get_n_electrons_from_aims_output,
    get_neutral_n_electrons, get_site_potentials,
    spin_degeneracy_from_aims_output, total_charge_from_aims_output)
from doped.io.outputs import CalculationOutputs
from doped.utils.efficiency import Structure


class AimsTest(unittest.TestCase):

    def setUp(self):
        self.prim_cdte = Structure.from_file(f"{EXAMPLE_DIR}/CdTe/relaxed_primitive_POSCAR")
        self.aims_output_path = Path(data_dir) / "aims"
        self.mgo_prim = Structure.from_file(f"{EXAMPLE_DIR}/MgO/Input_files/MgO_POSCAR_prim")
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
        """Generate MgO defects and write an FHI-aims input set for each.

        The real MgO VASP calculations used ``GGA = Ps`` (PBEsol) with
        ``LHFCALC = False`` (no hybridisation).
        """
        self._generate_and_check_defect_inputs(
            self.mgo_prim, "MgO", user_parameters={"xc": "pbesol"}, expect_soc=False  # Mg/O < 31
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

    def _generate_and_check_defect_inputs(self, structure, material_dir_name, user_parameters, expect_soc):
        """Generate defects for ``structure`` and write/check an FHI-aims input set
        for each, in ``self.aims_output_path / material_dir_name``.

        ``expect_soc`` is the expected value of ``DefectsSet.soc`` (whether
        ``aims_ncl`` -- single-point, spin-orbit-coupled -- input sets should
        also be generated); set to ``True`` if the max atomic number across the
        material's elements is >= 31 (i.e. further down the periodic table than
        Zn), mirroring ``doped.io.vasp.inputs.DefectsSet``'s auto-detection.
        """
        defect_generator = DefectsGenerator(structure)
        output_path = self.aims_output_path / material_dir_name
        os.makedirs(output_path, exist_ok=True)

        defects_set = DefectsSet(
            defect_generator,
            user_parameters=user_parameters,
            species_defaults="light",
        )
        assert defects_set.soc is expect_soc
        defects_set.write_files(output_path=output_path, rattle=False, processes=1)

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
        assert re.findall(r"^\s*sc_iter_limit\s+(\S+)\s*$", control, re.MULTILINE) == ["100"]

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


class AimsOutputsTest(unittest.TestCase):

    def setUp(self):
        self.aims_output_path = Path(data_dir) / "aims" / "CdTe"
        self.bulk_dir = self.aims_output_path / "CdTe_bulk" / "aims_gam"
        self.charged_dirs = {
            charge: self.aims_output_path / f"Cd_Te_+{charge}" / "aims_gam" for charge in (1, 2, 3, 4)
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
            get_aims_output(self.aims_output_path / "CdTe_bulk" / "aims_gam" / "not_a_real_file.out")

    def test_get_n_electrons_from_aims_output(self):
        """54-atom Cd27Te27 bulk cell: 27*48 + 27*52 = 2700 electrons."""
        assert get_n_electrons_from_aims_output(self.bulk_output) == 2700
        # Cd_Te antisite (Cd on Te site) with charge +1: one fewer Te, one more Cd, plus +1 charge:
        assert get_n_electrons_from_aims_output(self.charged_outputs[1]) == 2700 - 4 - 1

    def test_get_neutral_n_electrons(self):
        structure = self.bulk_output.get_image(-1).geometry.structure
        assert get_neutral_n_electrons(structure) == 2700
        charged_structure = self.charged_outputs[1].get_image(-1).geometry.structure
        assert get_neutral_n_electrons(charged_structure) == 2700 - 4  # Cd27+1Te27-1, neutral electron count

    def test_total_charge_from_aims_output(self):
        assert total_charge_from_aims_output(self.bulk_output) == 0
        for charge, aims_output in self.charged_outputs.items():
            assert total_charge_from_aims_output(aims_output) == charge

    def test_get_magnetization_from_aims_output(self):
        """All CdTe test outputs are non-spin-polarised, so magnetization is zero."""
        assert get_magnetization_from_aims_output(self.bulk_output) == 0.0
        for aims_output in self.charged_outputs.values():
            assert get_magnetization_from_aims_output(aims_output) == 0.0

    def test_spin_degeneracy_from_aims_output(self):
        # bulk: even (2700) electrons, zero magnetization -> singlet:
        assert spin_degeneracy_from_aims_output(self.bulk_output) == 1
        # Cd_Te_+1: odd (2695) electrons -> doublet:
        assert spin_degeneracy_from_aims_output(self.charged_outputs[1]) == 2
        # explicit charge_state overrides the auto-determined electron count:
        assert spin_degeneracy_from_aims_output(self.bulk_output, charge_state=1) == 2
        assert spin_degeneracy_from_aims_output(self.bulk_output, charge_state=0) == 1

    def test_get_atomic_magnetic_moments_from_aims_output_requires_mulliken(self):
        """None of the CdTe test outputs have ``output mulliken`` results parsed."""
        with self.assertRaises(KeyError):
            get_atomic_magnetic_moments_from_aims_output(self.bulk_output)

    def test_get_band_edge_eigenvalues_from_aims_output(self):
        assert np.allclose(
            get_band_edge_eigenvalues_from_aims_output(self.bulk_output),
            (-4.87610472, -4.22510898, 0.65099574),
        )
        # Cd_Te_+1/+3/+4 are (spuriously) flagged metallic by `aims` (defect state broadening
        # closes the gap), while Cd_Te_+2 retains a resolved gap:
        known_band_edges = {
            1: (-4.89197007, -4.03259341, 0.0),
            2: (-4.88423806, -4.02181037, 0.86242769),
            3: (-4.91784489, -4.02287586, 0.0),
            4: (-4.94692462, -4.02397834, 0.0),
        }
        for charge, aims_output in self.charged_outputs.items():
            assert np.allclose(
                get_band_edge_eigenvalues_from_aims_output(aims_output), known_band_edges[charge]
            )

    def test_get_calculation_outputs(self):
        """
        Test parsing an FHI-aims bulk supercell calculation to
        ``CalculationOutputs``, checking the parsed fields against known
        numerical values from the ``aims.out`` test file.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = get_calculation_outputs(self.bulk_dir, label="bulk")

        assert len(outputs.structure) == 54  # Cd27Te27 bulk supercell
        assert np.isclose(outputs.energy, -9226741.70719699)
        assert outputs.converged_electronic is True
        assert np.isclose(outputs.efermi, -4.55060685)
        assert outputs.nelect == 2700
        assert outputs.charge == 0
        assert outputs.magnetization == 0.0
        assert outputs.spin_degeneracy() == 1
        assert outputs.spin_degeneracy(charge_state=0) == 1
        assert int(outputs.run_metadata["charge"]) == 0
        assert outputs.run_metadata["num_electrons"] == 2700.0
        assert np.isclose(outputs.vbm, -4.87610472)
        assert np.isclose(outputs.cbm, -4.22510898)
        assert np.isclose(outputs.band_gap, 0.65099574)

        assert list(outputs.eigenvalues.keys()) == [Spin.up]  # non-spin-polarised
        eigenvalues = outputs.eigenvalues[Spin.up]
        assert eigenvalues.shape == (1, 1647, 2)  # (nkpoints, nbands, 2), Gamma-only
        assert np.allclose(eigenvalues[0, 0], (-32793.8863, 2.0))  # lowest (core) band
        assert np.allclose(eigenvalues[0, 1349], (-4.8761, 2.0))  # VBM, occupied
        assert np.allclose(eigenvalues[0, 1350], (-4.22511, 0.0))  # CBM, unoccupied
        assert np.allclose(eigenvalues[0, -1], (10.79808, 0.0))  # highest band

    def test_get_calculation_outputs_charged_defect(self):
        """
        Test parsing charged Cd_Te antisite defect supercell calculations to
        ``CalculationOutputs``, checking the parsed fields against known
        numerical values from the ``aims.out`` test files.
        """
        known_energies = {
            1: -9193276.22903702,
            2: -9193272.18784907,
            3: -9193267.31362098,
            4: -9193262.39367238,
        }
        known_efermis = {
            1: -4.0424661,
            2: -4.45302422,
            3: -4.88364145,
            4: -4.9316963,
        }
        known_band_edges = {
            1: (-4.89197007, -4.03259341, 0.0),
            2: (-4.88423806, -4.02181037, 0.86242769),
            3: (-4.91784489, -4.02287586, 0.0),
            4: (-4.94692462, -4.02397834, 0.0),
        }
        for charge, charged_dir in self.charged_dirs.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = get_calculation_outputs(charged_dir, label="defect")

            assert len(outputs.structure) == 54
            assert np.isclose(outputs.energy, known_energies[charge])
            assert outputs.converged_electronic is True
            assert np.isclose(outputs.efermi, known_efermis[charge])
            assert outputs.charge == charge
            assert outputs.nelect == 2700 - 4 - charge  # one fewer Te, one more Cd, plus `charge`
            assert outputs.magnetization == 0.0
            assert outputs.spin_degeneracy() == (2 if (2700 - 4 - charge) % 2 else 1)
            assert int(outputs.run_metadata["charge"]) == charge
            assert np.allclose((outputs.vbm, outputs.cbm, outputs.band_gap), known_band_edges[charge])

    def test_get_calculation_outputs_site_potentials(self):
        """
        Test that ``load_site_potentials=True`` populates ``site_potentials``
        with the expected numerical values (needed for the eFNV charge
        correction), matching ``get_site_potentials`` called directly on the
        same directory.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = get_calculation_outputs(self.bulk_dir, label="bulk", load_site_potentials=True)

        assert len(outputs.site_potentials) == len(outputs.structure)
        assert np.isclose(outputs.site_potentials[0], 8.91945973)

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
