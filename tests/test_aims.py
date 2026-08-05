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
from test_utils import EXAMPLE_DIR, data_dir

# temp
SETTINGS["AIMS_SPECIES_DIR"] = "~/Documents/fhi-aims.260331/species_defaults"

from doped.generation import DefectsGenerator
from doped.io.aims.inputs import DefectsSet, _resolve_species_defaults
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

    def test_species_defaults_path_resolution(self):
        """Resolve the default light species_defaults shorthand."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            species_defaults = Path(tmp_dir) / "species_defaults"
            basis_path = species_defaults / "defaults_2020" / "light"
            basis_path.mkdir(parents=True)
            assert _resolve_species_defaults(basis_path) == str(basis_path.resolve())
            with patch("doped.io.aims.inputs.SETTINGS", {"AIMS_SPECIES_DIR": str(species_defaults)}):
                assert _resolve_species_defaults("light") == str(basis_path.resolve())
                assert _resolve_species_defaults(Path("defaults_2020") / "light") == str(basis_path.resolve())

    def test_CdTe_defect_input_generation(self):
        """Generate CdTe defects and write an FHI-aims input set for each."""
        defect_generator = DefectsGenerator(self.prim_cdte)
        output_path = self.aims_output_path / "CdTe"
        os.makedirs(output_path, exist_ok=True)

        defects_set = DefectsSet(
            defect_generator,
            user_parameters={"xc": "pbe"},
            species_defaults="light",
        )
        defects_set.write_files(output_path=output_path, rattle=False, processes=1)

        assert len(defects_set) == len(defect_generator.defect_entries)
        assert os.path.isfile(os.path.join(output_path, "CdTe_defects_generator.json.gz"))
        reloaded_generator = loadfn(os.path.join(output_path, "CdTe_defects_generator.json.gz"))
        assert set(reloaded_generator.defect_entries) == set(defect_generator.defect_entries)
        assert reloaded_generator.bulk_supercell == defect_generator.bulk_supercell

        for defect_name in defect_generator.defect_entries:
            defect_dir = os.path.join(output_path, defect_name)
            for subfolder in ("aims_gam", "aims_std"):
                subfolder_dir = os.path.join(defect_dir, subfolder)
                assert os.path.isfile(os.path.join(subfolder_dir, "parameters.json"))
                assert os.path.isfile(os.path.join(subfolder_dir, f"{defect_name}.json.gz"))
                self._check_valid_control_and_geometry(subfolder_dir, subfolder)

        bulk_dir = os.path.join(output_path, "CdTe_bulk")
        for subfolder in ("aims_gam", "aims_std"):
            self._check_valid_control_and_geometry(os.path.join(bulk_dir, subfolder), subfolder)

    def _check_valid_control_and_geometry(self, subfolder_dir, subfolder):
        """Check that ``control.in``/``geometry.in`` in ``subfolder_dir`` are valid,
        self-consistent FHI-aims inputs for a periodic calculation (i.e. a `k`-point
        grid is specified, matching ``aims_gam``/``aims_std`` as appropriate, and the
        total charge in ``control.in`` matches the summed ``initial_charge``\\s in
        ``geometry.in``)."""
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
        else:  # aims_std
            assert not k_grid_matches
            assert k_grid_density_matches == ["5.0"]

        charge_matches = re.findall(r"^\s*charge\s+([+-]?[0-9]+(?:\.[0-9]*)?)", control, re.MULTILINE)
        assert len(charge_matches) == 1
        control_charge = float(charge_matches[0])
        geometry_initial_charge = sum(
            float(match)
            for match in re.findall(r"^\s*initial_charge\s+([+-]?[0-9]+(?:\.[0-9]*)?)", geometry, re.MULTILINE)
        )
        assert geometry_initial_charge == control_charge


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
