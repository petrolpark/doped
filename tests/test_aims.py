import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from monty.serialization import loadfn
from pymatgen.core import SETTINGS
from test_utils import EXAMPLE_DIR, data_dir

# temp
SETTINGS["AIMS_SPECIES_DIR"] = "~/Documents/fhi-aims.260331/species_defaults"

from doped.generation import DefectsGenerator
from doped.io.aims.inputs import AimsDefectsSet, _resolve_species_defaults
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

        defects_set = AimsDefectsSet(
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
