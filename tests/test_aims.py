import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from monty.serialization import loadfn
from pymatgen.core import SETTINGS
from test_utils import EXAMPLE_DIR

# temp
SETTINGS["AIMS_SPECIES_DIR"] = "/home/hw653/Documents/fhi-aims.260331/species_defaults"

from doped.aims.inputs import AimsDefectsSet, _resolve_species_defaults
from doped.generation import DefectsGenerator
from doped.utils.efficiency import Structure


class AimsTest(unittest.TestCase):

    def setUp(self):
        self.prim_cdte = Structure.from_file(f"{EXAMPLE_DIR}/CdTe/relaxed_primitive_POSCAR")

    def test_species_defaults_path_resolution(self):
        """Resolve the default light species_defaults shorthand."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            species_defaults = Path(tmp_dir) / "species_defaults"
            basis_path = species_defaults / "defaults_2020" / "light"
            basis_path.mkdir(parents=True)
            assert _resolve_species_defaults(basis_path) == str(basis_path.resolve())
            with patch("doped.aims.inputs.SETTINGS", {"AIMS_SPECIES_DIR": str(species_defaults)}):
                assert _resolve_species_defaults("light") == str(basis_path.resolve())
                assert _resolve_species_defaults(Path("defaults_2020") / "light") == str(basis_path.resolve())

    def test_CdTe_defect_input_generation(self):
        """Generate CdTe defects and write an FHI-aims input set for each."""
        defect_generator = DefectsGenerator(self.prim_cdte)

        with tempfile.TemporaryDirectory() as output_path:
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
                control_path = os.path.join(defect_dir, "control.in")
                geometry_path = os.path.join(defect_dir, "geometry.in")
                assert os.path.isfile(control_path)
                assert os.path.isfile(geometry_path)
                assert os.path.isfile(os.path.join(defect_dir, "parameters.json"))
                assert os.path.isfile(os.path.join(defect_dir, f"{defect_name}.json.gz"))

                with open(control_path, encoding="utf-8") as control_file:
                    control = control_file.read()
                with open(geometry_path, encoding="utf-8") as geometry_file:
                    geometry = geometry_file.read()

                charge_matches = re.findall(r"^\s*charge\s+([+-]?[0-9]+(?:\.[0-9]*)?)", control, re.MULTILINE)
                assert len(charge_matches) == 1
                control_charge = float(charge_matches[0])
                geometry_initial_charge = sum(
                    float(match)
                    for match in re.findall(r"^\s*initial_charge\s+([+-]?[0-9]+(?:\.[0-9]*)?)", geometry, re.MULTILINE)
                )
                assert geometry_initial_charge == control_charge

            bulk_dir = os.path.join(output_path, "CdTe_bulk")
            assert os.path.isfile(os.path.join(bulk_dir, "control.in"))
            assert os.path.isfile(os.path.join(bulk_dir, "geometry.in"))
