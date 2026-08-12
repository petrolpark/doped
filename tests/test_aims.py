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
from test_utils import EXAMPLE_DIR, data_dir, vasp_data_dir

# temp
SETTINGS["AIMS_SPECIES_DIR"] = "~/Documents/fhi-aims.260331/species_defaults"

from doped.analysis import DefectParser
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
    ``test_analysis.py``) requires ``projected_eigenvalues``, which the
    aims backend does not yet parse (see ``doped/io/aims/outputs.py``), so
    raises ``FileNotFoundError`` for aims-parsed data; and
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
