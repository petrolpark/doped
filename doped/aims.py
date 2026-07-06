"""
Code to generate FHI-aims defect calculation input files.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.aims.sets import \
    AimsInputSet  # TODO remove dependency on pre-alpha package
from pymatgen.util.typing import PathLike

from doped.core import DefectEntry
from doped.generation import DefectsGenerator, get_defect_name_from_entry
from doped.utils.parsing import _get_bulk_supercell, _get_defect_supercell


class DopedAimsInputSet(AimsInputSet):
    """
    Extension to ``pymatgen-io-aims`` ``AimsInputSet`` object for ``FHI-aims`` defect calculations,
    including SnB rattling functionality
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
        super.__init__(parameters, structure, properties)

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
            self.structure: Structure = SnB_rattle(self.structure, stdev=stdev, d_min=d_min)
        
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
        user_parameters: dict[str, Any] = {},
        user_properties: Sequence[str] = [],
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
            
        """

        self.defect_entry = defect_entry
        self.charge_state = charge_state or defect_entry.charge_state

        if isinstance(self.defect_entry, Structure):
            self.defect_supercell = self.defect_entry
            self.bulk_supercell = None
        elif isinstance(self.defect_entry, DefectEntry):
            self.defect_supercell = _get_defect_supercell(self.defect_entry)
            self.bulk_supercell = _get_bulk_supercell(self.defect_entry)
        else:
            raise TypeError("defect_entparameters=sry must be a doped/pymatgen DefectEntry or Structure object.")
    
        #TODO probably need some required properties in user_properties in order to get the info needed for parsing

    @property
    def aims_input_set(
        self,
    ) -> DopedAimsInputSet:
        return DopedAimsInputSet(
            parameters=self.user_parameters,
            structure=self.defect_supercell,
            properties=self.user_properties
        )

    #TODO currently copied from VASP implementation. would like to move this to ABC for both VASP, aims and others
    def _get_output_path(self, defect_dir: PathLike | None = None, subfolder: PathLike | None = None):
        if defect_dir is None:
            if self.defect_entry.name is None:
                self.defect_entry.name = get_defect_name_from_entry(self.defect_entry, relaxed=False)

            defect_dir = self.defect_entry.name

        return f"{defect_dir}/{subfolder}" if subfolder is not None else defect_dir

    def _write_aims_xxx_files(
        self,
        defect_dir: PathLike | None,
        subfolder: PathLike | None,
        rattle: bool,
        aims_xxx_attribute: AimsInputSet
    ):
        output_path = self._get_output_path(defect_dir, subfolder)

        stdev = d_min = None
        if rattle:
            for trial_structure in [
                self.defect_entry.defect.structure,
                self.defect_entry.bulk_supercell,
                self.defect_entry.bulk_supercell * 2,
            ]:
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
            # **kwargs,  # kwargs to allow POTCAR testing on GH Actions
        )

        if "bulk" not in defect_dir:  # not a bulk supercell
            self.defect_entry.to_json(f"{output_path}/{self.defect_entry.name}.json.gz")

    def write_all(
        self,
        defect_dir: PathLike | None = None,
        rattle: bool = False,
    ):
        r"""
        Write FHI-aims input files to subfolders in the ``defect_dir`` folder.

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
        
        self.aims_input_set.write_input(

        )

class AimsDefectsSet(MSONable):
    """
    Class for generating input files for ``FHI-aims`` defect calculations for a set
    of ``doped``/``pymatgen`` ``DefectEntry`` objects.
    """

    def __init__(
        self,
        defect_entries: DefectsGenerator | dict[str, DefectEntry] | list[DefectEntry] | DefectEntry,
    ):
        r"""
        
        """