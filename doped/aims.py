"""
Code to generate FHI-aims defect calculation input files.
"""

import os
from os import PathLike

from doped.core import DefectEntry
from doped.generation import DefectsGenerator
from doped.vasp import DefectRelaxSet, DefectsSet

from pymatgen.io.aims.inputs import AimsGeometryIn, AimsControlIn

class AimsDefectRelaxSet(DefectRelaxSet): # currently subclassing from VASP DefectRelaxSet, but should be replaced with abstract base class. we ignore the existing write_xyz methods
    """
    Class for generating input files for ``FHI-aims`` defect relaxation
    calculations for a single ``pymatgen`` ``DefectEntry`` or ``Structure``
    object.
    """

    def __init__(
        self,
        defect_entry: DefectEntry,
        charge_state: int | None = None,
        **kwargs,
    ):
        r"""
        
        """
        super().__init__(defect_entry, charge_state=charge_state, **kwargs)

    def write(
        self,
        defect_dir: PathLike | None = None,
        **kwargs
    ):
        geometryIn = AimsGeometryIn.from_structure(self.defect_supercell)
        geometryIn.write_file(directory=defect_dir)
        
        

class AimsDefectsSet(DefectsSet): # currently subclassing from VASP DefectsSet, but should be replaced with abstract base class

    """
    Class for generating input files (``geometry.in`` and ``control.in``) for ``FHI-aims`` defect calculations for a set
    of ``doped``/``pymatgen`` ``DefectEntry`` objects.
    """

    def __init__(
        self,
        defect_entries: DefectsGenerator | dict[str, DefectEntry] | list[DefectEntry] | DefectEntry,
        **kwargs
    ):
        r"""
        
        """

        self.defect_sets: dict[str, AimsDefectRelaxSet] = {} # overload the VASP defect sets
        for defect_species, defect_entry in defect_entries.items():
            self.defect_sets[defect_species] = AimsDefectRelaxSet(
                defect_entry=defect_entry,
                charge_state=defect_entry.charge_state,
                # # VASP-specific settings that are not relevant for FHI-aims, so commented out for now:
                # user_incar_settings=self.user_incar_settings,
                # user_kpoints_settings=self.user_kpoints_settings,
                # usdefect_dir: PathLike | None = None,er_potcar_functional=self.user_potcar_functional,
                # user_potcar_settings=self.user_potcar_settings,
                **self.kwargs,
            )

        super().__init__(defect_entries, kwargs=kwargs) # VASP DefectsSet handles getting the DefectEntries in the right format. this should be a static or super method

    def _write_defect(args):
        defect_species, defect_relax_set, output_path, poscar, rattle, vasp_gam, bulk, kwargs = args
        defect_dir = PathLike(os.path.join(output_path, defect_species))
        defect_relax_set.write_all(
            defect_dir=defect_dir,
            # poscar=poscar,
            # rattle=rattle,
            # vasp_gam=vasp_gam,
            # bulk=bulk,
            **kwargs,
        )