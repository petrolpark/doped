"""
Code to generate FHI-aims defect calculation input files.
"""

from os import PathLike

import pymatgen.io.aims as aims

from doped.core import DefectEntry
from doped.generation import DefectsGenerator
from doped.vasp import DefectsSet

class AimsDefectsSet(DefectsSet): # currently subclassing from VASP DefectsSet, but should be replaced with abstract base class

    """
    Class for generating input files (``geometry.in`` and ``control.in``) for ``FHI-aims`` defect calculations for a set
    of ``doped``/``pymatgen`` ``DefectEntry`` objects.
    """

    def __init__(
        self,
        defect_entries: DefectsGenerator | dict[str, DefectEntry] | list[DefectEntry] | DefectEntry
    ):
        r"""
        
        """
        super().__init__(defect_entries) # VASP DefectsSet handles getting the DefectEntries in the right format

    def write_files(
        self,
        output_path: PathLike = ".",
        # poscar: bool = False,
        rattle: bool = True,
        # vasp_gam: bool | None = None,
        bulk: bool | str = True,
        # processes: int | None = None,
        **kwargs,
    ):
        r"""

        """

