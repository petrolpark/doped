"""
Parsing of FHI-aims defect / bulk supercell calculation outputs.

These functions load and process FHI-aims output files (``aims.out``, and --
for charge corrections, not yet implemented -- ``atom_proj_dos`` files), and
can provide the parsed outputs in calculator-agnostic form
(:class:`~doped.io.outputs.CalculationOutputs`) via :func:`get_calculation_outputs`.
"""

import numpy as np
from pyfhiaims.outputs.stdout import AimsStdout
from pymatgen.core.structure import Structure
from pymatgen.util.typing import PathLike

from doped.io.outputs import CalculationOutputs
from doped.io.utils import find_archived_fname

# --------------------------------------------------------------------------
# Backend protocol entry points (see the "Adding Support for a New
# Calculator" docs page)
# --------------------------------------------------------------------------

CALC_OUTPUT_MASK = ("aims.out", "aims.out.gz")
"""
Filename patterns identifying (FHI-aims) calculation output files, used for
calculation folder discovery.

Part of the ``doped.io`` backend protocol.
"""

SUBFOLDER_PRIORITY = ["aims_std", "aims_gam"]
"""
Priority order when auto-detecting (FHI-aims) calculation subfolders (see
``doped/io/aims/inputs.py``, which writes both an ``aims_std`` (non-Γ-only
`k`-point mesh) and ``aims_gam`` (Γ-point-only) subfolder per defect).

Part of the ``doped.io`` backend protocol.
"""

FILE_PARSING_ACTIONS = {
    "aims.out": "parse the calculation energy, structure and metadata.",
}
"""
The (FHI-aims) calculation output file types parsed by ``doped``, and what
they are used for (for informative warning messages).

Part of the ``doped.io`` backend protocol.
"""


def get_calculation_outputs(
    path: PathLike,
    label: str = "calculation",
    parse_projected_eigen: bool | None = None,
    **kwargs,
) -> CalculationOutputs:
    """
    Parse the outputs of an FHI-aims supercell calculation in ``path`` to a
    (calculator-agnostic) :class:`~doped.io.outputs.CalculationOutputs`
    object.

    Not yet implemented -- FHI-aims output parsing (and charge corrections)
    are still in development; see the other functions in this module for the
    individual parsing steps planned. ``label`` and ``parse_projected_eigen``
    are accepted (and currently ignored) to match the generic ``doped.io``
    backend protocol (used for informative warnings/parsing efficiency
    choices with other calculators).

    Part of the ``doped.io`` backend protocol.

    Args:
        path (PathLike):
            Path to the FHI-aims calculation directory (containing the
            ``aims.out(.gz)`` file to parse).
        label (str):
            Label for the type of calculation being parsed (e.g. ``"bulk"``,
            ``"defect"``), for informative warnings. Default is
            ``"calculation"``.
        parse_projected_eigen (bool):
            Whether to parse orbital projections, for eigenvalue / shallow
            defect analyses. Default is ``None``.
        **kwargs:
            Additional keyword arguments (currently unused).

    Returns:
        CalculationOutputs: The parsed calculation outputs.
    """
    raise NotImplementedError

def get_aims_output(aims_out_path: PathLike, **kwargs) -> AimsStdout:
    """
    Read the ``aims.out`` (stdout) file as a ``pyfhiaims`` ``AimsStdout`` object.

    Locates the file (handling ``.gz``/``.xz``/``.bz``/``.lzma`` compression,
    via ``doped.io.utils.find_archived_fname``) and wraps it in ``AimsStdout``.
    """
    aims_out_path = str(aims_out_path)
    try:
        return AimsStdout(find_archived_fname(aims_out_path), **kwargs)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"aims.out file not found at {aims_out_path}(.gz/.xz/.bz/.lzma)"
        ) from exc


def get_atom_projected_dos(atom_proj_dos_dir: PathLike, atom_index: int, **kwargs):
    """
    Parse the ``atom_proj_dos_<Species><index>.dat`` file for a given atom
    (output by FHI-aims when ``output atom_proj_dos`` is set in
    ``control.in``).

    Needed for core-level-shift-based potential alignment (see
    ``get_core_level_shift``) -- not currently parsed by ``pyfhiaims`` at all
    (only the ``aims.out`` stdout is parsed, via ``AimsStdout``; the
    ``atom_proj_dos_*.dat`` files are separate output files that would need
    their own reader).
    """
    raise NotImplementedError


def get_n_electrons_from_aims_output(aims_output: AimsStdout) -> int:
    """
    Get the number of electrons used in the calculation.

    FHI-aims directly prints "Formal number of electrons (from input
    files)", which ``pyfhiaims`` already parses into
    ``AimsStdout.header_summary["n_electrons"]``. So this is a direct read,
    not a reconstruction.
    """
    return aims_output.header_summary["n_electrons"]


def get_neutral_n_electrons(structure: Structure) -> int:
    """
    Get the number of electrons corresponding to a neutral charge state for
    ``structure``.

    Since FHI-aims is all-electron, this is just the sum of atomic numbers
    over the composition. Can reuse
    ``doped.utils.symmetry._num_electrons_from_charge_state(structure, 0)``
    directly.
    """
    from doped.utils.symmetry import \
        _num_electrons_from_charge_state  # avoid circular import (symmetry imports doped.core)

    return _num_electrons_from_charge_state(structure, 0)


def total_charge_from_aims_output(aims_output: AimsStdout) -> int:
    """
    Determine the total charge state of a system from its ``AimsStdout``.

    ``charge = get_neutral_n_electrons(structure) -
    get_n_electrons_from_aims_output(aims_output)``. This is exact, with no
    approximation or fallback logic needed.
    """
    structure = aims_output.get_image(-1).geometry.structure
    return get_neutral_n_electrons(structure) - get_n_electrons_from_aims_output(aims_output)


def get_magnetization_from_aims_output(aims_output: AimsStdout) -> float:
    """
    Get the total magnetization (``N_up - N_down``) of the system.

    FHI-aims directly prints the total spin ('N_up - N_down'), already
    parsed by ``pyfhiaims`` and exposed as
    ``AimsStdout.get_image(-1).results["magmom"]``. Direct read, not a
    reconstruction.
    """
    magmom = aims_output.get_image(-1).results.get("magmom")
    return 0.0 if magmom is None else float(magmom)


def get_atomic_magnetic_moments_from_aims_output(aims_output: AimsStdout) -> np.ndarray:
    """
    Get the per-atom Mulliken spin moments (requires ``output mulliken`` in
    ``control.in``).

    Exposed as ``AimsStdout.get_image(-1).results["mulliken_spins"]``.
    Useful for spin-density sanity-checking on defect supercells.
    """
    return np.array(aims_output.get_image(-1).results["mulliken_spins"])


def spin_degeneracy_from_aims_output(aims_output: AimsStdout, charge_state: int | None = None) -> int:
    """
    Get the spin degeneracy (multiplicity, ``2S + 1``) of the system.

    Uses ``get_magnetization_from_aims_output`` for the total magnetization.
    If ``charge_state`` is ``None``, the electron count for parity-checking
    comes from ``get_n_electrons_from_aims_output``, else from
    ``doped.utils.symmetry._num_electrons_from_charge_state``.
    """
    from doped.utils.symmetry import (  # avoid circular import (symmetry imports doped.core)
        _num_electrons_from_charge_state,
        _spin_degeneracy_from_num_electrons_and_magnetization)

    if charge_state is None:
        num_electrons = get_n_electrons_from_aims_output(aims_output)
    else:
        structure = aims_output.get_image(-1).geometry.structure
        num_electrons = _num_electrons_from_charge_state(structure, charge_state)

    magnetization = get_magnetization_from_aims_output(aims_output)
    return _spin_degeneracy_from_num_electrons_and_magnetization(int(num_electrons), magnetization)


def get_mulliken_charges_from_aims_output(aims_output: AimsStdout) -> np.ndarray:
    """
    Get the per-atom Mulliken charges (requires ``output mulliken``).

    Exposed as ``AimsStdout.get_image(-1).results["mulliken_charges"]``.
    """
    return np.array(aims_output.get_image(-1).results["mulliken_charges"])


def get_hirshfeld_charges_from_aims_output(aims_output: AimsStdout) -> np.ndarray:
    """
    Get the per-atom Hirshfeld charges (requires ``output hirshfeld``).

    Exposed as ``AimsStdout.get_image(-1).results["hirshfeld_charges"]``. A
    cheaper (but less rigorous) potential-alignment reference than
    core-level shifts (``get_core_level_shift``), and doesn't require a
    separate ``atom_proj_dos`` output file.
    """
    return np.array(aims_output.get_image(-1).results["hirshfeld_charges"])


def get_core_level_shift(
    defect_atom_proj_dos_dir: PathLike,
    bulk_atom_proj_dos_dir: PathLike,
    atom_index: int,
    core_level: str = "1s",
) -> float:
    r"""
    Get the core-level shift for a given atom, for use in potential
    alignment (the atomic-site-potential analog for FHI-aims defect
    calculations).

    Per the FHI-aims manual (Section 4.11, "Formation energies of charged
    defects"), potential alignment for charged-defect formation energies
    should align on the shift in core-state eigenvalues of an atom far from
    the defect, between the defect and (equivalent) bulk calculations,
    since FHI-aims is all-electron and so has no averaged core potential to
    align on instead:

    ``core_level_shift = (defect_core_eigenvalue - defect_VBM) -
    (bulk_core_eigenvalue - bulk_VBM)``

    Uses ``get_atom_projected_dos`` to obtain the core-state eigenvalue for
    ``atom_index`` in each calculation (matching ``core_level``, e.g.
    ``"1s"``).
    """
    raise NotImplementedError


def _get_bulk_site_potentials_aims(bulk_path: PathLike, **kwargs):
    """
    Orchestration wrapper: get the reference atomic-site potentials needed
    for the eFNV-style charge correction from a bulk FHI-aims calculation,
    via ``get_core_level_shift`` (needs ``output atom_proj_dos``).
    """
    raise NotImplementedError
