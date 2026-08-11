"""
Parsing of FHI-aims defect / bulk supercell calculation outputs.

These functions load and process FHI-aims output files (``aims.out``; ``atom_proj_dos`` files
for eFNV charge corrections; and ``hartree_potential`` cube files for FNV charge corrections),
and can provide the parsed outputs in calculator-agnostic form
(:class:`~doped.io.outputs.CalculationOutputs`) via :func:`get_calculation_outputs`.
"""

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io.cube import read_cube
from ase.units import Hartree
from pyfhiaims.outputs.stdout import AimsStdout
from pymatgen.core.structure import Structure
from pymatgen.util.typing import PathLike

from doped.io.aims.utils import (cube_header_angstrom_to_bohr,
                                  reshape_eigenvalues_and_occupations)
from doped.io.outputs import CalculationOutputs
from doped.io.utils import (_get_output_files_and_check_if_multiple,
                             _multiple_files_warning, find_archived_fname)

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
    "hartree_potential": "compute planar-averaged electrostatic potentials for the FNV "
    "(Freysoldt) charge correction.",
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
    load_site_potentials: bool = False,
    load_planar_averaged_potentials: bool = False,
    **kwargs,
) -> CalculationOutputs:
    """
    Parse the outputs of an FHI-aims supercell calculation in ``path`` to a
    (calculator-agnostic) :class:`~doped.io.outputs.CalculationOutputs`
    object.

    The ``aims.out(.gz)`` file in ``path`` is parsed for the final energy,
    structure and basic calculation metadata (see the ``get_X_from_aims_
    output`` functions in this module). FHI-aims output parsing (and charge
    corrections) is still in development, so several ``CalculationOutputs``
    fields are not yet populated here -- see the commented-out fields below.
    ``label`` and ``parse_projected_eigen`` are accepted (and currently
    ignored) to match the generic ``doped.io`` backend protocol (used for
    informative warnings/parsing efficiency choices with other calculators).

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
        load_site_potentials (bool):
            Whether to also parse the atomic-site potentials (via
            :func:`get_site_potentials`, from the ``atom_proj_dos`` files in
            ``path`` -- the same directory as the parsed ``aims.out``), for
            the Kumagai (eFNV) charge correction. Default is ``False``.
        load_planar_averaged_potentials (bool):
            Whether to also parse the planar-averaged electrostatic
            potentials (via :func:`get_hartree_potential_cube`, from the
            ``hartree_potential`` cube file in ``path``), for the FNV
            (Freysoldt) charge correction. Default is ``False``. If ``True``,
            the full parsed cube (atoms, 3D data grid, origin, spacing) is
            also retained (for possible future use) as
            ``CalculationOutputs.raw["hartree_potential_cube"]``.
        **kwargs:
            Additional keyword arguments passed to :func:`get_site_potentials`
            (e.g. ``core_level``, ``min_relative_height``) if
            ``load_site_potentials`` is ``True``; otherwise unused.

    Returns:
        CalculationOutputs: The parsed calculation outputs.
    """
    aims_out_path, multiple = _get_output_files_and_check_if_multiple("aims.out", path)
    if multiple:
        _multiple_files_warning(
            "aims.out", path, aims_out_path, action=FILE_PARSING_ACTIONS["aims.out"], dir_type=label
        )
    aims_output = get_aims_output(aims_out_path)
    image = aims_output.get_image(-1)
    header_summary = aims_output.header_summary
    # "eigenvalues"/"occupations" are filtered out of `image.results` (`get_results(verbosity=
    # "converged")`), so need `verbosity="all"` here:
    all_results = image.get_results(verbosity="all")

    site_potentials = None
    if load_site_potentials:
        site_potentials = get_site_potentials(path, dir_type=label, **kwargs)

    planar_averaged_potentials = None
    hartree_potential_cube = None
    if load_planar_averaged_potentials:
        hartree_potential_cube = get_hartree_potential_cube(path, dir_type=label)
        planar_averaged_potentials = _planar_average_from_cube(hartree_potential_cube)

    vbm, cbm, band_gap = get_band_edge_eigenvalues_from_aims_output(aims_output)

    eigenvalues = None
    if all_results.get("eigenvalues") is not None:
        eigenvalues = reshape_eigenvalues_and_occupations(
            all_results["eigenvalues"],
            all_results["occupations"],
            n_spins=header_summary["n_spins"],
            n_kpoints=header_summary["n_k_points"],
        )

    return CalculationOutputs(
        structure=image.geometry.structure,
        energy=image.results["total_energy"],
        calculator="aims",
        directory=path,
        converged_electronic=image.converged,
        converged_ionic=aims_output.geometry_converged,  # None if not a relaxation
        efermi=image.results.get("fermi_energy"),
        eigenvalues=eigenvalues,
        # projected_eigenvalues=None,  # TODO: orbital-projected eigenvalue parsing not yet implemented
        # projected_magnetisation=None,  # TODO: non-collinear projected magnetisation not yet implemented
        kpoint_coords=_as_array_or_none(header_summary["k_points"]),
        kpoint_weights=_as_array_or_none(header_summary["k_point_weights"]),
        nelect=get_n_electrons_from_aims_output(aims_output),
        charge=total_charge_from_aims_output(aims_output),
        magnetization=get_magnetization_from_aims_output(aims_output),
        # noncollinear=None,  # TODO: determine from `aims_output.metadata`/`control.in`, not yet
        #   implemented
        vbm=vbm,
        cbm=cbm,
        band_gap=band_gap,
        planar_averaged_potentials=planar_averaged_potentials,
        site_potentials=site_potentials,
        run_metadata=dict(aims_output.metadata),
        raw={"aims_output": aims_output, "hartree_potential_cube": hartree_potential_cube},
    )


def _as_array_or_none(values: list | None) -> np.ndarray | None:
    """
    Convert ``values`` to a ``np.ndarray``, or return ``None`` if ``values``
    is ``None`` (as ``aims_output.header_summary["k_points"]``/
    ``["k_point_weights"]`` are, if ``output k_point_list`` was not set in
    ``control.in``).
    """
    return None if values is None else np.array(values)


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


_ATOM_PROJ_DOS_RE = re.compile(r"^atom_proj_dos_[A-Za-z]+(\d+)(_raw)?\.dat$")
"""
Matches FHI-aims ``atom_proj_dos_<Species><index>(_raw).dat`` filenames,
capturing the (1-indexed) atom index and whether it is the ``_raw`` variant.

Assumes species symbols contain only letters (standard element symbols),
matching all observed FHI-aims output; species names with trailing digits
would be ambiguous with the atom index and are not supported.
"""


def _find_atom_proj_dos_file(atom_proj_dos_dir: PathLike, atom_index: int, raw: bool) -> Path:
    """
    Locate the ``atom_proj_dos_<Species><atom_index>(_raw).dat`` file for
    ``atom_index`` (1-indexed, matching FHI-aims' ``geometry.in`` atom
    ordering) in ``atom_proj_dos_dir``.
    """
    atom_proj_dos_dir = Path(atom_proj_dos_dir)
    matches = [
        f
        for f in atom_proj_dos_dir.iterdir()
        if (match := _ATOM_PROJ_DOS_RE.match(f.name))
        and int(match.group(1)) == atom_index
        and bool(match.group(2)) == raw
    ]
    if not matches:
        raise FileNotFoundError(
            f"No `atom_proj_dos_<Species>{atom_index}{'_raw' if raw else ''}.dat` file found in "
            f"{atom_proj_dos_dir} (`output atom_proj_dos` must be set in `control.in`)."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple `atom_proj_dos` files found for atom index {atom_index} (raw={raw}) in "
            f"{atom_proj_dos_dir}: {matches}"
        )
    return matches[0]


def get_atom_projected_dos(atom_proj_dos_dir: PathLike, atom_index: int, raw: bool = True, **kwargs) -> pd.DataFrame:
    """
    Parse the ``atom_proj_dos_<Species><index>(_raw).dat`` file for a given
    atom (output by FHI-aims when ``output atom_proj_dos`` is set in
    ``control.in``; see ``AIMS_DefectSet.yaml``).

    Needed for core-level-based potential alignment (see
    ``get_core_level_eigenvalue``/``get_site_potentials``).

    FHI-aims writes two energy references per atom: the plain file is
    referenced to the chemical potential (Fermi level/VBM), while the
    ``_raw`` file is referenced to the internal zero (vacuum level for
    non-periodic systems, or the G=0 component of the long-range Hartree
    potential for periodic systems). The ``_raw`` reference is shared by
    both bulk and defect supercell calculations, so is the one used for
    potential alignment (``raw=True``, the default).

    Args:
        atom_proj_dos_dir (PathLike):
            Directory containing the ``atom_proj_dos_*.dat`` files (e.g. the
            ``aims_std``/``aims_gam`` calculation subfolder).
        atom_index (int):
            1-indexed atom number, matching the atom ordering in
            ``geometry.in`` (and thus the ``<index>`` in the
            ``atom_proj_dos_<Species><index>.dat`` filename).
        raw (bool):
            Whether to parse the ``_raw`` (internal-zero-referenced) or
            plain (chemical-potential-referenced) file. Default: ``True``.
        **kwargs:
            Additional keyword arguments (currently unused).

    Returns:
        pd.DataFrame: Columns ``"energy"`` (eV) and ``"total"``, plus one
        column per angular-momentum channel present in the file (``"l0"``
        for l=0/s, ``"l1"`` for l=1/p, etc).
    """
    dos_file = _find_atom_proj_dos_file(atom_proj_dos_dir, atom_index, raw=raw)
    dos = pd.read_csv(dos_file, comment="#", sep=r"\s+", header=None)
    n_l_channels = dos.shape[1] - 2
    dos.columns = ["energy", "total", *(f"l{l_num}" for l_num in range(n_l_channels))]
    return dos


def get_n_electrons_from_aims_output(aims_output: AimsStdout) -> int:
    """
    Get the number of electrons used in the calculation.

    FHI-aims directly prints "Formal number of electrons (from input
    files)", which ``pyfhiaims`` already parses into
    ``AimsStdout.header_summary["n_electrons"]``. However, this printed value
    is computed from the (neutral) input geometry/species alone, *before*
    the ``charge`` keyword (if set in ``control.in``) is applied to
    renormalise the electron count -- so any explicit ``charge`` must be
    subtracted here to get the actual number of electrons in the
    calculation.
    """
    charge = float(aims_output.metadata.get("charge", 0))
    return int(aims_output.header_summary["n_electrons"] - charge)


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


def get_band_edge_eigenvalues_from_aims_output(
    aims_output: AimsStdout,
) -> tuple[float | None, float | None, float | None]:
    """
    Get the (VBM, CBM, band gap) eigenvalues from an ``AimsStdout``.

    FHI-aims directly prints the "Highest occupied state (VBM)", "Lowest
    unoccupied state (CBM)" and overall "HOMO-LUMO gap" (already summed over
    k-points and spin channels) at the end of each SCF cycle, already parsed
    by ``pyfhiaims`` and exposed as ``AimsStdout.get_image(-1).results
    ["vbm"]``/``["cbm"]``/``["gap"]``. Direct read, not a reconstruction.
    All three are referenced to the internal zero (see
    ``get_atom_projected_dos``), consistent with ``efermi``.
    """
    results = aims_output.get_image(-1).results
    return results.get("vbm"), results.get("cbm"), results.get("gap")


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


_L_LETTER_TO_INT = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


def _l_from_core_level(core_level: str) -> int:
    """
    Get the angular momentum quantum number corresponding to ``core_level``
    (e.g. ``"s"``, ``"1s"``, ``"2p"`` -> 0, 0, 1 respectively).

    ``atom_proj_dos`` files only resolve the DOS by angular momentum, not by
    principal quantum number, so only the trailing letter of ``core_level``
    is used; any leading digit is ignored.
    """
    letter = core_level.strip()[-1].lower()
    try:
        return _L_LETTER_TO_INT[letter]
    except KeyError:
        raise ValueError(
            f"Unrecognised angular-momentum letter '{letter}' in `core_level='{core_level}'` -- "
            f"expected one of {sorted(_L_LETTER_TO_INT)}."
        ) from None


def _lowest_energy_peak(energy: np.ndarray, dos: np.ndarray, min_relative_height: float = 0.1) -> float:
    """
    Get the energy of the lowest-energy (most negative) resolvable local
    maximum in ``dos(energy)``, used to locate the core-like state in an
    angular-momentum-resolved ``atom_proj_dos`` channel.

    A local maximum is only considered "resolvable" if its height exceeds
    ``min_relative_height`` times the maximum DOS value in the channel, to
    exclude numerical noise/shoulders on the tails of broader (valence/
    conduction) bands. ``energy`` is assumed sorted in ascending order (as
    written by FHI-aims).
    """
    if len(dos) < 3:
        raise ValueError("`atom_proj_dos` array is too short to locate a peak.")
    threshold = min_relative_height * np.max(dos)
    is_local_max = (dos[1:-1] > dos[:-2]) & (dos[1:-1] > dos[2:]) & (dos[1:-1] > threshold)
    peak_indices = np.flatnonzero(is_local_max) + 1
    if len(peak_indices) == 0:
        raise ValueError(
            "No resolvable peak found in the requested `atom_proj_dos` angular-momentum channel "
            "above `min_relative_height`; try lowering `min_relative_height`, or check that the "
            "`atom_proj_dos` energy window/`core_level` are appropriate for this atom."
        )
    return float(energy[peak_indices[0]])


def get_core_level_eigenvalue(
    atom_proj_dos_dir: PathLike,
    atom_index: int,
    core_level: str = "s",
    min_relative_height: float = 0.1,
) -> float:
    r"""
    Get the core-level eigenvalue for a given atom in a single FHI-aims
    calculation (bulk or defect supercell), for use in potential alignment
    (the FHI-aims analog of VASP's atomic-site potentials from ``OUTCAR``;
    see ``get_site_potentials``).

    Per the FHI-aims manual (Section 4.11, "Formation energies of charged
    defects"), potential alignment for charged-defect formation energies
    should align on the shift in core-state eigenvalues of an atom far from
    the defect, between the defect and (equivalent) bulk calculations, since
    FHI-aims is all-electron and so has no averaged core potential to align
    on instead. As FHI-aims does not directly print core-state eigenvalues,
    this is obtained from the atom-projected density of states
    (``get_atom_projected_dos``, using the internal-zero-referenced ``_raw``
    file), by locating the lowest-energy resolvable peak in the
    ``core_level`` angular-momentum channel (see ``_l_from_core_level`` --
    ``atom_proj_dos`` only resolves the DOS by angular momentum, not
    principal quantum number).

    Note that no VBM referencing is applied here, unlike the naive
    alignment formula given in the FHI-aims manual (``core_level_shift =
    (defect_core_eigenvalue - defect_VBM) - (bulk_core_eigenvalue -
    bulk_VBM)``); the returned eigenvalue is on the internal-zero (``_raw``)
    reference, which is common to bulk and defect supercell calculations
    (unlike the VBM, which shifts with charge state/supercell), so the
    defect-bulk difference computed generically in ``doped.corrections``
    already gives the correct potential shift without it.

    Args:
        atom_proj_dos_dir (PathLike):
            Directory containing the ``atom_proj_dos_*.dat`` files for this
            calculation (e.g. the ``aims_std``/``aims_gam`` calculation
            subfolder).
        atom_index (int):
            1-indexed atom number, matching the atom ordering in
            ``geometry.in``.
        core_level (str):
            Angular-momentum channel to search, given by its trailing
            letter (e.g. ``"s"``, ``"1s"``, ``"2p"``). Default: ``"s"``.
        min_relative_height (float):
            Minimum peak height, as a fraction of the maximum DOS value in
            the chosen channel, to be considered a resolvable peak (see
            ``_lowest_energy_peak``). Default: 0.1.

    Returns:
        float: The core-level eigenvalue (eV), referenced to the internal
        zero (vacuum level / G=0 Hartree potential component).
    """
    l_num = _l_from_core_level(core_level)
    dos = get_atom_projected_dos(atom_proj_dos_dir, atom_index, raw=True)
    column = f"l{l_num}"
    if column not in dos.columns:
        available = [c for c in dos.columns if c.startswith("l")]
        raise ValueError(
            f"Angular momentum channel l={l_num} (from `core_level='{core_level}'`) not present "
            f"in the `atom_proj_dos` file for atom {atom_index} in {atom_proj_dos_dir}; available "
            f"channels: {available}."
        )
    return _lowest_energy_peak(
        dos["energy"].to_numpy(), dos[column].to_numpy(), min_relative_height=min_relative_height
    )


SITE_POTENTIALS_FILE = "atom_proj_dos"
"""
Substring identifying (FHI-aims) atom-projected DOS files, used generically
by ``doped.analysis`` to check for the presence of eFNV charge-correction
data, and in informative warning/error messages.

Part of the ``doped.io`` backend protocol.
"""


def get_site_potentials(
    path: PathLike,
    dir_type: str = "bulk",
    quiet: bool = False,
    outputs: CalculationOutputs | None = None,
    total_energy: list | float | None = None,
    core_level: str = "s",
    min_relative_height: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """
    Get the atomic-site electrostatic potentials for the FHI-aims
    calculation in ``path``, needed for Kumagai (eFNV) finite-size charge
    corrections.

    FHI-aims is an all-electron code with no averaged core potential to
    align on directly, so the site potentials are instead obtained from the
    atomic core-level eigenvalues, located via the atom-projected DOS (see
    ``get_core_level_eigenvalue`` and the FHI-aims manual, Section 4.11).
    The core eigenvalues are read from the ``_raw`` (internal-zero-
    referenced) ``atom_proj_dos`` files, since that reference is common to
    both the bulk and defect supercell calculations (unlike, e.g., the VBM,
    which shifts with charge state/supercell -- no VBM referencing is used
    here). As with the analogous VASP function (``-1 *`` the ``OUTCAR`` core
    potentials), the returned potentials are the negative of the core
    eigenvalues, so that ``defect_site_potentials - bulk_site_potentials``
    (computed generically in ``doped.corrections``) has the correct sign
    for the eFNV correction.

    ``path`` should be the exact calculation directory containing the
    ``atom_proj_dos_*.dat`` files (e.g. the ``aims_std``/``aims_gam``
    subfolder), not (necessarily) its parent.

    Part of the ``doped.io`` backend protocol.

    Args:
        path (PathLike):
            Path to the FHI-aims calculation directory for which to compute
            the atomic-site potentials.
        dir_type (str):
            Type of directory being parsed (``"bulk"`` or ``"defect"``), for
            informative warnings/errors. Default: ``"bulk"``.
        quiet (bool):
            Currently unused for FHI-aims (accepted to match the generic
            ``doped.io`` backend protocol).
        outputs (CalculationOutputs):
            Parsed calculation outputs, if already available; used to get
            the number of atoms directly from ``outputs.structure`` rather
            than inferring it from the number of ``atom_proj_dos`` files
            found in ``path`` (relevant if ``atom_proj_dos`` was only
            requested for a subset of atoms).
        total_energy (list | float):
            Not currently used for FHI-aims (accepted to match the generic
            ``doped.io`` backend protocol).
        core_level (str):
            Angular-momentum channel to search for the core-like peak in
            each atom's projected DOS (see ``get_core_level_eigenvalue``).
            Default: ``"s"``.
        min_relative_height (float):
            Minimum peak height (relative to the channel maximum) for a
            peak to be considered resolvable (see ``_lowest_energy_peak``).
            Default: 0.1.
        **kwargs:
            Additional keyword arguments (currently unused).

    Returns:
        np.ndarray: The atomic-site electrostatic potentials (in eV), one
        per site, ordered to match the sites of the corresponding structure.
    """
    path = Path(path)
    if outputs is not None and outputs.structure is not None:
        n_atoms = len(outputs.structure)
    else:
        raw_dos_files = list(path.glob("atom_proj_dos_*_raw.dat"))
        if not raw_dos_files:
            raise FileNotFoundError(
                f"No `atom_proj_dos_*_raw.dat` files found in {dir_type} directory {path}; these "
                f"are required to compute atomic-site potentials for the eFNV charge correction "
                f"with FHI-aims (`output atom_proj_dos` must be set in `control.in`)."
            )
        n_atoms = len(raw_dos_files)

    return np.array(
        [
            -1
            * get_core_level_eigenvalue(
                path, atom_index, core_level=core_level, min_relative_height=min_relative_height
            )
            for atom_index in range(1, n_atoms + 1)
        ]
    )


HARTREE_TO_EV = Hartree  # 1 Hartree in eV (``ase.units.Hartree``)

PLANAR_POTENTIAL_CUBE_FILE = "hartree_potential"
"""
Substring identifying the FHI-aims ``hartree_potential`` cube-file output, used generically by
``doped.analysis`` to check for the presence of FNV charge-correction data, and in informative
warning/error messages.

Part of the ``doped.io`` backend protocol.
"""


def get_hartree_potential_cube(path: PathLike, dir_type: str = "bulk") -> dict:
    r"""
    Parse the FHI-aims ``hartree_potential`` cube file for the calculation in
    ``path``, giving the full-electrostatic-potential grid, needed for the
    FNV (Freysoldt) finite-size charge correction (see
    ``get_planar_averaged_potentials``).

    Locates the ``hartree_potential*.cube`` file output by FHI-aims when
    ``output cube hartree_potential`` is set in ``control.in`` (requested by
    default in ``doped``-generated ``control.in`` files; see
    ``doped.io.aims.inputs._planar_potential_cube``).

    Unlike the standard Gaussian cube-file format (which FHI-aims' cube
    output otherwise follows), FHI-aims writes the header coordinates
    (origin, grid vectors, atom positions) in Angstrom rather than Bohr (see
    the FHI-aims manual, Section 4.5, "Visualizing charge densities and
    orbitals": "Although these files are written by default in Å, some
    programs (including jmol), read them in atomic units (bohr) by
    default."). Generic cube-file readers (e.g. ``ase.io.cube``) assume the
    standard Bohr convention and would silently mis-scale an FHI-aims cube
    file's header coordinates if used directly, so the header is first
    converted to the standard Bohr convention (see ``doped.io.aims.utils.
    cube_header_angstrom_to_bohr``), then parsed with ``ase.io.cube.
    read_cube`` -- giving the *full* parsed cube (atoms, 3D data grid,
    origin, spacing), for possible future use beyond the planar average
    computed in ``get_planar_averaged_potentials`` (e.g. visualisation, or
    alternative correction schemes).

    Note that the sign/absolute-reference convention of FHI-aims'
    ``hartree_potential`` cube output, relative to that of VASP's ``LOCPOT``
    (which the downstream ``pymatgen-analysis-defects`` Freysoldt correction
    code was written against), has not yet been empirically checked against
    a real defect/bulk calculation pair -- treat FNV corrections computed
    from this with some caution until verified.

    Part of the ``doped.io`` backend protocol.

    Args:
        path (PathLike):
            Path to the FHI-aims calculation directory containing the
            ``hartree_potential*.cube`` file.
        dir_type (str):
            Type of directory being parsed (``"bulk"`` or ``"defect"``), for
            informative warnings. Default: ``"bulk"``.

    Returns:
        dict: As returned by ``ase.io.cube.read_cube`` (keys ``"atoms"``,
        ``"data"``, ``"origin"``, ``"spacing"``, ``"labels"``, ``"datas"``),
        with ``"data"``/``"datas"`` converted from Hartree to eV.
        ``"origin"``/``"spacing"``/``"atoms"`` are in Angstrom, as returned
        by ``ase`` (the Bohr-to-Angstrom conversion cancels out the
        Angstrom-to-Bohr conversion applied beforehand).
    """
    cube_path, multiple = _get_output_files_and_check_if_multiple(
        f"{PLANAR_POTENTIAL_CUBE_FILE}.cube", path, search_patterns=[PLANAR_POTENTIAL_CUBE_FILE]
    )
    if multiple:
        _multiple_files_warning(
            PLANAR_POTENTIAL_CUBE_FILE,
            path,
            cube_path,
            action=FILE_PARSING_ACTIONS[PLANAR_POTENTIAL_CUBE_FILE],
            dir_type=dir_type,
        )
    if not Path(cube_path).exists():
        raise FileNotFoundError(
            f"No `{PLANAR_POTENTIAL_CUBE_FILE}*.cube` file found in {dir_type} directory {path}; "
            f"this is required to compute planar-averaged potentials for the FNV charge correction "
            f"with FHI-aims (`output cube hartree_potential` must be set in `control.in`)."
        )

    with open(cube_path) as f:
        cube_text = f.read()

    dct = read_cube(io.StringIO(cube_header_angstrom_to_bohr(cube_text)), read_data=True)
    dct["data"] = dct["data"] * HARTREE_TO_EV
    dct["datas"] = dct["datas"] * HARTREE_TO_EV
    return dct


def get_planar_averaged_potentials(path: PathLike, dir_type: str = "bulk") -> dict[int, np.ndarray]:
    r"""
    Get the planar-averaged electrostatic potential along each lattice
    vector for the FHI-aims calculation in ``path``, needed for the FNV
    (Freysoldt) finite-size charge correction.

    Averages the 3D grid from :func:`get_hartree_potential_cube` over the
    two dimensions perpendicular to each grid axis in turn. As that cube
    grid is generated with its edges set explicitly proportional to the
    structure's lattice vectors (rather than FHI-aims' internal default,
    which is Cartesian-axis-aligned), grid axis ``i`` corresponds directly
    to lattice vector ``i`` -- so this is the same "planar average along a
    lattice vector" quantity as ``pymatgen``'s ``Locpot.
    get_average_along_axis`` (used for the analogous VASP FNV correction),
    including for non-orthogonal cells.

    Part of the ``doped.io`` backend protocol.

    Args:
        path (PathLike):
            Path to the FHI-aims calculation directory containing the
            ``hartree_potential*.cube`` file.
        dir_type (str):
            Type of directory being parsed (``"bulk"`` or ``"defect"``), for
            informative warnings. Default: ``"bulk"``.

    Returns:
        dict[int, np.ndarray]: ``{axis index: 1D array}`` (in eV), for axis
        in ``[0, 1, 2]`` (matching the ``a``, ``b``, ``c`` lattice vectors).
    """
    return _planar_average_from_cube(get_hartree_potential_cube(path, dir_type=dir_type))


def _planar_average_from_cube(cube: dict) -> dict[int, np.ndarray]:
    """
    Average the 3D ``cube["data"]`` grid (as parsed by
    ``get_hartree_potential_cube``) over the two dimensions perpendicular to
    each grid axis in turn.
    """
    data = cube["data"]
    return {axis: data.mean(axis=tuple(i for i in range(3) if i != axis)) for axis in range(3)}

    data = _read_hartree_potential_cube(cube_path)
    return {axis: data.mean(axis=tuple(i for i in range(3) if i != axis)) for axis in range(3)}
