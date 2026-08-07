"""
Utilities for resolving the FHI-aims binary and species-defaults directory,
and for reshaping ``pyfhiaims``-parsed quantities to ``doped``'s
calculator-agnostic conventions.
"""

import shutil
import warnings
from pathlib import Path

import numpy as np
from pymatgen.core import SETTINGS
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.typing import PathLike

_SPECIES_DEFAULTS_SHORTHANDS = ("light", "tight", "really_tight")
AIMS_PATH: str | None = None
AIMS_PATH_SEARCHED = False


def _infer_species_dir_from_exe_path(aims_exe: Path) -> str | None:
    candidate_dirs = [
        aims_exe.parent.parent / "species_defaults",
        aims_exe.parent.parent / "share" / "aims" / "species_defaults",
        aims_exe.parent.parent / "aims" / "species_defaults",
        aims_exe.parent / "species_defaults",
    ]
    for candidate in candidate_dirs:
        if (candidate / "defaults_2020" / "light").is_dir():
            return str(candidate)
    return None


def _discover_aims_species_dir() -> str | None:
    r"""
    Lazily infer the AIMS species-defaults directory if ``AIMS_SPECIES_DIR`` is not
    set. This allows fallback discovery from common AIMS binary locations on the host.
    """
    global AIMS_PATH, AIMS_PATH_SEARCHED
    if AIMS_PATH is not None:
        return AIMS_PATH
    if AIMS_PATH_SEARCHED:
        return None
    AIMS_PATH_SEARCHED = True

    aims_species_dir = SETTINGS.get("AIMS_SPECIES_DIR")
    if aims_species_dir:
        AIMS_PATH = aims_species_dir
        return AIMS_PATH

    search_paths = []
    binary_path = shutil.which("aims")
    if binary_path:
        search_paths.append(Path(binary_path))
    for path in (
        "/opt/aims/bin/aims",
        "/opt/fhi-aims/bin/aims",
        "/usr/local/bin/aims",
        "/usr/bin/aims",
        "/snap/bin/aims",
    ):
        exe = Path(path)
        if exe.exists():
            search_paths.append(exe)

    for exe in search_paths:
        species_dir = _infer_species_dir_from_exe_path(exe)
        if species_dir:
            warnings.warn(
                f"Found FHI-aims executable at {exe}. "
                "Set AIMS_SPECIES_DIR in pymatgen settings to the directory containing "
                "defaults_2020 for reliable species-default resolution.",
                UserWarning,
                stacklevel=3,
            )
            AIMS_PATH = species_dir
            return AIMS_PATH

    warnings.warn(
        "AIMS_SPECIES_DIR is not configured in pymatgen settings and a valid "
        "FHI-aims species defaults directory could not be inferred from a common "
        "binary location. Please set AIMS_SPECIES_DIR to the directory containing "
        "defaults_2020.",
        UserWarning,
        stacklevel=3,
    )
    return None


def _resolve_species_defaults(species_defaults: PathLike | str) -> str:
    """Return an existing directory containing FHI-aims species-default files.

    ``"light"``, ``"tight"``, and ``"really_tight"`` are reserved
    shorthands for ``<AIMS_SPECIES_DIR>/defaults_2020/<shorthand>``, where
    ``AIMS_SPECIES_DIR`` is pymatgen's configured FHI-aims species-defaults
    directory.

    If the input is a string that is not one of the three shorthand values,
    it is first interpreted as a literal path. If that literal path is not
    absolute and ``AIMS_SPECIES_DIR`` is configured, the path is also
    interpreted relative to ``AIMS_SPECIES_DIR``.

    If ``AIMS_SPECIES_DIR`` is not configured, this module will attempt to
    infer it lazily from a known AIMS executable location. It will warn if
    the executable is found and a warning if it cannot infer a valid path.
    """
    if isinstance(species_defaults, str) and species_defaults in _SPECIES_DEFAULTS_SHORTHANDS:
        aims_species_dir = SETTINGS.get("AIMS_SPECIES_DIR") or _discover_aims_species_dir()
        if not aims_species_dir:
            raise ValueError(
                "AIMS_SPECIES_DIR must be configured in pymatgen settings to use the "
                f"{species_defaults!r} species_defaults shorthand. Alternatively, pass the full "
                "path to the directory containing the element default files."
            )
        species_defaults_path = Path(aims_species_dir).expanduser() / "defaults_2020" / species_defaults
    else:
        species_defaults_path = Path(species_defaults).expanduser()
        if not species_defaults_path.is_absolute():
            aims_species_dir = SETTINGS.get("AIMS_SPECIES_DIR") or _discover_aims_species_dir()
            if aims_species_dir:
                relative_path = Path(aims_species_dir).expanduser() / species_defaults
                if relative_path.is_dir():
                    species_defaults_path = relative_path

    if not species_defaults_path.is_dir():
        raise FileNotFoundError(
            "species_defaults must be an existing directory containing FHI-aims element default "
            f"files, got: {species_defaults_path}"
        )
    return str(species_defaults_path.resolve())


def reshape_eigenvalues_and_occupations(
    eigenvalues: list[float] | dict[str, list[float]],
    occupations: list[float] | dict[str, list[float]],
    n_spins: int,
    n_kpoints: int,
) -> dict[Spin, np.ndarray]:
    """
    Reshape ``pyfhiaims``-parsed eigenvalues/occupations to the ``pymatgen``-
    style ``{Spin: array}`` format used by
    :attr:`~doped.io.outputs.CalculationOutputs.eigenvalues` (array shape
    ``(nkpoints, nbands, 2)``, with the last axis being (energy in eV,
    occupation), matching ``Vasprun.eigenvalues``).

    ``AimsImage.get_results(verbosity="all")["eigenvalues"]``/
    ``["occupations"]`` are flat, per-(k-point, spin) block: a flat list of
    ``nbands`` values if there is only one (k-point, spin) block (i.e.
    ``n_spins == n_kpoints == 1``), else a ``dict`` of one such flat list per
    block, keyed by an (opaque) string identifying the block, in the same
    order as printed in ``aims.out`` -- i.e. outer loop over k-points, inner
    loop over spin channels (FHI-aims prints all spin channels together for
    a given k-point before moving to the next k-point).

    Args:
        eigenvalues (list[float] | dict[str, list[float]]):
            ``AimsImage.get_results(verbosity="all")["eigenvalues"]``.
        occupations (list[float] | dict[str, list[float]]):
            ``AimsImage.get_results(verbosity="all")["occupations"]``.
        n_spins (int):
            Number of spin channels (1 or 2), from
            ``AimsStdout.header_summary["n_spins"]``.
        n_kpoints (int):
            Number of k-points, from
            ``AimsStdout.header_summary["n_k_points"]``.

    Returns:
        dict[Spin, np.ndarray]: Eigenvalues/occupations in ``pymatgen``-style
        ``{Spin: array}`` format.
    """
    eigenvalue_blocks = list(eigenvalues.values()) if isinstance(eigenvalues, dict) else [eigenvalues]
    occupation_blocks = list(occupations.values()) if isinstance(occupations, dict) else [occupations]
    n_bands = len(eigenvalue_blocks[0])

    # blocks are ordered k-point-major, spin-minor (see docstring), matching this row-major reshape:
    eigenvalue_array = np.array(eigenvalue_blocks).reshape(n_kpoints, n_spins, n_bands)
    occupation_array = np.array(occupation_blocks).reshape(n_kpoints, n_spins, n_bands)
    stacked = np.stack([eigenvalue_array, occupation_array], axis=-1)  # (nkpoints, nspins, nbands, 2)

    spins = [Spin.up, Spin.down][:n_spins]
    return {spin: stacked[:, spin_index, :, :] for spin_index, spin in enumerate(spins)}
