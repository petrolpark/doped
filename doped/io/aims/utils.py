"""
Utilities for resolving the FHI-aims binary and species-defaults directory.
"""

import shutil
import warnings
from pathlib import Path

from pymatgen.core import SETTINGS
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
