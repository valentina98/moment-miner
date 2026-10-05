"""The one gate every file the tool writes passes through.

Footage is only ever read. `writable()` refuses a path inside a source folder
(one named to the command, one `mm index` has scanned, or one holding an
indexed video) and a path outside the data folder and the output folders the
user named. The CLI arms it for each command; library code called directly,
as the tests do, runs unguarded.
"""
from pathlib import Path

import click

_data: Path | None = None
_outputs: list[Path] = []
_sources: list[Path] = []
_indexed: list[Path] | None = None


class WriteRefused(click.ClickException):
    pass


def arm(data_dir) -> None:
    global _data, _indexed
    _data = Path(data_dir).expanduser().resolve()
    _outputs.clear()
    _sources.clear()
    _indexed = None


def disarm() -> None:
    global _data
    _data = None


def source(*paths) -> None:
    """Folders the command reads from; a file stands for its folder."""
    for p in paths:
        p = Path(p).expanduser().resolve()
        _sources.append(p if p.is_dir() else p.parent)


def output(folder) -> None:
    """A folder the user named for output."""
    _outputs.append(Path(folder).expanduser().resolve())


def writable(path) -> Path:
    """`path`, unchanged, if the tool may write there; WriteRefused if not."""
    path = Path(path)
    if _data is None:
        return path
    p = path.expanduser().resolve()
    for s in [*_sources, *_indexed_folders()]:
        if p.is_relative_to(s):
            raise WriteRefused(
                f"refusing to write {p}: it is inside {s}, a source folder, "
                "which is only ever read. Choose another output or --data-dir.")
    if not any(p.is_relative_to(r) for r in [_data, *_outputs]):
        raise WriteRefused(
            f"refusing to write {p}: it is outside the data folder {_data} "
            "and every output folder named to this command.")
    return path


def _indexed_folders() -> list[Path]:
    global _indexed
    if _indexed is None:
        from .manifest import source_folders

        _indexed = source_folders(_data / "manifest.db")
    return _indexed
