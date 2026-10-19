"""The one gate every file the tool writes passes through.

`writable()` allows the data folder and the output folders the user named,
footage folders included, and never lets an existing video be replaced. The
CLI arms it for each command; library code called directly, as the tests do,
runs unguarded.
"""
from pathlib import Path

import click

_data: Path | None = None
_outputs: list[Path] = []


class WriteRefused(click.ClickException):
    pass


def arm(data_dir) -> None:
    global _data
    _data = Path(data_dir).expanduser().resolve()
    _outputs.clear()


def disarm() -> None:
    global _data
    _data = None


def output(folder) -> None:
    """A folder the user named for output."""
    _outputs.append(Path(folder).expanduser().resolve())


def writable(path) -> Path:
    """`path`, unchanged, if the tool may write there; WriteRefused if not."""
    from .manifest import VIDEO_EXTS

    path = Path(path)
    if _data is None:
        return path
    p = path.expanduser().resolve()
    if p.suffix.lower() in VIDEO_EXTS and p.exists():
        raise WriteRefused(f"refusing to write {p}: an existing video is never replaced.")
    if not any(p.is_relative_to(r) for r in [_data, *_outputs]):
        raise WriteRefused(
            f"refusing to write {p}: it is outside the data folder {_data} "
            "and every output folder named to this command.")
    return path
