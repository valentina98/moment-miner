import os
from pathlib import Path

import click


def _pairs() -> list[tuple[str, str]]:
    return [(inside, outside) for inside, _, outside in
            (pair.partition("=") for pair in
             filter(None, os.environ.get("MM_HOST_PATHS", "").split(";")))]


def on_host(path: str | Path) -> str:
    """`path` as the host sees it, from MM_HOST_PATHS (`/videos=/host/dir;...`).

    LosslessCut runs on the host, so a project written in a container must
    name its video relative to where both files sit there. And a folder
    mounted twice, as footage and as output, is one place only by host path.
    """
    path = os.path.abspath(path)
    for inside, outside in _pairs():
        if Path(path).is_relative_to(inside):
            return os.path.join(outside, os.path.relpath(path, inside))
    return path


def in_container(path: str | Path) -> str:
    """The reverse of `on_host`: a host path given to a command in a
    container, as the container sees it. Anything else is returned as is.

    The deepest mount wins, so an output folder inside the footage maps to
    its own writable mount, not into the read-only footage."""
    path = os.path.normpath(os.path.expanduser(str(path)))
    matches = [(inside, outside) for inside, outside in _pairs()
               if outside and Path(path).is_relative_to(outside)]
    if not matches:
        return path
    inside, outside = max(matches, key=lambda m: len(m[1]))
    return os.path.join(inside, os.path.relpath(path, outside))


class UserPath(click.Path):
    """A path option or argument that also takes the user's own host path in
    Docker, so a command takes the same paths it prints."""

    def convert(self, value, param, ctx):
        return super().convert(in_container(value), param, ctx)
