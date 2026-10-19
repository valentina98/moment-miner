import os
from pathlib import Path


def on_host(path: str | Path) -> str:
    """`path` as the host sees it, from MM_HOST_PATHS (`/videos=/host/dir;...`).

    LosslessCut runs on the host, so a project written in a container must
    name its video relative to where both files sit there. And a folder
    mounted twice, as footage and as output, is one place only by host path.
    """
    path = os.path.abspath(path)
    for pair in filter(None, os.environ.get("MM_HOST_PATHS", "").split(";")):
        inside, _, outside = pair.partition("=")
        if Path(path).is_relative_to(inside):
            return os.path.join(outside, os.path.relpath(path, inside))
    return path
