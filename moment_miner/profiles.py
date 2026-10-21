"""Profiles: the defaults of each command's options for one kind of footage.

A profile is a YAML file with one section per command, its keys that
command's long option names as `--help` shows them, without the dashes:

    caption:
      caption-prompt: parkour
    annotate:
      template: parkour

`--profile NAME` on any command applies that command's section as the
defaults of its options: an option typed on the command line still wins, and
without `--profile` the built-in defaults apply. NAME is looked up in
`<data-dir>/profiles/NAME.yaml`, then among the shipped profiles; a path to
a `.yaml` file works too. Keys never go into a profile.
"""

from importlib.resources import files as pkg_files
from pathlib import Path

import yaml

from .hostpaths import in_container


def profile_names() -> list[str]:
    tdir = pkg_files("moment_miner") / "templates"
    return sorted(f.name[len("profile_"):-len(".yaml")]
                  for f in tdir.iterdir() if f.name.startswith("profile_"))


def find_profile(name_or_path: str, data_dir: str | Path):
    """The file a profile name or path stands for, the user's own first."""
    if Path(name_or_path).suffix in (".yaml", ".yml"):
        path = Path(in_container(name_or_path))
        if path.is_file():
            return path
        raise ValueError(f"no profile file at {name_or_path}")
    own = Path(data_dir) / "profiles" / f"{name_or_path}.yaml"
    if own.is_file():
        return own
    shipped = pkg_files("moment_miner") / "templates" / f"profile_{name_or_path}.yaml"
    if shipped.is_file():
        return shipped
    raise ValueError(f"unknown profile {name_or_path!r}: not in {own.parent}, and the shipped "
                     f"ones are {', '.join(profile_names())}; or give a .yaml path")


def load_profile(name_or_path: str, data_dir: str | Path) -> dict[str, dict]:
    """Section per command -> {option name: value}, as written in the file."""
    source = find_profile(name_or_path, data_dir)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"profile {name_or_path!r} is not valid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise ValueError(f"profile {name_or_path!r}: expected one section per command, "
                         "each mapping option names to values")
    return data
