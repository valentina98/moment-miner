import fnmatch
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_shipped_file_is_in_the_package_data():
    """An editable install reads the source tree, so a file missing from
    package-data works in the image and is absent from a built wheel."""
    globs = tomllib.loads((ROOT / "pyproject.toml").read_text())[
        "tool"]["setuptools"]["package-data"]["moment_miner"]
    package = ROOT / "moment_miner"
    shipped = [p.relative_to(package).as_posix() for p in (package / "templates").iterdir()]
    assert shipped
    assert [f for f in shipped if not any(fnmatch.fnmatch(f, g) for g in globs)] == []
