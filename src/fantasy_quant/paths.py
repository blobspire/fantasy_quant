"""Where this project's data lives, found from anywhere.

Every data root here is written relative to the working directory --
`data/snapshots/espn`, `data/reference`, `data/manual/etr` -- and the CLI is always
run from the repo root, so it never mattered. It matters as soon as something else
runs the code: the API server behind the dashboard, a cron entry, a notebook.

What makes it worth a module rather than a shrug is that **the failures are silent**.
`etr.load_all` on a missing directory returns `{}`; `default_id_index` on a missing
crosswalk returns an index that resolves nothing and says so only in a log line. Put
together, a process with the wrong working directory prices every surface on ESPN
alone while the dashboard renders it as though a second opinion were applied. That
happened: `fq trades` from the repo root showed the tilted board and the dashboard
showed the untilted one, with nothing on screen to tell them apart.

So: prefer the working directory, fall back to the repo root, and let the caller
override. Absence stays the caller's business to report -- this only decides where
to look.
"""

from __future__ import annotations

from pathlib import Path

#: The installed package's repo root: src/fantasy_quant/paths.py -> ../../..
REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir(subpath: str | Path) -> Path:
    """`subpath` under the working directory if it exists there, else under the repo.

    Returns the working-directory candidate when neither exists, so an error message
    names the place the caller most likely meant.
    """
    here = Path.cwd() / subpath
    if here.exists():
        return here
    rooted = REPO_ROOT / subpath
    return rooted if rooted.exists() else here
