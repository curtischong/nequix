from nequix.config.models import RunConfig
from nequix.config.runs import nequix, pft


_RUNS: list[RunConfig] = [*nequix.RUNS, *pft.RUNS]

RUNS: dict[str, RunConfig] = {config.name: config for config in _RUNS}
assert len(RUNS) == len(_RUNS), "duplicate training config names"
