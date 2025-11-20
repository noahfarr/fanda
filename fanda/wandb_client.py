import wandb
import pandas as pd
from typing import Optional


def fetch_history(
    api: wandb.Api,
    entity: str,
    project: str,
    keys: Optional[list[str]] = None,
    filters: Optional[dict] = None,
    samples: int = 500,
) -> pd.DataFrame:
    runs = api.runs(f"{entity}/{project}", filters=filters)

    histories = runs.histories(samples=samples, keys=keys, format="pandas")
    histories = histories.reset_index()

    configs = pd.json_normalize(
        [
            {"run_id": run.id, "run_name": run.name, "group": run.group, **run.config}
            for run in runs
        ]
    )

    df = pd.merge(histories, configs, on="run_id", how="left")

    return df
