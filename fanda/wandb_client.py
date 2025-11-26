from typing import Optional
import wandb
import pandas as pd
from tqdm.rich import tqdm
from joblib import Memory

memory = Memory("./.fanda.cache", verbose=0)

# @memory.cache
def fetch_wandb(
    entity: str,
    project: str,
    keys: Optional[list[str]] = None,
    filters: Optional[dict] = None,
    samples: int = 500,
) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters)

    histories = pd.DataFrame()
    for run in tqdm(runs):
        df = run.history(
            samples=samples,
            keys=keys,
            pandas=True,
        )
        if df.empty:
            continue

        df["run_id"] = run.id

        histories = pd.concat([histories, df], ignore_index=True)

    if histories.empty:
        raise ValueError("No histories found")

    configs = pd.json_normalize(
        [
            {
                "run_id": run.id,
                "run_name": run.name,
                "group": run.group,
                **run.config,
            }
            for run in runs
        ]
    )

    df = pd.merge(histories, configs, on="run_id", how="left")
    return df

