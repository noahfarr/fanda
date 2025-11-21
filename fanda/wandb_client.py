from typing import Optional
import wandb
import pandas as pd
from tqdm.rich import tqdm


def fetch_history(
    api: wandb.Api,
    entity: str,
    project: str,
    keys: Optional[list[str]] = None,
    filters: Optional[dict] = None,
    samples: int = 500,
) -> pd.DataFrame:
    runs = api.runs(f"{entity}/{project}", filters=filters)

    histories = []
    for run in tqdm(runs):
        history_data = run.history(
            samples=samples,
            keys=keys,
            pandas=False,
        )
        if not history_data:
            continue
        df = pd.DataFrame.from_records(history_data)
        histories.append(df)
    histories = pd.concat(histories)
    histories.reset_index(drop=True, inplace=True)
    histories = histories[(sorted(histories.columns))]

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
