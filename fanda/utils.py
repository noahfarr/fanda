import pandas as pd


def smooth_series(series: pd.Series, alpha: float = 0.1) -> pd.DataFrame | pd.Series:
    return series.ewm(alpha=alpha).mean()
