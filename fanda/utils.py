import pandas as pd


def smooth_series(series: pd.Series, alpha: float = 0.1) -> pd.DataFrame | pd.Series:
    return series.ewm(alpha=alpha).mean()


def normalize_series(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df
