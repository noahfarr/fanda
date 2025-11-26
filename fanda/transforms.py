def exponential_moving_average(df, column, alpha=1.0, groupby="run_id"):
    df[column] = df.groupby(groupby)[column].transform(
        lambda x: x.ewm(alpha=alpha).mean()
    )
    return df


def remove_outliers(
    df, column, lower_quantile=0.01, upper_quantile=0.99, groupby="run_id"
):
    lower_quantile = df.groupby(groupby)[column].transform(
        lambda x: x.quantile(lower_quantile)
    )
    upper_quantile = df.groupby(groupby)[column].transform(
        lambda x: x.quantile(upper_quantile)
    )
    mask = df[column].between(lower_quantile, upper_quantile)
    return df[mask]


def truncate(df, column, groupby="run_id"):
    return df[df[column] <= df.groupby(groupby)[column].max().min()]


def normalize(df, column, groupby="run_id"):
    df[column] = df.groupby(groupby)[column].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    return df


def downsample(df, n, column, groupby="run_id"):
    df = df.sort_values(by=[groupby, column])
    df = df.groupby(groupby).nth(slice(None, None, n))
    return df


def align_column(df, column, groupby, transformation="mean"):
    df = df.copy()
    df[column] = df.groupby([groupby, "_step"])[column].transform(transformation)
    return df
