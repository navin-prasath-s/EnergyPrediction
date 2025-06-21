import pickle
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox


# def clean_actual_data(df, area='PJM RTO'):
#     """
#     Filters and standardizes actual load data for a given area.
#
#     Parameters:
#         df (pd.DataFrame): Raw actual data.
#         area (str): Zone/region to filter. Default is 'PJM RTO'.
#
#     Returns:
#         pd.DataFrame: Cleaned actual data with standardized column names.
#     """
#     df = df[df['area'] == area]
#     df = df[['datetime_beginning_utc', 'instantaneous_load']]
#     df = df.rename(columns={
#         'datetime_beginning_utc': 'timestamp',
#         'instantaneous_load': 'load'
#     })
#     df['timestamp'] = df['timestamp'].apply(_parse_datetime)
#
#     failed = df['timestamp'].isna().sum()
#     if failed:
#         print(f"Warning: {failed} timestamps failed to parse and will be set as NaT")
#
#     df = df.sort_values(by=['timestamp']).reset_index(drop=True)
#     return df

def clean_actual_data(df, area='PJM RTO'):
    """
    Filters and standardizes actual load data for a given area.

    Parameters:
        df (pd.DataFrame): Raw actual data.
        area (str): Zone/region to filter. Default is 'PJM RTO'.

    Returns:
        Tuple[pd.DataFrame, pd.Timestamp]: Cleaned actual data and latest timestamp (hour+minute only)
    """
    df = df[df['area'] == area]
    df = df[['datetime_beginning_utc', 'instantaneous_load']]
    df = df.rename(columns={
        'datetime_beginning_utc': 'timestamp',
        'instantaneous_load': 'load'
    })
    df['timestamp'] = df['timestamp'].apply(_parse_datetime)

    failed = df['timestamp'].isna().sum()
    if failed:
        print(f"Warning: {failed} timestamps failed to parse and will be set as NaT")

    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Extract latest hour+minute timestamp (drop seconds and microseconds)
    latest_ts = df['timestamp'].max().replace(second=0, microsecond=0)

    return df, latest_ts


# def clean_forecast_data(df, forecast_area='RTO_COMBINED'):
#     """
#     Filters and standardizes forecast load data for a given area.
#
#     Parameters:
#         df (pd.DataFrame): Raw forecast data.
#         forecast_area (str): Forecast zone/region to filter. Default is 'RTO_COMBINED'.
#
#     Returns:
#         pd.DataFrame: Cleaned forecast data with standardized column names.
#     """
#     df = df[df['forecast_area'] == forecast_area]
#     df = df[['evaluated_at_utc', 'forecast_datetime_beginning_utc', 'forecast_load_mw']]
#     df = df.rename(columns={
#         'forecast_datetime_beginning_utc': 'target_time',
#         'evaluated_at_utc': 'issued_at',
#         'forecast_load_mw': 'forecast_load'
#     })
#
#     df['issued_at'] = df['issued_at'].apply(_parse_datetime)
#     df['target_time'] = df['target_time'].apply(_parse_datetime)
#
#     issued_fail = df['issued_at'].isna().sum()
#     target_fail = df['target_time'].isna().sum()
#     if issued_fail or target_fail:
#         print(f"Warning: {issued_fail} 'issued_at' and {target_fail} 'target_time' values failed to parse.")
#
#     df = df.sort_values(by=['issued_at', 'target_time']).reset_index(drop=True)
#
#     df["horizon"] = ((df['target_time'] - df['issued_at']) / pd.Timedelta(minutes=5)).astype(int)
#
#     # latest_issued_at = df['issued_at'].max()
#     # # df = df[df['issued_at'] < latest_issued_at]
#
#     return df

def clean_forecast_data(df, latest_ts=None, forecast_area='RTO_COMBINED'):
    """
    Filters and standardizes forecast load data for a given area.
    Optionally returns only forecasts issued at a specific timestamp.

    Parameters:
        df (pd.DataFrame): Raw forecast data.
        latest_ts (pd.Timestamp or None): If provided, filter to this 'issued_at' time.
        forecast_area (str): Forecast zone/region to filter. Default is 'RTO_COMBINED'.

    Returns:
        pd.DataFrame: Cleaned forecast data.
    """
    df = df[df['forecast_area'] == forecast_area]
    df = df[['evaluated_at_utc', 'forecast_datetime_beginning_utc', 'forecast_load_mw']]
    df = df.rename(columns={
        'forecast_datetime_beginning_utc': 'target_time',
        'evaluated_at_utc': 'issued_at',
        'forecast_load_mw': 'forecast_load'
    })

    df['issued_at'] = df['issued_at'].apply(_parse_datetime)
    df['target_time'] = df['target_time'].apply(_parse_datetime)

    issued_fail = df['issued_at'].isna().sum()
    target_fail = df['target_time'].isna().sum()
    if issued_fail or target_fail:
        print(f"Warning: {issued_fail} 'issued_at' and {target_fail} 'target_time' values failed to parse.")

    df = df.sort_values(by=['issued_at', 'target_time']).reset_index(drop=True)
    df["horizon"] = ((df['target_time'] - df['issued_at']) / pd.Timedelta(minutes=5)).astype(int)

    # Optionally filter to a specific issued_at timestamp
    if latest_ts is not None:
        df = df[df['issued_at'] == latest_ts]

    return df

def transform_data(df, path_to_scalers):
    df = _standardize_frequency(df)
    scalers = _load_scalers(path_to_scalers)
    df = _apply_transformation(df, scalers, use_boxcox=True)
    df = _add_time_features(df)
    return df

def inverse_transform_data(series, path_to_scalers):
    scalers = _load_scalers(path_to_scalers)
    return _inverse_transform(series, scalers)




def _parse_datetime(date):
    formats = ['%m/%d/%Y %I:%M:%S %p', '%m-%d-%Y %H:%M']
    # Format 1: '5/13/2025 12:00:00 AM'
    # Format 2: '05-12-2025 23:55'
    date = str(date).strip()
    for fmt in formats:
        try:
            return pd.to_datetime(date, format=fmt)
        except ValueError:
            continue
    return pd.NaT

def _load_scalers(filepath: str):
    with open(filepath, "rb") as f:
        scalers = pickle.load(f)
    return scalers

def _apply_transformation(df, scalers, use_boxcox=True):
    """
    Applies previously fitted Box-Cox and StandardScaler to 'load' column in a new DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'load' column to transform.
        scalers  (dict): Dictionary with 'lambda' and 'scaler' from transform_series.
        use_boxcox (bool): Should match the flag used during training.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    df = df.copy()
    series = df['load'].values

    if use_boxcox:
        lam = scalers['load']['lambda']
        series = boxcox(series, lmbda=lam)

    scaler = scalers['load']['scaler']
    df['load'] = scaler.transform(series.reshape(-1, 1)).flatten()

    return df


def _inverse_transform(series, scalers):
    """
    Inverts StandardScaler and Box-Cox using provided scalers dict.

    Parameters:
        series (ndarray): Shape [n_samples, horizon] — model output in transformed space
        scalers (dict): Dictionary with keys:
                        - 'load': {'scaler': StandardScaler, 'lambda': float}

    Returns:
        ndarray: Series restored to original scale
    """
    scaler = scalers['load']['scaler']
    boxcox_lambda = scalers['load']['lambda']

    # Step 1: inverse standard scaling
    reshaped = series.reshape(-1, 1)
    unscaled = scaler.inverse_transform(reshaped).reshape(series.shape)

    # Step 2: inverse box-cox (always applied)
    unscaled = inv_boxcox(unscaled, boxcox_lambda)
    return unscaled


def _standardize_frequency(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df.index = df.index.round('5min')
    df = df.asfreq('5min')
    df = df.interpolate()
    return df

def _add_time_features(df, interval_minutes=5):
    """
    Adds sin_time and cos_time features based on the time of day, assuming a fixed interval in minutes.

    Parameters:
        df (pd.DataFrame): DataFrame with a datetime index.
        interval_minutes (int): Time interval between observations in minutes (default is 5).

    Returns:
        pd.DataFrame: DataFrame with 'sin_time' and 'cos_time' columns added.
    """
    df = df.copy()

    minutes_in_day = 24 * 60
    steps_per_day = minutes_in_day // interval_minutes

    time_of_day = (df.index.hour * 60 + df.index.minute) / interval_minutes
    angle = 2 * np.pi * time_of_day / steps_per_day

    df['sin_time'] = np.sin(angle)
    df['cos_time'] = np.cos(angle)

    return df


