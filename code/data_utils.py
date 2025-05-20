import pandas as pd


def clean_actual_data(df, area='PJM RTO'):
    """
    Filters and standardizes actual load data for a given area.

    Parameters:
        df (pd.DataFrame): Raw actual data.
        area (str): Zone/region to filter. Default is 'PJM RTO'.

    Returns:
        pd.DataFrame: Cleaned actual data with standardized column names.
    """
    df = df[df['area'] == area]
    df = df[['datetime_beginning_utc', 'instantaneous_load']]
    df = df.rename(columns={
        'datetime_beginning_utc': 'timestamp',
        'instantaneous_load': 'load'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def clean_forecast_data(df, forecast_area='RTO_COMBINED'):
    """
    Filters and standardizes forecast load data for a given area.

    Parameters:
        df (pd.DataFrame): Raw forecast data.
        forecast_area (str): Forecast zone/region to filter. Default is 'RTO_COMBINED'.

    Returns:
        pd.DataFrame: Cleaned forecast data with standardized column names.
    """
    df = df[df['forecast_area'] == forecast_area]
    df = df[['evaluated_at_utc', 'forecast_datetime_beginning_utc', 'forecast_load_mw']]
    df = df.rename(columns={
        'forecast_datetime_beginning_utc': 'target_time',
        'evaluated_at_utc': 'issued_at',
        'forecast_load_mw': 'forecast_load'
    })
    df['issued_at'] = pd.to_datetime(df['issued_at'])
    df['target_time'] = pd.to_datetime(df['target_time'])
    df["horizon"] = ((df['target_time'] - df['issued_at']) / pd.Timedelta(minutes=5)).astype(int)
    return df