import pandas as pd
import datetime


def parse_datetime(date):
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
    df['timestamp'] = df['timestamp'].apply(parse_datetime)

    failed = df['timestamp'].isna().sum()
    if failed:
        print(f"Warning: {failed} timestamps failed to parse and will be set as NaT")

    df = df.sort_values(by=['timestamp']).reset_index(drop=True)

    # Format to unified readable format
    # df['timestamp'] = df['timestamp'].dt.strftime('%m/%d/%Y %I:%M:%S %p')

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

    df['issued_at'] = df['issued_at'].apply(parse_datetime)
    df['target_time'] = df['target_time'].apply(parse_datetime)

    issued_fail = df['issued_at'].isna().sum()
    target_fail = df['target_time'].isna().sum()
    if issued_fail or target_fail:
        print(f"Warning: {issued_fail} 'issued_at' and {target_fail} 'target_time' values failed to parse.")

    df = df.sort_values(by=['issued_at', 'target_time']).reset_index(drop=True)

    df["horizon"] = ((df['target_time'] - df['issued_at']) / pd.Timedelta(minutes=5)).astype(int)

    latest_issued_at = df['issued_at'].max()
    df = df[df['issued_at'] < latest_issued_at]

    # df['issued_at_str'] = df['issued_at'].dt.strftime('%m/%d/%Y %I:%M:%S %p')
    # df['target_time_str'] = df['target_time'].dt.strftime('%m/%d/%Y %I:%M:%S %p')
    return df