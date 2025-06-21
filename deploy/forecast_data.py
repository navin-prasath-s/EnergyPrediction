from dataclasses import dataclass
import pandas as pd
from typing import Optional, List
from collections import defaultdict

@dataclass
class ForecastRecord:
    issued_at: pd.Timestamp
    target_time: pd.Timestamp
    horizon: int
    your_pred: Optional[float] = None
    agency_pred: Optional[float] = None
    actual: Optional[float] = None

    def __post_init__(self):
        self.issued_at = self.issued_at.floor("min")
        self.target_time = self.target_time.floor("min")

def floor_to_min(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("min")


def create_my_records(issued_at: pd.Timestamp, your_preds: List[float]) -> List[ForecastRecord]:
    issued_at = floor_to_min(issued_at)
    return [
        ForecastRecord(
            issued_at=issued_at,
            target_time=issued_at + pd.Timedelta(minutes=5 * h),
            horizon=h,
            your_pred=float(your_preds[h])
        ) for h in range(len(your_preds))
    ]


def agency_df_to_records(df_agency: pd.DataFrame) -> List[ForecastRecord]:
    df_agency['issued_at'] = pd.to_datetime(df_agency['issued_at']).dt.floor("min")
    df_agency['target_time'] = pd.to_datetime(df_agency['target_time']).dt.floor("min")

    return [
        ForecastRecord(
            issued_at=row['issued_at'],
            target_time=row['target_time'],
            horizon=int(row['horizon']),
            agency_pred=row['forecast_load']
        )
        for _, row in df_agency.iterrows()
    ]

def merge_records(your_list: List[ForecastRecord], agency_list: List[ForecastRecord]) -> List[ForecastRecord]:
    merged = {}

    for r in your_list:
        key = (r.issued_at, r.horizon)
        merged[key] = r

    for r in agency_list:
        key = (r.issued_at, r.horizon)
        if key in merged:
            merged[key].agency_pred = r.agency_pred
        else:
            merged[key] = r  # agency-only forecast

    return list(merged.values())