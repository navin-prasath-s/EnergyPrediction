import pandas as pd
from datetime import timedelta

class SlidingDataBuffer:
    def __init__(self, initial_data: pd.DataFrame = None, max_size: int = 64, timestamp_col: str = "timestamp"):
        """
        Initializes the sliding data buffer.

        Args:
            initial_data (pd.DataFrame): Initial buffer content. Must have exactly `max_size` rows.
            max_size (int): Number of rows to retain in buffer.
            timestamp_col (str): Column name for timestamps.

        Raises:
            ValueError: If initial_data has wrong length.
        """
        self.max_size = max_size
        self.timestamp_col = timestamp_col

        if initial_data is not None:
            if not isinstance(initial_data, pd.DataFrame):
                raise TypeError("initial_data must be a pandas DataFrame.")
            if len(initial_data) != max_size:
                raise ValueError(f"initial_data must have exactly {max_size} rows, got {len(initial_data)}.")
            self.buffer = initial_data.reset_index(drop=True)
        else:
            self.buffer = pd.DataFrame()

    def add_data(self, new_row_df: pd.DataFrame):
        """
        Adds a single new row if timestamp is exactly 5 minutes ahead (ignoring seconds).

        Args:
            new_row_df (pd.DataFrame): One-row DataFrame with a datetime-like timestamp.

        Raises:
            ValueError: If row count ≠ 1 or timestamp delta is invalid.
        """
        if not isinstance(new_row_df, pd.DataFrame):
            raise TypeError("new_row_df must be a pandas DataFrame.")
        if len(new_row_df) != 1:
            raise ValueError(f"add_data expects exactly 1 row, got {len(new_row_df)}.")
        if self.timestamp_col not in new_row_df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in new row.")

        if not self.buffer.empty:
            last_ts = pd.to_datetime(self.buffer.iloc[-1][self.timestamp_col])
            new_ts = pd.to_datetime(new_row_df.iloc[0][self.timestamp_col])

            # Strip seconds/microseconds
            last_ts = last_ts.replace(second=0, microsecond=0)
            new_ts = new_ts.replace(second=0, microsecond=0)

            if (new_ts - last_ts) != timedelta(minutes=5):
                raise ValueError(f"New row timestamp {new_ts} is not 5 minutes after last timestamp {last_ts}.")

        self.buffer = pd.concat([self.buffer, new_row_df], ignore_index=True)

        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer.iloc[-self.max_size:].reset_index(drop=True)

    def get_data(self) -> pd.DataFrame:
        """Returns a copy of the current buffer."""
        return self.buffer.copy()