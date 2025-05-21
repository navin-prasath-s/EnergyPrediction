import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

def compute_forecast_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    smape = 100 * np.mean( 2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


    return {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAPE (%)': float(mape),
        'SMAPE (%)': float(smape)
    }


def compute_horizon_degradation_metrics(actual_df, forecast_df):
    horizon_metrics = {
        'horizon': [],
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'MAPE (%)': [],
        'SMAPE (%)': []
    }

    max_h = forecast_df['horizon'].max()

    for h in range(max_h + 1):
        forecast_horizon = forecast_df[forecast_df['horizon'] == h]

        y_pred = forecast_horizon['forecast_load'].values
        # y_true = actual_df['load'].values
        y_true = actual_df['load'].values[:len(y_pred)]

        # if len(y_true) != len(y_pred) or len(y_true) == 0:
        #     continue


        metrics = compute_forecast_metrics(y_true, y_pred)
        horizon_metrics['horizon'].append(h)
        horizon_metrics['MAE'].append(metrics['MAE'])
        horizon_metrics['MSE'].append(metrics['MSE'])
        horizon_metrics['RMSE'].append(metrics['RMSE'])
        horizon_metrics['MAPE (%)'].append(metrics['MAPE (%)'])
        horizon_metrics['SMAPE (%)'].append(metrics['SMAPE (%)'])


    return pd.DataFrame(horizon_metrics)


def plot_forecast_degradation(df_metrics):
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 15), sharex=True)

    metrics_to_plot = ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)']

    for i, metric in enumerate(metrics_to_plot):
        axes[i].plot(df_metrics['horizon'], df_metrics[metric])
        axes[i].set_ylabel(metric)
        axes[i].set_title(f"{metric} vs Forecast Horizon")

    axes[-1].set_xlabel("Forecast Horizon (5-min steps)")
    plt.tight_layout()
    plt.show()


def plot_selected_horizons(actual_df, forecast_df, selected_horizons=None):
    if selected_horizons is None:
        selected_horizons = [0, 5, 10, 15, 20, 23]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 15), sharex=True)

    for i, h in enumerate(selected_horizons):
        forecast_horizon = forecast_df[forecast_df['horizon'] == h]

        y_pred = forecast_horizon['forecast_load'].values
        y_true = actual_df['load'].values[:len(y_pred)]
        ax = axes[i // 2, i % 2]
        ax.plot(y_pred, label='Forecasted Demand')
        ax.plot(y_true, label='Actual Demand', alpha=0.8)
        ax.set_title(f"Forecast Horizon: {h}")
        ax.set_xlabel("Time (5 min intervals)")
        ax.set_ylabel("Demand (MW)")
        ax.legend()
        ax.grid(True)

    for j in range(len(selected_horizons), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_forecast_trajectory(actual_df, forecast_df, target_time):
    target_time = pd.Timestamp(target_time)
    forecast_for_t = forecast_df[forecast_df['target_time'] == target_time]
    forecast_for_t = forecast_for_t.sort_values(by='issued_at')

    if forecast_for_t.empty:
        print(f"No forecasts found for target time {target_time}")
        return

    actual_match = actual_df.loc[actual_df['timestamp'] == target_time, 'load']
    if actual_match.empty:
        print(f"No actual value found for target time {target_time}")
        return

    actual_value = actual_match.values[0]

    plt.figure(figsize=(10, 5))
    plt.plot(forecast_for_t['issued_at'], forecast_for_t['forecast_load'], label='Forecasts over time')
    plt.axhline(y=actual_value, color='red', linestyle='--', label='Actual Load')
    plt.xlabel("Forecast Made At")
    plt.ylabel("Forecasted Load for " + target_time.strftime("%Y-%m-%d %H:%M"))
    plt.title("Forecast Evolution Toward Target Time")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()