import os
os.environ["KAGGLE_API_TOKEN"] = "KGAT_ff4a9092a34bc04d7606af0f65d222c4"
import zipfile
import pandas as pd
import holidays
from kaggle.api.kaggle_api_extended import KaggleApi

def create_requirements():
    """Generates the requirements.txt file with necessary data science packages."""
    packages = [
        "pandas",
        "numpy",
        "statsmodels",
        "matplotlib",
        "scikit-learn",
        "kaggle"
    ]
    with open("requirements.txt", "w") as f:
        f.write("\n".join(packages) + "\n")
    print("requirements.txt has been generated successfully.")

def download_dataset():
    """Downloads and extracts the Store Item Demand Forecasting Challenge dataset."""
    
    competition = "demand-forecasting-kernels-only"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Authenticating with Kaggle API and downloading '{competition}' dataset...")
    try:
        api = KaggleApi()
        api.authenticate() 
        api.competition_download_files(competition, path=data_dir)
        
        zip_file = os.path.join(data_dir, f"{competition}.zip")
        if os.path.exists(zip_file):
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_file)
            print("Extraction complete.")
        else:
            print("Download finished, but zip file was not found. Data might be unzipped already.")
    except OSError as e:
        print(f"Error during authentication: {e}")
        print("Please ensure your Kaggle API token is located at ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\kaggle.json on Windows)")
        raise

def load_time_series_data(csv_path):
    """
    Reads the CSV, parses the date column correctly, 
    and sets the date as the index for time-series forecasting.
    """
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    
    df = pd.read_csv(csv_path, parse_dates=['date'])

    df['date'] = pd.to_datetime(df['date'])

    df['day_of_week'] = df['date'].dt.dayofweek

    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week

    years = df['date'].dt.year.unique().tolist()
    us_holidays = holidays.US(years=years)

    df['is_holiday'] = df['date'].dt.date.isin(us_holidays)

    print(df.head())

    import numpy as np
    from statsmodels.tsa.stattools import adfuller
    import pmdarima as pm
    import matplotlib.pyplot as plt

    daily_sales = (
        df.groupby('date')['sales']
        .sum()
        .reset_index()
        .set_index('date')
        .sort_index()
    )

    daily_sales = daily_sales.asfreq('D')

    daily_sales['sales'] = daily_sales['sales'].interpolate(method='linear')

    print(f"Aggregated series shape: {daily_sales.shape}")
    print(daily_sales.head())

    adf_result = adfuller(daily_sales['sales'].dropna())

    print("\n--- ADF Test on Raw Series ---")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value:.4f}")

    if adf_result[1] <= 0.05:
        print("=> p-value <= 0.05: Reject H0. Series is likely STATIONARY.")
    else:
        print("=> p-value > 0.05: Fail to reject H0. Series is likely NON-STATIONARY "
            "(differencing will probably be required).")

    model = pm.auto_arima(
        daily_sales['sales'],
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,              
        seasonal=True,
        m=7,                 
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        D=None,              
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print("\n--- Best Model Summary ---")
    print(model.summary())
    print(f"\nSelected order (p,d,q): {model.order}")
    print(f"Selected seasonal order (P,D,Q,m): {model.seasonal_order}")

    n_periods = 90
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

    last_date = daily_sales.index[-1]
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_periods,
        freq='D'
    )

    forecast_df = pd.DataFrame({
        'forecast': forecast,
        'lower_ci': conf_int[:, 0],
        'upper_ci': conf_int[:, 1]
    }, index=forecast_index)

    print("\n--- 90-Day Forecast (first 5 rows) ---")
    print(forecast_df.head())

    daily_sales['rolling_7d'] = daily_sales['sales'].rolling(window=7).mean()

    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    ax.plot(daily_sales.index, daily_sales['sales'], color='#ced4da', label='Raw Daily Sales', linewidth=1, alpha=0.7)
    ax.plot(daily_sales.index, daily_sales['rolling_7d'], color='#0d6efd', label='7-Day Smoothed Trend', linewidth=2)

    ax.plot(forecast_df.index, forecast_df['forecast'], color='#fd7e14', label='ARIMA 90-Day Forecast', linewidth=2, linestyle='--')
    ax.fill_between(forecast_df.index, forecast_df['lower_ci'], forecast_df['upper_ci'], color='#fd7e14', alpha=0.2, label='95% Confidence Interval')

    ax.set_title('Retail Demand Forecasting: Historical Trends vs. 90-Day Projection', fontsize=16, fontweight='bold', pad=20, color='#212529')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold', color='#495057')
    ax.set_ylabel('Total Sales Volume', fontsize=12, fontweight='bold', color='#495057')

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#adb5bd')
    ax.spines['bottom'].set_color('#adb5bd')

    ax.legend(loc='upper left', fontsize=11, frameon=True, edgecolor='#adb5bd', facecolor='white')

    plt.tight_layout()
    plt.show()
    
    df.set_index('date', inplace=True)
    
    df.sort_index(inplace=True)
    
    return df

if __name__ == "__main__":
    create_requirements()
    
    try:
        download_dataset()
    except Exception as e:
        print(f"Skipping data load due to download error: {e}")
        exit(1)
        
    train_file = os.path.join("data", "train.csv")
    try:
        df_train = load_time_series_data(train_file)
        print("\nData loaded successfully. Here is a preview of the training data:")
        print(df_train.head())
        print(f"\nDataframe Info:")
        print(df_train.info())
    except FileNotFoundError as e:
        print(e)