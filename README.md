# Supply Chain & Retail Demand Forecaster


An end-to-end statistical time-series forecasting pipeline built to optimize retail inventory tracking. This project models consumer habits and seasonal patterns to predict future retail sales volumes, addressing the costly business problem of overstocking and understocking.

##  Project Overview

Knowing exactly how much inventory to stock is a critical supply chain challenge. This project leverages historical daily sales data to forecast a 90-day future demand window. By isolating behavioral variance and applying autoregressive statistical modeling, the pipeline provides actionable, data-driven inventory projections.

### Key Features
* **Statistical Time-Series Forecasting:** Accomplished accurate demand forecasting using the Statsmodels and `pmdarima` libraries (ARIMA) to predict future sales volumes.
* **Behavioral Feature Engineering:** Isolated market variance by engineering specific temporal features to capture multi-cycle seasonality, major US holiday trends, and day-of-the-week fluctuations.
* **Automated Parameter Tuning:** Utilized Augmented Dickey-Fuller (ADF) testing for stationarity checks and algorithmic stepwise searching to determine optimal $(p, d, q)$ and seasonal parameters.
* **Dynamic Data Visualization:** Constructed predictive data visualizations using Matplotlib to validate future demand projections and 95% confidence intervals against historical smoothed growth trends.

##  Dataset
This project uses the **Store Item Demand Forecasting Challenge** dataset from Kaggle. It contains 5 years of daily sales data across multiple stores and items. 

The pipeline automatically aggregates this granular data into a univariate time series representing total daily corporate sales volume.

##  Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/shaivatva-shukla/retail-demand-forecasting.git](https://github.com/shaivatva-shukla/retail-demand-forecasting.git)
cd retail-demand-forecasting
