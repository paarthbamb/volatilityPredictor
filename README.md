# stock market volatility predictor

developed independently by Paarth Bamb

this repo contains a python script (`volatility_predictor.py`) i made for a school project. it trains simple regression and classification models that try to predict **short-term realized volatility** from historical price (and optional implied vol / options) data.

## what it does

- loads a csv with historical price data (must contain a close/adjclose column)
- computes log returns and a short-term realized volatility (default 5-day)
- builds lagged features and some rolling statistics
- creates two targets:
  - `target_rv` — regression target (next-period realized volatility)
  - `target_high_vol` — classification target (whether next-period rv is in the top 25%)
- trains three regressors: random forest, gradient boosting, bayesian ridge
- trains three classifiers: random forest, gradient boosting, logistic regression
- evaluates using rmse for regressors and log loss / auc / accuracy for classifiers
- saves trained models and scaler in a folder called `models/`

## requirements

- python 3.8+
- pandas
- numpy
- scikit-learn
- joblib

install with pip:

```bash
pip install pandas numpy scikit-learn joblib
````

## how to run

general usage (from repo root):

```bash
python volatility_predictor.py --data path/to/your/data.csv --out models --rv-window 5
```

* `--data` defaults to `data.csv`.
* `--out` defaults to `models`.
* `--rv-window` controls the window used to compute short-term realized volatility (default 5 days).

## expected input csv

the script will look for a close-like column (`close`, `adjclose`, `adj_close`, or `price`) automatically. if you have a date column (`date` or similar) it will use it as the index. optional columns that help model performance:

* `volume` (or similar)
* `iv` or `implied_vol` (implied volatility from options)

if your csv uses different column names just rename them (or add a `close` column pointing to your adjusted close price).

## notes / limitations

* this is a student project intended to be a solid baseline, not a production-grade system.
* data leakage: i tried to shift/lag features so the models don't see future data, but please double-check when you use other features.
* performance depends a lot on data quality and timeframe. try tuning `rv-window`, and the model hyperparameters.

## license

feel free to use and modify. if you're going to copy this into a public repo, maybe add an appropriate license.

```
```
