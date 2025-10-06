# stock market volatility predictor
# developed independently by ayo

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score, accuracy_score
import joblib

def load_data(path: Path):
    df = pd.read_csv(path, parse_dates=True)
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df = df.sort_values(by=date_cols[0]).reset_index(drop=True)
        df.index = df[date_cols[0]]
    else:
        df = df.sort_index()

    candidates = [c for c in df.columns if c.lower() in ('close', 'adjclose', 'adj_close', 'price')]
    if not candidates:
        raise ValueError('could not find a close price column. please provide a csv with a close/adjclose column')
    close_col = candidates[0]
    df['close'] = df[close_col].astype(float)

    vol_cols = [c for c in df.columns if 'volume' in c.lower()]
    if vol_cols:
        df['volume'] = df[vol_cols[0]]

    iv_cols = [c for c in df.columns if 'iv' in c.lower() or 'implied' in c.lower()]
    if iv_cols:
        df['implied_vol'] = df[iv_cols[0]]

    return df

def make_features(df: pd.DataFrame, rv_window: int = 5):
    df = df.copy()
    df['logret'] = np.log(df['close']).diff()
    df['rv'] = df['logret'].rolling(rv_window).std() * np.sqrt(252)

    for lag in range(1, 6):
        df[f'logret_lag{lag}'] = df['logret'].shift(lag)
        df[f'rv_lag{lag}'] = df['rv'].shift(lag)

    df[f'logret_mean_{rv_window}'] = df['logret'].rolling(rv_window).mean().shift(1)
    df[f'logret_std_{rv_window}'] = df['logret'].rolling(rv_window).std().shift(1)

    if 'volume' in df.columns:
        df['vol_mean_5'] = df['volume'].rolling(5).mean().shift(1)
        df['vol_change_1'] = df['volume'].pct_change().shift(1).fillna(0)

    if 'implied_vol' in df.columns:
        df['implied_vol'] = df['implied_vol'].astype(float)
        df['iv_diff_1'] = df['implied_vol'].diff().shift(1)

    df['target_rv'] = df['rv'].shift(-1)

    thresh = df['rv'].quantile(0.75)
    df['target_high_vol'] = (df['target_rv'] > thresh).astype(int)

    df = df.dropna()
    return df

def prepare_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if any(prefix in c for prefix in ('logret_lag', 'rv_lag', 'logret_mean', 'logret_std', 'vol_', 'implied_vol', 'iv_'))]
    if not feature_cols:
        raise ValueError('no features found after preprocessing; check your input data')

    X = df[feature_cols].values
    y_reg = df['target_rv'].values
    y_clf = df['target_high_vol'].values
    return X, y_reg, y_clf, feature_cols

def train_and_eval(X, y_reg, y_clf, feature_names, outdir: Path):
    X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, outdir / 'scaler.joblib')

    models_reg = {
        'random_forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'bayesian_ridge': BayesianRidge()
    }

    reg_results = {}
    for name, model in models_reg.items():
        print(f'training {name} (regression)')
        model.fit(X_train_s, yreg_train)
        preds = model.predict(X_test_s)
        rmse = mean_squared_error(yreg_test, preds, squared=False)
        reg_results[name] = {'model': model, 'rmse': rmse}
        joblib.dump(model, outdir / f'{name}_reg.joblib')
        print(f'  {name} rmse: {rmse:.6f}')

    models_clf = {
        'random_forest_clf': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'gradient_boosting_clf': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'logistic': LogisticRegression(max_iter=1000)
    }

    clf_results = {}
    for name, model in models_clf.items():
        print(f'training {name} (classification)')
        model.fit(X_train_s, yclf_train)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test_s)[:, 1]
            ll = log_loss(yclf_test, probs, labels=[0, 1])
        else:
            scores = model.decision_function(X_test_s)
            probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            ll = log_loss(yclf_test, probs, labels=[0, 1])

        preds = model.predict(X_test_s)
        acc = accuracy_score(yclf_test, preds)
        auc = roc_auc_score(yclf_test, probs)

        clf_results[name] = {'model': model, 'log_loss': ll, 'accuracy': acc, 'auc': auc}
        joblib.dump(model, outdir / f'{name}_clf.joblib')
        print(f'  {name} log_loss: {ll:.6f} acc: {acc:.4f} auc: {auc:.4f}')

    report = {
        'regression': {k: {'rmse': v['rmse']} for k, v in reg_results.items()},
        'classification': {k: {'log_loss': v['log_loss'], 'accuracy': v['accuracy'], 'auc': v['auc']} for k, v in clf_results.items()},
        'features': feature_names
    }
    with open(outdir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print('\ntraining complete. models and scaler saved to', outdir)
    return report

def main():
    parser = argparse.ArgumentParser(description='train short-term volatility predictor')
    parser.add_argument('--data', type=str, default='data.csv', help='csv file with historical prices')
    parser.add_argument('--out', type=str, default='models', help='output folder for models')
    parser.add_argument('--rv-window', type=int, default=5, help='window (days) for realized volatility')
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f'could not find data file: {data_path}')

    df = load_data(data_path)
    df = make_features(df, rv_window=args.rv_window)
    X, y_reg, y_clf, feature_names = prepare_xy(df)

    report = train_and_eval(X, y_reg, y_clf, feature_names, Path(args.out))

    print('\nsummary report:')
    print(report)

if __name__ == '__main__':
    main()
