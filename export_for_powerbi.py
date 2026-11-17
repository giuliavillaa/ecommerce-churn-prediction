"""Export the e-commerce churn data for a Power BI report.

Usage: python export_for_powerbi.py data.csv output_folder

Writes three CSV files (comma separated, UTF-8):
  customers.csv          one row per customer: original fields, imputed flags,
                         out-of-fold churn probability, risk band, tenure band
  feature_importance.csv permutation importance (drop in ROC AUC) per original feature
  model_metrics.csv      out-of-fold ROC AUC, accuracy, recall, precision

Same model as the notebook (200 trees, depth 5, learning rate 0.1, subsample 0.8)
and the same six dropped features. Two differences, both deliberate:
  1. Imputation and encoding are fitted inside each fold (fixes the leakage noted
     in the README).
  2. Every customer gets a probability from a model that never saw them
     (5-fold out-of-fold predictions), so the report can show all 5,630 customers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DROPPED = ["Gender", "OrderCount", "CashbackAmount", "CouponUsed", "CityTier", "HourSpendOnApp"]
SEED = 42


def main(data_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, sep=";", encoding="utf-8-sig")
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Same label clean-up as is common for this dataset: merge duplicate category names
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace({"Phone": "Mobile Phone"})
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})

    y = df["Churn"].astype(int).values
    X = df.drop(columns=["CustomerID", "Churn"] + DROPPED)
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])
    model = Pipeline([
        ("pre", pre),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              subsample=0.8, eval_metric="logloss", random_state=SEED)),
    ])

    oof = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, test_idx in skf.split(X, y):
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    pred = (oof >= 0.5).astype(int)
    metrics = pd.DataFrame([
        ("ROC AUC", roc_auc_score(y, oof)),
        ("Accuracy", accuracy_score(y, pred)),
        ("Recall (churned)", recall_score(y, pred)),
        ("Precision (churned)", precision_score(y, pred)),
        ("Customers", len(df)),
        ("Churned customers", int(y.sum())),
    ], columns=["Metric", "Value"])
    metrics["Value"] = metrics["Value"].round(3)
    metrics.to_csv(out / "model_metrics.csv", index=False)

    # Permutation importance on the held-out fold, per original feature:
    # how much ROC AUC drops when that column is shuffled. Avoids the bias of
    # summing gain over one-hot columns, which favours many-level categoricals.
    rng = np.random.RandomState(SEED)
    drops = {c: [] for c in X.columns}
    for train_idx, test_idx in skf.split(X, y):
        model.fit(X.iloc[train_idx], y[train_idx])
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        base = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        for col in X.columns:
            for _ in range(3):
                Xp = X_test.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                drops[col].append(base - roc_auc_score(y_test, model.predict_proba(Xp)[:, 1]))
    fi = pd.DataFrame({"Feature": list(drops), "AUCDrop": [np.mean(v) for v in drops.values()]})
    fi = fi.sort_values("AUCDrop", ascending=False)
    fi["AUCDrop"] = fi["AUCDrop"].round(4)
    fi["Rank"] = range(1, len(fi) + 1)
    fi.to_csv(out / "feature_importance.csv", index=False)

    cust = df.copy()
    cust["ChurnProbability"] = oof.round(4)
    cust["RiskBand"] = pd.cut(oof, [-0.01, 0.2, 0.5, 1.0], labels=["Low", "Medium", "High"]).astype(str)
    tb = pd.cut(cust["Tenure"], [-1, 1, 5, 12, 100], labels=["0 to 1 months", "2 to 5 months", "6 to 12 months", "Over 12 months"])
    cust["TenureBand"] = tb.cat.add_categories("Unknown").fillna("Unknown").astype(str)
    cust["ComplainLabel"] = cust["Complain"].map({1: "Complaint filed", 0: "No complaint"})
    cust["ChurnLabel"] = cust["Churn"].map({1: "Churned", 0: "Stayed"})
    cust.to_csv(out / "customers.csv", index=False)

    print(metrics.to_string(index=False))
    print(fi.head(8).to_string(index=False))
    print(cust["RiskBand"].value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
