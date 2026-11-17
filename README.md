# E-commerce churn prediction

Binary churn classification on e-commerce customer records, with exploratory analysis
driving feature selection and gradient boosting for the model.

Artificial Intelligence for Marketing and Customer Analytics, BSc in Artificial
Intelligence, University of Pavia, University of Milano-Bicocca, University of Milan.

![ROC curve and feature importance](figures/churn.jpg)

*Left: ROC on the held-out set. Right: the twenty highest-ranked features.*

## Task

5,630 customer records with 20 attributes covering tenure, order history, payment and
login preferences, satisfaction score, complaints, and delivery distance. Target:
whether the customer churned.

## Method

**Exploratory analysis.** Missingness inspected with `missingno`, univariate
distributions and boxplots per feature, bivariate analysis of each feature against the
target, churn rate by category, and a correlation matrix. Six features were dropped as
uninformative or redundant: gender, order count, cashback amount, coupons used, city
tier and hours spent on the app.

**Preprocessing.** Median imputation for numerical features, mode imputation for
categorical ones, one-hot encoding for the remaining categoricals.

**Model.** XGBoost tuned by grid search over 32 configurations with 3-fold
cross-validation on ROC AUC. Selected configuration: 200 estimators, maximum depth 5,
learning rate 0.1, subsample 0.8.

## Results

| Metric | Value |
|---|---|
| ROC AUC | 0.991 |
| Accuracy | 0.965 |
| Recall | 0.853 |
| Precision | 0.936 |

Feature importance ranks tenure first, followed by whether the customer had filed a
complaint. Both are consistent with the bivariate analysis, where they separate the
classes clearly. Preferred order category, marital status and payment method follow.

ROC AUC is reported alongside accuracy because the classes are imbalanced: unlike
accuracy, it is not inflated by predicting the majority class.

## Power BI report

The model is also the basis of a two-page report built in Power BI Service.

`export_for_powerbi.py` prepares the data for it. It uses the same XGBoost configuration
and the same dropped features as the notebook, with two changes: imputation and encoding
are fitted inside each fold, and every customer gets an out-of-fold churn probability from
5-fold cross-validation, so each customer is scored by a model that never saw them. On
these predictions ROC AUC is 0.986, accuracy 0.959, recall 0.833, precision 0.916.

```
python export_for_powerbi.py data.csv output_folder
```

It writes `customers.csv` (one row per customer with churn probability, risk band and
tenure band), `feature_importance.csv` (permutation importance: drop in ROC AUC when a
feature is shuffled) and `model_metrics.csv`.

- **Overview page**: churn by risk band and by tenure, what drives the model, and three
  KPI cards. 52% of customers in their first month churn, against about 5% after six
  months. Tenure is by far the strongest driver.
- **Who to call page**: the 72 high-risk customers (probability above 0.5) who have not
  churned, sorted by probability and filterable by tenure, complaint and product category.
  63 of them joined less than a month ago.

## Limitations

Imputation is performed on the full dataset before the train/test split, so test-set
statistics influence the imputed values in training. The effect is small for median and
mode, but the correct structure is to fit the imputers inside a pipeline. `export_for_powerbi.py`
already does this with a `ColumnTransformer` fitted per fold.

Class imbalance is not explicitly addressed through resampling or class weights; the
model handles it adequately here, but recall on the churned class (0.853) is the metric
that would benefit most from doing so.

## Repository

```
churn_model.ipynb        EDA, preprocessing, grid search, evaluation
export_for_powerbi.py    out-of-fold scoring and export for the Power BI report
figures/                 ROC curve, feature importance
```
