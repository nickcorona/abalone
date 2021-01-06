import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess

df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")
df.info()

y = df["DEATH_EVENT"]
X = df.drop("DEATH_EVENT", axis=1)

d = lgb.Dataset(X, y, silent=True)

OBJECTIVE = "binary"
METRIC = "binary_logloss"
MAXIMIZE = False

params = {
    "objective": OBJECTIVE,
    "metric": METRIC,
    "force_col_wise": True,
    "verbose": -1,
}

history = lgb.cv(
    params,
    d,
    num_boost_round=10000,
    early_stopping_rounds=50,
    verbose_eval=10,
    return_cvbooster=False,
)


def loguniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))


best_etas = {"eta": [], "score": []}

for _ in range(60):
    eta = loguniform(-4, 0)
    best_etas["eta"].append(eta)
    params["eta"] = eta
    model = lgb.cv(
        params,
        d,
        num_boost_round=10000,
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    best_etas["score"].append(model[f"{METRIC}-mean"][-1])

best_eta_df = pd.DataFrame.from_dict(best_etas)
lowess_data = lowess(
    best_eta_df["score"],
    best_eta_df["eta"],
)

# use log scale as it's easier to observe the whole graph
plt.xscale("log")
rounded_data = lowess_data.copy()
rounded_data[:, 1] = rounded_data[:, 1].round(4)
rounded_data = rounded_data[::-1]  # reverse to find first best

# maximize or minimize metric
# e.g. binary loss needs minimizing, whereas AUC requires maximizing
if MAXIMIZE:
    best = np.argmax
else:
    best = np.argmin
good_eta = rounded_data[best(rounded_data[:, 1]), 0]

# plot relationship between learning rate and performance, with an eta selected just before diminishing returns
print(f"Good learning rate: {good_eta:4f}")
plt.axvline(good_eta, color="orange")
plt.title("Smoothed relationship between learning rate and metric.")
plt.xlabel("learning rate")
plt.ylabel(METRIC)
sns.lineplot(x=lowess_data[:, 0], y=lowess_data[:, 1])
plt.show()

params["eta"] = good_eta

history = lgb.cv(
    params,
    d,
    num_boost_round=10000,
    early_stopping_rounds=50,
    verbose_eval=10,
    return_cvbooster=True,
)


threshold = 0.75
corr = X.corr(method="kendall")
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
upper = upper.stack()
high_upper = upper[(abs(upper) > threshold)]
abs_high_upper = abs(high_upper).sort_values(ascending=False)
pairs = abs_high_upper.index.to_list()
print(f"Correlated features: {pairs if len(pairs) > 0 else None}")

model = history["cvbooster"]
importances = model.feature_importance(importance_type="split")
n_features = len(importances[0])
n_folds = len(importances)
average_importance = np.array([0] * n_features)
for importance in importances:
    average_importance += importance
total_importance = average_importance / n_folds

sorted_features = [
    feature
    for _, feature in sorted(
        zip(average_importance, model.feature_name()[0]),
        reverse=False,
    )
]

best_score = history[f"{METRIC}-mean"][-1]
print(f"starting score: {best_score:.4f}")
drop_unimportant_features = []
for feature in sorted_features:
    drop_unimportant_features.append(feature)
    d = lgb.Dataset(X.drop(drop_unimportant_features, axis=1), y)
    drop_history = lgb.cv(
        params,
        d,
        num_boost_round=10000,
        early_stopping_rounds=50,
        verbose_eval=False,
        stratified=False,
    )
    score = drop_history[f"{METRIC}-mean"][-1]
    if score > best_score:
        del drop_unimportant_features[-1]  # remove from drop list
        print(f"Dropping {feature} worsened score to {score:.4f}.")
        break
    else:
        best_score = score
print(f"ending score: {best_score:.4f}")
print(
    f"dropped features: {drop_unimportant_features if len(drop_unimportant_features) > 0 else None}"
)

import optuna.integration.lightgbm as lgb
from sklearn.model_selection import KFold

# params = {
#     "objective": OBJECTIVE,
#     "metric": METRIC,
#     "verbose": -1,
#     "boosting_type": "gbdt",
#     "eta": good_eta,
# }

d = lgb.Dataset(X, y)

tuner = lgb.LightGBMTunerCV(
    params, d, verbose_eval=False, early_stopping_rounds=50, folds=KFold(n_splits=3)
)
tuner.run()
score = history[f"{METRIC}-mean"][-1]

best_params = tuner.best_params
print("Best params:", best_params)
print(f"  {METRIC} = {score}")
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))

history = lgb.cv(
    best_params,
    d,
    num_boost_round=10000,
    early_stopping_rounds=50,
    verbose_eval=10,
    return_cvbooster=True,
)

model = history["cvbooster"]
importances = model.feature_importance(importance_type="split")
n_features = len(importances[0])
n_folds = len(importances)
average_importance = np.array([0] * n_features)
for importance in importances:
    average_importance += importance
average_importance = average_importance / n_folds

idx = np.argsort(average_importance)
sns.barplot(x=average_importance, y=model.feature_name()[0])
plt.show()
