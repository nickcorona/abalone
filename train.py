import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from statsmodels.nonparaMETRIC.smoothers_lowess import lowess

df = pd.read_csv("data/abalone_original.csv")
df.info()

y = df["rings"]
X = df.drop("rings", axis=1)

X['sex'] = X['sex'].astype('category')

SEED = 0
Xt, Xv, yt, yv = train_test_split(X, y, random_state=SEED)
dt = lgb.Dataset(Xt, yt, silent=True)
dv = lgb.Dataset(Xv, yv, silent=True)


OBJECTIVE = "regression"
METRIC = "rmse"
MAXIMIZE = False

params = {
    "objective": OBJECTIVE,
    "METRIC": METRIC,
    "verbose": -1,
}

history = lgb.train(
    params,
    dt,
    valid_sets=[dt, dv],
    valid_names=['training', 'valid'],
    num_boost_round=10000,
    early_stopping_rounds=25,
    verbose_eval=10,
)


def loguniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))


best_etas = {"eta": [], "score": []}

for _ in range(60):
    eta = loguniform(-5, -1)
    best_etas["eta"].append(eta)
    params["eta"] = eta
    model = lgb.train(
        params,
        dt,
        valid_sets=[dt, dv],
        valid_names=['training', 'valid'],
        num_boost_round=10000,
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    best_etas["score"].append(model.best_score['valid'][METRIC])

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

# maximize or minimize METRIC
# e.g. binary loss needs minimizing, whereas AUC requires maximizing
if MAXIMIZE:
    best = np.argmax
else:
    best = np.argmin
good_eta = rounded_data[best(rounded_data[:, 1]), 0]

# plot relationship between learning rate and performance, with an eta selected just before diminishing returns
print(f"Good learning rate: {good_eta:4f}")
plt.axvline(good_eta, color="orange")
plt.title("Smoothed relationship between learning rate and METRIC.")
plt.xlabel("learning rate")
plt.ylabel(METRIC)
sns.lineplot(x=lowess_data[:, 0], y=lowess_data[:, 1])
plt.show()

params["eta"] = good_eta

model = lgb.train(
    params,
    dt,
    valid_sets=[dt, dv],
    valid_names=['training', 'valid'],
    num_boost_round=10000,
    early_stopping_rounds=50,
    verbose_eval=10,
)


threshold = 0.75
corr = Xt.corr(method="kendall")
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
upper = upper.stack()
high_upper = upper[(abs(upper) > threshold)]
abs_high_upper = abs(high_upper).sort_values(ascending=False)
pairs = abs_high_upper.index.to_list()
print(f"Correlated features: {pairs if len(pairs) > 0 else None}")


# drop correlated features
best_score = model.best_score["valid"][METRIC]
print(f"starting score: {best_score:.4f}")
drop_dict = {pair: [] for pair in pairs}
correlated_features = set()
for pair in pairs:
    for feature in pair:
        correlated_features.add(feature)
        Xt, Xv, yt, yv = train_test_split(
            X.drop(correlated_features, axis=1), y, random_state=SEED
        )
        dt = lgb.Dataset(Xt, yt, silent=True)
        dv = lgb.Dataset(Xv, yv, silent=True)
        drop_model = lgb.train(
            params,
            dt,
            valid_sets=[dt, dv],
            valid_names=["training", "valid"],
            num_boost_round=10000,
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        drop_dict[pair].append(drop_model.best_score["valid"][METRIC])
        correlated_features.remove(feature)  # remove from drop list
    pair_min = np.min(drop_dict[pair])
    if pair_min <= best_score:
        drop_feature = pair[
            np.argmin(drop_dict[pair])
        ]  # add to drop_feature the one that reduces score
        best_score = pair_min
        correlated_features.add(drop_feature)
print(f"ending score: {best_score:.4f}")
print(
    f"dropped features: {correlated_features if len(correlated_features) > 0 else None}"
)

best_score = model.best_score['valid'][METRIC]
print(f"starting score: {best_score:.4f}")
drop_unimportant_features = []
for feature in sorted_features:
    drop_unimportant_features.append(feature)
    dt = lgb.Dataset(Xt.drop(drop_unimportant_features, axis=1), y)
    dv = lgb.Dataset(Xv.drop(drop_unimportant_features, axis=1), y)

    drop_model = lgb.train(
        params,
        dt,
        valid_sets=[dt, dv],
        valid_names=['training', 'valid'],
        num_boost_round=10000,
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    score = drop_model.best_score['valid'][METRIC]
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
#     "METRIC": METRIC,
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
