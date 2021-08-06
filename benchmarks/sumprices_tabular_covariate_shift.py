import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import wasserstein_distance
import tqdm
from shift15m.datasets import SumPricesRegression


sns.set_style("whitegrid")


dataset = SumPricesRegression()

n_trials = 20
train_sample_size = 100000
test_sample_size = 100000
test_mu = 80
test_sigma = 10

shifts = [
    {"train_mu": 80, "train_sigma": 10},
    {"train_mu": 75, "train_sigma": 10},
    {"train_mu": 70, "train_sigma": 10},
    {"train_mu": 65, "train_sigma": 10},
    {"train_mu": 60, "train_sigma": 10},
    {"train_mu": 55, "train_sigma": 10},
    {"train_mu": 50, "train_sigma": 10},
    {"train_mu": 45, "train_sigma": 10},
    {"train_mu": 40, "train_sigma": 10},
    {"train_mu": 35, "train_sigma": 10},
    {"train_mu": 30, "train_sigma": 10},
]

models = [
    TheilSenRegressor,
    HuberRegressor,
    RANSACRegressor,
    LinearRegression,
    DecisionTreeRegressor,
]

models_errors_mean = []
models_errors_std = []
dists = []

for model in models:
    model_errors_mean = []
    model_errors_std = []
    model_dists = []
    for shift in shifts:
        errors = []

        rv_train = np.random.normal(shift["train_mu"], shift["train_sigma"], 10000)
        rv_test = np.random.normal(test_mu, test_sigma, 10000)
        wd = wasserstein_distance(rv_train, rv_test)
        model_dists.append(wd)

        for i in tqdm.tqdm(range(n_trials)):
            (x_train, y_train), (x_test, y_test) = dataset.load_dataset(
                covariate_shift=True,
                train_size=train_sample_size,
                test_size=test_sample_size,
                test_mu=test_mu,
                test_sigma=test_sigma,
                train_mu=shift["train_mu"],
                train_sigma=shift["train_sigma"],
                random_seed=i,
            )

            x_train = x_train.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)

            reg = model()
            reg.fit(x_train, y_train)
            errors.append(mae(reg.predict(x_test), y_test))

        model_errors_mean.append(np.mean(errors))
        model_errors_std.append(np.std(errors))
        print(model_errors_mean)
        print(model_errors_std)
    models_errors_mean.append(model_errors_mean)
    models_errors_std.append(model_errors_std)
    dists.append(model_dists)

models_errors_mean = np.array(models_errors_mean)
models_errors_std = np.array(models_errors_std)

colors = ["purple", "green", "blue", "darkcyan", "red"]

print(dists)
print(models_errors_mean)
print(models_errors_std)

fig = plt.figure(figsize=(12, 6))
for i in range(len(models)):
    plt.plot(
        dists[i],
        models_errors_mean[i],
        alpha=0.8,
        color=colors[i],
        label=models[i].__name__,
    )
    plt.fill_between(
        dists[i],
        models_errors_mean[i],
        models_errors_mean[i] + models_errors_std[i],
        alpha=0.2,
        color=colors[i],
    )
    plt.fill_between(
        dists[i],
        models_errors_mean[i],
        models_errors_mean[i] - models_errors_std[i],
        alpha=0.2,
        color=colors[i],
    )

plt.legend()
plt.xlabel(r"$W_1(P_{train}, P_{test})$")
plt.ylabel("MAE")
plt.yscale("log")
plt.savefig("sumprices_regression_covariate_shift.png")
plt.show()
