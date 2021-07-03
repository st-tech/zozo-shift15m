import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import norm
from scipy.stats import wasserstein_distance
import tqdm
from shift28m.datasets import NumLikesRegression

# set the plot style.
sns.set_style("whitegrid")

# load dataset from SHIFT28M
dataset = NumLikesRegression()

#############################
# Experimental setup
#############################
n_trials = 20  # number of trials.
train_sample_size = 100000  # number of training data
test_sample_size = 100000  # number of test data
test_mu = 80  # mean of test distribution
test_sigma = 10  # std of test distribution


# specify the dataset shift.
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

# specify the model list
models = [
    {"model": LinearRegression, "sample_weights": "ERM"},
]


models_errors_mean = []
models_errors_std = []
dists = []

for k in range(len(models)):
    model = models[k]["model"]
    weighting = models[k]["sample_weights"]

    model_errors_mean = []
    model_errors_std = []
    model_dists = []
    for shift in shifts:
        errors = []

        # compute the distance between train dataset and test dataset.
        rv_train = np.random.normal(shift["train_mu"], shift["train_sigma"], 10000)
        rv_test = np.random.normal(test_mu, test_sigma, 10000)
        wd = wasserstein_distance(rv_train, rv_test)
        model_dists.append(wd)

        for i in tqdm.tqdm(range(n_trials)):
            (x_train, y_train), (x_test, y_test) = dataset.load_dataset(
                target_shift=True,
                train_size=train_sample_size,
                test_size=test_sample_size,
                test_mu=test_mu,
                test_sigma=test_sigma,
                train_mu=shift["train_mu"],
                train_sigma=shift["train_sigma"],
                random_seed=i,
            )

            p_tr = norm.pdf(y_train, loc=shift["train_mu"], scale=shift["train_sigma"])
            p_te = norm.pdf(y_train, loc=test_mu, scale=test_sigma)
            reg = model()
            if weighting == "ERM":
                reg.fit(x_train, y_train)

            errors.append(mae(reg.predict(x_test), y_test))

        model_errors_mean.append(np.mean(errors))
        model_errors_std.append(np.std(errors))

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
        label=models[i]["sample_weights"],
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
# plt.savefig("numlikes_regression.png")
plt.show()
