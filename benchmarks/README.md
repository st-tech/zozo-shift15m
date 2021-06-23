# Benchmarks

## Regression for the number of likes

* task type: regression
* shift type: target shift
* shift metric: wasserstein distance
* train/test sample size
  * train sample size: 100000
  * test sample size: 100000
* input/output dimension
  * input dimension: 25
  * output dimension: 1
* number of trials: 20
* hyperparameters: All hyperparameters are default settings of the [scikit-learn](https://scikit-learn.org/stable/index.html).
* source code is available [here](bechmarks/numlikes_tabular.py).

| Models            | W=0            | W=5            | W=10           | W=15           | W=20           | W=25           | W=30            | W=35           | W=40           | W=45           | W=50           |
|-------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Linear Regression | 9.364(±0.027)  | 9.495(±0.033)  | 10.446(±0.048) | 12.689(±0.053) | 17.101(±0.060) | 23.016(±0.056) | 28.800(±0.058) | 34.292(±0.047) | 39.564(±0.050) | 44.462(±0.050) | 48.844(±0.056) |
| RANSAC Regression | 10.394(±0.273) | 12.408(±1.273) | 15.934(±1.223) | 19.586(±1.161) | 23.700(±1.958) | 28.212(±1.398) | 33.012(±1.076) | 37.573(±1.600) | 42.726(±1.334) | 47.380(±1.358) | 53.038(±0.890) |
| HuberRegression   | 30.347(±0.939) | 30.428(±0.816) | 31.025(±0.712) | 32.496(±0.598) | 35.211(±0.475) | 38.747(±0.463) | 42.474(±0.391) | 46.224(±0.396) | 49.836(±0.302) | 53.244(±0.268) | 56.524(±0.221) |
| Decision Tree     | 12.828(±0.059) | 14.293(±0.096) | 16.671(±0.124) | 19.462(±0.197) | 22.182(±0.305) | 25.387(±0.251) | 29.131(±1.396) | 33.152(±0.396) | 37.046(±0.166) | 41.240(±0.325) | 44.959(±0.498) |

![](../assets/benchmarks/numlikes_regression.png)
