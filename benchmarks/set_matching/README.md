# Set-to-set matching

## Set up

### Requirements
- Python 3.6.0+
- [Chainer](https://github.com/chainer/chainer/) 5.0.0+
- [numpy](https://github.com/numpy/numpy)
- [cupy](https://github.com/cupy/cupy) (optional)
- [tqdm](https://github.com/tqdm/tqdm)
- and their dependencies

### Install Chainer

```
$ pip install chainer
```

For GPU, you need to install Cupy. Please refer to the [installation guide](https://docs.cupy.dev/en/latest/install.html).

## Training

### Vanilla set matching model

You can train a set matching model indicating the label directory.
For example, to train a model from the training data collected in the year 2013 split as 'label1', you can use the code as follows:

```
$ python outfits/train.py -m set_matching_sim -b 32 -e 32 -i ../../inputs/cnn-features -l ../../inputs/set_matching/set_matching_labels/2013-2014-label1 -o result_s2s/2013-label1 -gpu -1
```

Note that all the training data in the same label split are identical, so you can select any label directories regardless of the validation year.

Also, you can reduce the required memory size by setting minibatch-size -b to a small number.

### Weighted set matching model

To obtain a set matching model trained with the weighted loss, you need to train a weight estimator first and then the extended set matching model.

Also, setting training and testing years here are required for covariate adaptation. The following examples are adapting the data of the year 2013 to the year 2014.

#### Weight estimator

```
$ python weight_estimation/train.py -b 128 -e 16 -i ../../inputs/cnn-features -l ../../inputs/set_matching/set_matching_labels/2013-2014-label1 -o result_weight/2013-2014-label1 -gpu -1
```

#### Weighted training on set matching model

You can select the weighting strategy `-m cov_max` or `-m cov_mean,` which represent max-IW or mean-IW, respectively.

To compare the results, we recommend using the same number of minibatch-size and training epochs as the vanilla set matching model.

```
$ python outfits/train.py -m cov_max -b 32 -e 32 -i ../../inputs/cnn-features -l ../../inputs/set_matching/set_matching_labels/2013-2014-label1 -o result_s2s_cov_max/2013-2014-label1 -gpu -1 -w result_weight/2013-2014-label1
```

## Testing

Using our test data, you can evaluate the trained model.
For example, the command for testing the *cov-max* weighted model on the set matching task, a.k.a. Fill-in-the-N-blank with four candidates in the covariate assumption for the years from 2013 to 2014, is:

```
$ python outfits/test.py -i ../../inputs/set_matching_test_data/test_ncand4/2013-2014-label1 -d result_s2s_cov_max/2013-2014-label1 -gpu -1
```

# Remarks

The set matching modules on this repository use the [OSS](https://github.com/soskek/attention_is_all_you_need), which is distributed under the [BSD 3-Clause License](networks/LICENSE).

