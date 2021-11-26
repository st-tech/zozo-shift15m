<p align="center">
  <img src="./assets/shift15m.png" width="70%" style="display: block; margin: 0 auto" />
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/st-tech/zozo-shift15m)
[![Downloads](https://static.pepy.tech/personalized-badge/shift15m?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/shift15m)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/st-tech/zozo-shift15m/tests)
[![PyPI version](https://badge.fury.io/py/shift15m.svg)](https://badge.fury.io/py/shift15m)
![GitHub issues](https://img.shields.io/github/issues/st-tech/zozo-shift15m)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/st-tech/zozo-shift15m)
![GitHub last commit](https://img.shields.io/github/last-commit/st-tech/zozo-shift15m)
[![arXiv](https://img.shields.io/badge/arXiv-2108.12992-b31b1b.svg)](https://arxiv.org/abs/2108.12992)

[[arXiv]](https://arxiv.org/abs/2108.12992)

The main motivation of the SHIFT15M project is to provide a dataset that contains natural dataset shifts collected from a web service IQON, which was actually in operation for a decade.
In addition, the SHIFT15M dataset has several types of dataset shifts, allowing us to evaluate the robustness of the model to different types of shifts (e.g., covariate shift and target shift).

We provide the [Datasheet for SHIFT15M](./DATASHEET.md).
This datasheet is based on the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) [1] template.

| System      | Python 3.6 | Python 3.7 | Python 3.8 |
| :---:              | :---:             | :---:            | :--:              |
| Linux CPU    |  <img src="https://img.shields.io/badge/build-success-brightgreen" /> | <img src="https://img.shields.io/badge/build-success-brightgreen" /> | <img src="https://img.shields.io/badge/build-success-brightgreen" /> |
| Linux GPU    |   <img src="https://img.shields.io/badge/build-success-brightgreen" />  | <img src="https://img.shields.io/badge/build-success-brightgreen" /> | <img src="https://img.shields.io/badge/build-success-brightgreen" /> |
| Windows CPU / GPU | <center>Status Currently Unavailable</center> | <center>Status Currently Unavailable</center> |  <center>Status Currently Unavailable</center> |
| Mac OS CPU|   <img src="https://img.shields.io/badge/build-success-brightgreen" /> |  <img src="https://img.shields.io/badge/build-success-brightgreen" />   |  <img src="https://img.shields.io/badge/build-success-brightgreen" /> |

SHIFT15M is a large-scale dataset based on approximately 15 million items accumulated by the fashion search service IQON.

![](./assets/iqon.png)

## Installation

### From PyPi

```bash
$ pip install shift15m
```

### From source

```bash
$ git clone https://github.com/st-tech/zozo-shift15m.git
$ cd zozo-shift15m
$ poetry build
$ pip install dist/shift15m-xxxx-py3-none-any.whl
```

## Download SHIFT15M dataset

### Use Dataset class

You can download SHIFT15M dataset as follows:

```python
from shift15m.datasets import NumLikesRegression

dataset = NumLikesRegression(root="./data", download=True)
```

### Download directly by using download scripts

Please download the dataset as follows:

```bash
$ bash scripts/download_all.sh
```

To avoid downloading the test dataset for set matching (80GB), which is not required in training, you can use the following script.

```bash
$ bash scripts/download_all_wo_set_testdata.sh
```

## Tasks

The following tasks are now available:

| Tasks                      | Task type           | Shift type   | # of input dim | # of output dim |
|----------------------------|---------------------|--------------|----------------|-----------------|
| [NumLikesRegression](https://github.com/st-tech/zozo-shift15m/tree/main/benchmarks#regression-for-the-number-of-likes) | regression     | target shift |     (N, 25)                       | (N, 1)               |
| [SumPricesRegression](https://github.com/st-tech/zozo-shift15m/tree/main/benchmarks#regression-for-the-sum-of-prices) | regression    | covariate shift, target shift | (N, 1) | (N, 1)|
| ItemPriceRegression        | regression          | target shift | (N, 4096)      | (N, 1)          |
| [ItemCategoryClassification](https://github.com/st-tech/zozo-shift15m/tree/main/benchmarks/item_category_prediction) | classification      | target shift| (N, 4096)      | (N, 7)          |
| [Set2SetMatching](https://github.com/st-tech/zozo-shift15m/tree/main/benchmarks/set_matching)      | set-to-set matching | covariate shift | (N, 4096)x(M, 4096)| (1)          |

## Benchmarks

As templates for numerical experiments on the SHIFT15M dataset, we have published [experimental results for each task with several models](./benchmarks).

## Original Dataset Structure

The original dataset is maintained in json format, and a row consists of the following:

```
{
  "user":{"user_id":"xxxx", "fav_brand_ids":"xxxx,xx,..."},
  "like_num":"xx",
  "set_id":"xxx",
  "items":[
    {"price":"xxxx","item_id":"xxxxxx","category_id1":"xx","category_id2":"xxxxx"},
    ...
  ],
  "publish_date":"yyyy-mm-dd"
}
```

## Contributing
To learn more about making a contribution to SHIFT15M, please see the following materials:
- [Developers Guide](./DEVELOPMENT.md)
- [Task Proposal Guide](./TASK_PROPOSAL.md)
- [Benchmark Proposal Guide](./BENCHMARK.md)

## License
The dataset itself is provided under a [CC BY-NC 4.0 license](./LICENSE.CC).
On the other hand, the software in this repository is provided under the [MIT license](./LICENSE.MIT).

## Dataset metadata
The following table is necessary for this dataset to be indexed by search engines such as [Google Dataset Search](https://datasetsearch.research.google.com/).

<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">SHIFT15M Dataset</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">SHIFT15M</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">shift15m-dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/st-tech/zozo-shift15m</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/st-tech/zozo-shift15m</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">SHIFT15M is a multi-objective, multi-domain dataset which includes multiple dataset shifts.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">ZOZO Research</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://ja.wikipedia.org/wiki/ZOZO</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">CC BY-NC 4.0</code></td>
          </tr>
          <tr>
            <td>url</td>
            <td><code itemprop="url">https://github.com/st-tech/zozo-shift15m/blob/main/LICENSE.CC</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
</table>
</div>


## Errata
No errata are currently available.

## References
- [1] Gebru, Timnit, et al. "Datasheets for datasets." arXiv preprint arXiv:1803.09010 (2018).
