#!/bin/bash

set -e

readonly TASK_ROOT=data/sum_prices_regression

mkdir -p ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/sum_prices_regression/sumprices.pickle -P ${TASK_ROOT}
