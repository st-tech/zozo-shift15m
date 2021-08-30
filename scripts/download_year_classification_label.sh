#!/bin/bash
set -e

readonly TASK_ROOT=data/year_classification

mkdir -p ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/year_classification/year_classification_labels.tar.gz -P ${TASK_ROOT}
tar zxvf inputs/year_classification/year_classification_labels.tar.gz -C ${TASK_ROOT}
