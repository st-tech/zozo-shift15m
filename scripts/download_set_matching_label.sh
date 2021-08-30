#!/bin/bash

set -e

readonly TASK_ROOT=data/set_matching

mkdir -p ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/set_matching/set_matching_labels.tar.gz -P ${TASK_ROOT}
tar zxvf inputs/set_matching/set_matching_labels.tar.gz -C ${TASK_ROOT}
