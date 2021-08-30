#!/bin/bash

set -e

readonly TASK_ROOT=data/set_matching

mkdir -p ${TASK_ROOT}
wget -i https://research.zozo.com/data_release/shift15m/set_matching/set_matching_test_data_splits/filelist.txt -P ${TASK_ROOT}/set_matching_test_data_splits
cat inputs/set_matching/set_matching_test_data_splits/set_matching_test_data.tar.gz-* > ${TASK_ROOT}/set_matching_test_data.tar.gz
tar zxvf inputs/set_matching/set_matching_test_data.tar.gz -C ${TASK_ROOT}
