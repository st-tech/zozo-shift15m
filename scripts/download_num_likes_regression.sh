#!/bin/bash

set -e

readonly TASK_ROOT=data/num_likes_regression

mkdir -p ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/num_likes_regression/xy_00.pickle -P ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/num_likes_regression/xy_01.pickle -P ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/num_likes_regression/xy_02.pickle -P ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/num_likes_regression/xy_03.pickle -P ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/num_likes_regression/xy_04.pickle -P ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/num_likes_regression/xy_05.pickle -P ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/num_likes_regression/xy_06.pickle -P ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/num_likes_regression/xy_07.pickle -P ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/num_likes_regression/xy_08.pickle -P ${TASK_ROOT}
wget https://research.zozo.com/data_release/shift15m/num_likes_regression/xy_09.pickle -P ${TASK_ROOT}
