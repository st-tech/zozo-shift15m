#!/bin/bash

wget -i https://research.zozo.com/data_release/shift15m/set_matching/set_matching_test_data_splits/filelist.txt -P inputs/set_matching/set_matching_test_data_splits
cat inputs/set_matching/set_matching_test_data_splits/set_matching_test_data.tar.gz-* > inputs/set_matching/set_matching_test_data.tar.gz
mkdir inputs/set_matching/set_matching_test_data
tar zxvf inputs/set_matching/set_matching_test_data.tar.gz -C inputs/set_matching
