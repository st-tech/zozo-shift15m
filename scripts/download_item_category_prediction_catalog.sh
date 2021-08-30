#!/bin/bash

set -e

readonly ROOT=data

mkdir -p ${ROOT}
wget https://research.zozo.com/data_release/shift15m/item_category_prediction/item_catalog.txt -P ${ROOT}
