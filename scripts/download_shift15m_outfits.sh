#!/bin/bash

set -e

readonly ROOT=data

mkdir -p ${ROOT}
wget https://research.zozo.com/data_release/shift15m/label/iqon_outfits.json -P ${ROOT}
