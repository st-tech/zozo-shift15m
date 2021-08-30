#!/bin/bash

set -e

readonly ROOT=data

mkdir -p ${ROOT}
wget -i https://research.zozo.com/data_release/shift15m/vgg16-features/filelist.txt -P ${ROOT}
python shift15m/datasets/feature_tar_extractor.py -d ${ROOT}
while read line
do
  rm ${ROOT}/${line}
done < ${ROOT}/tar_files.txt