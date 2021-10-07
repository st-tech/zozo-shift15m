#!/bin/bash

set -e

readonly ROOT=data

mkdir -p ${ROOT}
python shift15m/datasets/download_tarfiles.py --root ${ROOT}
python shift15m/datasets/feature_tar_extractor.py -d ${ROOT}
while read line
do
  rm ${ROOT}/${line}
done < ${ROOT}/tar_files.txt
