#!/bin/bash

wget -i https://research.zozo.com/data_release/shift15m/vgg16-features/filelist.txt -P inputs/cnn-features
python shift15m/datasets/feature_tar_extractor.py -d inputs/cnn-features
