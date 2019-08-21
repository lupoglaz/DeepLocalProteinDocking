#!/bin/bash

python local_test.py \
-start 0 \
-end 150 \
-angle_inc 15 \
-threshold_clash 400 \
-experiment LocalSE3MultiResReprScalar \
-group SE3 \
-model SE3MultiResReprScalar \
-filter SimpleFilter \
-dataset DockingBenchmarkV4:TableS2.csv \
-load_epoch 99 \
-rewrite 0