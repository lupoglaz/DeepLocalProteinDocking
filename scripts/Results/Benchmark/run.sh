#!/bin/bash

python EvaluateBenchmark.py \
-experiment ZDOCK \
-dataset DockingBenchmarkV4 \
-table TableS1.csv \
-threshold_clash 300 \
-angle_inc 15 \
-overwrite 1 \
-plot 1

python EvaluateBenchmark.py \
-experiment LocalE3MultiResRepr4x4 \
-dataset DockingBenchmarkV4 \
-table TableS1.csv \
-threshold_clash 300 \
-angle_inc 15 \
-overwrite 1 \
-plot 1

python EvaluateBenchmark.py \
-experiment LocalSE3MultiResReprScalar \
-dataset DockingBenchmarkV4 \
-table TableS1.csv \
-threshold_clash 300 \
-angle_inc 15 \
-overwrite 1 \
-plot 1