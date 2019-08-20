#!/bin/bash

#python local_train.py -experiment LocalSE3MultiResReprScalar -model SE3MultiResReprScalar -dataset Docking/SplitComplexes:training_set.dat:validation_set.dat -group SE3 -max_epoch 200 -load_epoch 127
python local_train.py -experiment LocalE3MultiResRepr4x4 -model E3MultiResRepr4x4 -dataset Docking/SplitComplexes:training_set.dat:validation_set.dat -group E3 -max_epoch 151 -load_epoch 149
