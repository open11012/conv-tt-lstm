#!/bin/bash

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

cd "$(dirname "$0")"
cd ..

# Pytorch standard implementation
python3 model_train.py --dataset MNIST --use-sigmoid --img-channels 1 --img-height 64 --img-width 64 --kernel-size 5 --model convlstm --batch-size 1 --learning-rate 1e-3 --valid-samples 448 --num-epochs 500 --ssr-decay-ratio 4e-3 # --use-amp --use-checkpointing
