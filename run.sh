#!/bin/bash

python main.py \
    --root ~/Data/Datasets/Diabetic_Retinopathy/images --csv ~/Data/Datasets/Diabetic_Retinopathy/csv_files \
    --net resnet50 --pretrained \
    --input_shape 512 512 --epochs 10 \
    --save weights/model_50_pretrained.weight \
    --trainable

python main.py \
    --root ~/Data/Datasets/Diabetic_Retinopathy/images --csv ~/Data/Datasets/Diabetic_Retinopathy/csv_files \
    --net resnet50 \
    --input_shape 512 512 --epochs 10 \
    --save weights/model_50_scratch.weight \
    --trainable

python main.py \
    --root ~/Data/Datasets/Diabetic_Retinopathy/images --csv ~/Data/Datasets/Diabetic_Retinopathy/csv_files \
    --net resnet18 --pretrained \
    --input_shape 512 512 --epochs 10 \
    --save weights/model_18_pretrained.weight \
    --trainable

python main.py \
    --root ~/Data/Datasets/Diabetic_Retinopathy/images --csv ~/Data/Datasets/Diabetic_Retinopathy/csv_files \
    --net resnet18 \
    --input_shape 512 512 --epochs 10 \
    --save weights/model_18_scratch.weight \
    --trainable
