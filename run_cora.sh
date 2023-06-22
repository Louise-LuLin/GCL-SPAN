#!/bin/bash 

DATA=$1
DEV=$2

for LR in 10 50 100 200
do
    python3 unsupervised_node.py --device=$DEV --dataset=$DATA --aug_lr1=$LR --aug_lr2=0.001 --aug_iter=20 --pe=0.2 &> "./results/"$DATA\_$LR\_"0.1_0.2.txt"
done