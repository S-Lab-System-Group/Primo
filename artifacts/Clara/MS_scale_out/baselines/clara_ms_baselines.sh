#!/bin/sh
start=$(date "+%s")
echo /////////////////////////Preprocessing data//////////////////////////////////
python3 preprocess_hit.py --dataset dataset_hit --testset testset_hit
echo
echo /////////////////////////Clara GBDT model////////////////////////////////////
python3 train_gbdt.py
echo
echo /////////////////////////Decision tree model/////////////////////////////////
python3 train_dt.py
echo
echo /////////////////////////Delete trivial features to boost up performance for weaker baselines//////////////////////////////////
python3 preprocess_hit_simple.py --dataset dataset_hit --testset testset_hit
echo
echo /////////////////////////Neural network model//////////////////////////////
python3 train_dnn.py
echo
echo /////////////////////////Support Vector Machine model//////////////////////////////// 
python3 train_svm.py
echo
echo ------------------------------
now=$(date "+%s")
time=$((now-start))
echo "Scale out analysis time used:$time seconds"
