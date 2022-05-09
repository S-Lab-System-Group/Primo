import numpy as np
from xgbranker import XGBRanker
import sys
import os
from collections import defaultdict

# import matplotlib.pyplot as plt
import itertools
import xgboost as xg
import random
import pickle
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset")
parser.add_argument("--testset", help="testset")
args = parser.parse_args()


def feature_extraction(file_name, length):
    features = []
    tags = []
    groups = []
    groupid = 0
    elements = [
        "ISIPFILTER1",
        "ISIPFILTER2",
        "ISIPREWRITER",
        "ISAVCOUNTER",
        "ISCOUNTER",
        "ISFTPPORTMAPPER",
        "ISWEBGEN",
        "ISUDPIPENCAP",
        "ISTCPDEMUX",
        "ISTCPCONN",
        "ISCOMPUTE1",
        "ISCOMPUTE2",
        "ISCOMPUTE3",
        "ISCOMPUTE4",
        "ISCOMPUTE5",
        "ISCOMPUTE6",
        "ISCOMPUTE7",
        "ISCOMPUTE8",
        "ISCOMPUTE9",
        "ISCOMPUTE10",
        "ISACCELERATE1",
        "ISACCELERATE2",
        "ISACCELERATE3",
        "ISACCELERATE4",
        "ISACCELERATE5",
        "ISACCELERATE6",
        "ISACCELERATE7",
        "ISACCELERATE8",
        "ISACCELERATE9",
        "ISACCELERATE10",
        "ISIMEM0",
        "ISIMEM1",
        "ISIMEM2",
        "ISIMEM3",
        "ISIMEM4",
    ]
    defines = [
        "RULE_NUM1",
        "RULE_NUM2",
        "BUCKET_SIZE1",
        "BUCKET_SIZE2",
        "BUCKET_SIZE3",
        "BUCKET_SIZE4",
        "BUCKET_SIZE5",
        "TUNECOMPUTE1",
        "TUNECOMPUTE2",
        "TUNECOMPUTE3",
        "TUNECOMPUTE4",
        "TUNECOMPUTE5",
        "TUNECOMPUTE6",
        "TUNECOMPUTE7",
        "TUNECOMPUTE8",
        "TUNECOMPUTE9",
        "TUNECOMPUTE10",
        "TUNEACCELERATE1",
        "TUNEACCELERATE2",
        "TUNEACCELERATE3",
        "TUNEACCELERATE4",
        "TUNEACCELERATE5",
        "TUNEACCELERATE6",
        "TUNEACCELERATE7",
        "TUNEACCELERATE8",
        "TUNEACCELERATE9",
        "TUNEACCELERATE10",
        "TUNEIMEM0",
        "TUNEIMEM1",
        "TUNEIMEM2",
        "TUNEIMEM3",
        "TUNEIMEM4",
    ]
    mapping = defaultdict(list)
    groupid = 0
    for n in range(0, length):
        groupid += 1
        try:
            ff = open(file_name + "/trainingset_feature_" + str(n), "r")
            ft = open(file_name + "/trainingset_tag_" + str(n), "r")
        except:
            print(n)
            continue
        # mapping = defaultdict(list)

        line = ft.readline()
        throughputs = []
        while line:
            if "throughput" in line:
                start = line.find("throughput:")
                begin = line[start:].find(" ") + 1 + start
                end = line[start:].find(";") + start
                throughputs.append(float(line[begin:end]))
            # mapping[groupid].append(top_choice)
            line = ft.readline()
        flag_th = 0
        for index in range(1, len((throughputs))):
            if throughputs[index] - throughputs[index - 1] < 0.12 * throughputs[0] / 5.4:
                top_choice = (index + 3) * 6
                flag_th = 1
                break
            elif throughputs[index] - throughputs[index - 1] < 0.24 * throughputs[0] / 5.4:
                top_choice = (index + 3) * 6 + 1
                flag_th = 1
                break
            elif throughputs[index] - throughputs[index - 1] < 0.36 * throughputs[0] / 5.4:
                top_choice = (index + 3) * 6 + 2
                flag_th = 1
                break
            elif throughputs[index] - throughputs[index - 1] < 0.48 * throughputs[0] / 5.4:
                top_choice = (index + 3) * 6 + 3
                flag_th = 1
                break
            elif throughputs[index] - throughputs[index - 1] < 0.60 * throughputs[0] / 5.4:
                top_choice = (index + 3) * 6 + 4
                flag_th = 1
                break
            elif throughputs[index] - throughputs[index - 1] < 0.72 * throughputs[0] / 5.4:
                top_choice = (index + 3) * 6 + 5
                flag_th = 1
                break
            else:
                continue
        if flag_th == 0:
            top_choice = 60
        # top_choice = float(top_choice)
        mapping[groupid].append(top_choice)

        line = ff.readline()
        while line:
            if "choices" in line:
                seq_start = line.find("[") + 1
                seq_end = line.find("]")
                choices = line[seq_start:seq_end].split(", ")
                choices = [int(item[1]) for item in choices[:-1]] + [int(choices[-1][1])]
                if len(choices) == 32:
                    choices = choices[:22] + choices[24:]
                mapping[groupid].append(choices)
                # print(choices)
            elif "configs" in line:
                seq_start = line.find("[") + 1
                seq_end = line.find("]")
                configs = line[seq_start:seq_end].split(", ")
                # print(configs)
                configs = [int(item[1:-1]) for item in configs]
                mapping[groupid].append(configs)
                # print(configs)
                break
            line = ff.readline()
    datalist = []
    features = []
    tags = []
    testlist = []
    features_test = []
    tags_test = []
    for key in mapping:
        if key <= 880:
            # print(key, mapping[key])
            datalist.append(tuple(mapping[key]))
        else:
            # print(key, mapping[key])
            testlist.append(tuple(mapping[key]))
    # print(testlist)
    if "data" in file_name:
        random.seed(42)
        random.shuffle(datalist)
    for item in datalist:
        if len(item) == 3:
            features.append(item[1] + item[2])
            tags.append(item[0])
    for item in testlist:
        if len(item) == 3:
            features_test.append(item[1] + item[2])
            tags_test.append(item[0])
    features += features_test
    tags += tags_test
    for index in range(len(features)):
        iscompute = features[index][10:20]
        isaccelerate = features[index][20:30]
        isimem = features[index][30:35]
        cfcompute = features[index][37 + 5 : 47 + 5]
        cfaccelerate = features[index][47 + 5 : 57 + 5]
        cfimem = features[index][57 + 5 : 62 + 5]
        # print(isimem, cfimem)
        numcompute = float(sum(iscompute)) + 1
        numaccelerate = float(sum(isaccelerate)) + 1
        numimem = float(sum(isimem)) + 1
        sizecompute = float(sum([a * b for a, b in zip(iscompute, cfcompute)])) + 1
        sizeaccelerate = float(sum([a * b for a, b in zip(isaccelerate, cfaccelerate)])) + 1
        sizeimem = float(sum([a * b for a, b in zip(isimem, cfimem)]))
        if features[index][0] == 0:
            ipfilter1 = 0
            isipfilter1 = 0
        else:
            ipfilter1 = features[index][30 + 5]
            isipfilter1 = 1
        if features[index][1] == 0:
            ipfilter2 = 0
            isipfilter2 = 0
        else:
            ipfilter2 = features[index][31 + 5]
            isipfilter2 = 1
        if features[index][2] == 0:
            iprewriter = 0
            isiprewriter = 0
        else:
            iprewriter = features[index][34 + 5]
            isiprewriter = 1
        if features[index][3] == 0:
            avcounter = 0
        else:
            avcounter = 1
        if features[index][4] == 0:
            counter = 0
        else:
            counter = 1
        if features[index][5] == 0:
            ftpportmapper1 = 0
            ftpportmapper2 = 0
            isftpportmapper1 = 0
            isftpportmapper2 = 0
        else:
            ftpportmapper1 = features[index][32 + 5]
            ftpportmapper2 = features[index][33 + 5]
            isftpportmapper1 = 1
            isftpportmapper2 = 1
        if features[index][6] == 0:
            webgen = 0
            iswebgen = 0
        else:
            webgen = features[index][35 + 5] + features[index][35 + 5]
            iswebgen = 1
        if features[index][7] == 0:
            udpipencap = 0
        else:
            udpipencap = 1
        if features[index][8] == 0:
            demux = 0
            isdemux = 0
        else:
            demux = features[index][36 + 5]
            isdemux = 1
        if features[index][9] == 0:
            conn = 0
            isconn = 0
        else:
            conn = features[index][36 + 5]
            isconn = 1
        imem_feature = [2.0 * ipfilter1, 2.0 * ipfilter2, 2 * avcounter, 2 * counter, 1 * udpipencap]
        emem_bool = [
            2.0 * isiprewriter,
            2.0 * isftpportmapper1,
            2.0 * isftpportmapper2,
            1.0 * iswebgen,
            2.0 * isdemux,
            2.0 * isconn,
        ]
        compute_bool = [
            25 * sizeaccelerate,
            10 * sizecompute,
            60 * ipfilter1,
            60 * ipfilter2,
            35 * isiprewriter + 20,
            35 * isftpportmapper1 + 20,
            35 * isftpportmapper2 + 20,
            12 * isconn,
            18 * isdemux,
            40 * avcounter,
            20 * counter,
            60 * udpipencap,
            16 * iswebgen + 50,
        ]
        compute_logic = [25 * sizeaccelerate, 10 * sizecompute]

        features[index] = []

        # Compute intensity of non-stareftul data structure elements/stateful data structure elements.
        features[index] += [sum(compute_logic) / (sum(compute_bool) - sum(compute_logic))]
        # IMEM access
        features[index] += [sum(imem_feature)]
        # EMEM access
        features[index] += [sum(emem_bool)]
        # Compute intensity
        features[index] += [sum(compute_bool)]
        # IMEM-Compute ratio
        features[index] += [(sum(imem_feature) / sum(compute_bool))]
        # EMEM-Compute ratio
        features[index] += [(sum(emem_bool) / sum(compute_bool))]
        # IMEM ratio
        features[index] += [(sum(imem_feature)) / (120 * sum(emem_bool) + sum(compute_bool) + 55 * sum(imem_feature))]
        # EMEM ratio
        features[index] += [(sum(emem_bool)) / (120 * sum(emem_bool) + sum(compute_bool) + 55 * sum(imem_feature))]
        # MEMSum-Compute ratio
        features[index] += [(1.0 * sum(imem_feature) + 2.1 * sum(emem_bool)) / sum(compute_bool)]
        # MEMRes-Compute ratio
        features[index] += [(1.0 * sum(imem_feature) - 2.1 * sum(emem_bool)) / sum(compute_bool)]
    print(len(features), len(tags))
    return features, tags


if __name__ == "__main__":

    file_name = str(args.dataset)
    features, tags = feature_extraction(file_name, 880)
    features_train = np.array(features[:680])
    tags_train = np.array(tags[:680])
    features_test = np.array(features[680:])
    tags_test = np.array(tags[680:])
    with open("dataset.pickle", "wb") as f:
        pickle.dump([features_train, tags_train, features_test, tags_test], f)
    file_name = str(args.testset)
    features, tags = feature_extraction(file_name, 4)
    features_nf = np.array(features)
    tags_nf = np.array(tags)
    with open("testset.pickle", "wb") as f:
        pickle.dump([features_nf, tags_nf], f)

