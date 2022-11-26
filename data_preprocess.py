import pandas as pd
import numpy as np
import os
import time
import itertools
from collections import defaultdict
from operator import neg
import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import Data, Batch
from rdkit import Chem
np.set_printoptions(suppress=True)
print('ok')
extract_file = "../BigDeepSynergy/data0704/drugdrug_extract28newddi.csv"
drugfeature_file = "../BigDeepSynergy/data0704/drugfeature.csv"
cell_line_feature = "../BigDeepSynergy/data0704/cell-line-feature_express_extract.csv"

# drug a shape 541 / drug b shape 541 / gene shape 927
# precess
# print(os.listdir('BigDeepSynergy/data0704'))
# print(os.getcwd())
# drugdrug_extract28newddi需要新加inter与type两个字段
def load_data(cell_line_name="all", score="S", is_class=True,cuts=None):
    sta_time = time.time()
    extract = pd.read_csv(extract_file)
    drug_feature = pd.read_csv(drugfeature_file)
    cell_feature = pd.read_csv(cell_line_feature)
    drug_comb = extract

    n_sample = drug_comb.shape[0]
    n_feature = (drug_feature.shape[1] - 1) * 2 + 1 + cell_feature.shape[0] + 4 + 1+1
    drug_comb.index = range(n_sample)
    if cuts:
        n_sample=cuts
    data = np.zeros((n_sample, n_feature))
    for i in range(n_sample):
        drugA_id = drug_comb.at[i, "drug_row_cid"]
        drugB_id = drug_comb.at[i, "drug_col_cid"]
        drugA_feature = get_drug_feature(drug_feature, drugA_id)
        drugB_feature = get_drug_feature(drug_feature, drugB_id)
        cell_line_name = drug_comb.at[i, "cell_line_name"]
        feature = get_cell_feature(cell_feature, cell_line_name)
        chr_array = ['synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'S', 'label','inter']
        label = [drug_comb.at[i, name] for name in chr_array]
        if label[2]>5:label[-2]=1
        elif label[2]<-5:label[-2]=0
        else:label[-2]=0
        sample = np.hstack((drugA_feature, drugB_feature, feature, label))
        data[i] = sample
        if i % 10000 == 0:
            print(i)
            print('this costs {:.2f} minutes'.format((time.time() - sta_time) / 60))
    return data[:, 0:-7], data[:, -7::]


def get_drug_feature(feature, drug_id):
    drug_feature = feature.loc[feature["cid"] == drug_id]
    drug_feature = np.array(drug_feature)
    drug_feature = drug_feature.reshape(drug_feature.shape[1])[1:]
    return drug_feature


def get_cell_feature(feature, cell_line_name):
    cell_feature = feature[str(cell_line_name)]
    cell_feature = np.array(cell_feature)
    return cell_feature






