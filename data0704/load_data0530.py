import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)



extract_file="BigDeepSynergy/data0704/drugdrug_extract.csv"
drugfeature_file="BigDeepSynergy/data0704/drugfeature.csv"
cell_line_feature="BigDeepSynergy/data0704/cell-line-feature_express_extract.csv"


def load_data(cell_line_name="all",score="S",is_class=True):

    extract=pd.read_csv(extract_file,usecols=[2,3,4,5,6])
    drug_feature=pd.read_csv(drugfeature_file)
    cell_feature=pd.read_csv(cell_line_feature)
    drug_comb=extract
    
    n_sample=drug_comb.shape[0]
    # n_sample = 10000
    n_feature=(drug_feature.shape[1]-1)*2+1+cell_feature.shape[0]
    drug_comb.index=range(n_sample)
    data=np.zeros((n_sample,n_feature))
    for i in range(n_sample):
        drugA_id=drug_comb.at[i,"drug_row_cid"]
        drugB_id=drug_comb.at[i,"drug_col_cid"]
        drugA_feature=get_drug_feature(drug_feature,drugA_id)
        drugB_feature=get_drug_feature(drug_feature,drugB_id)
        cell_line_name=drug_comb.at[i,"cell_line_name"]
        feature=get_cell_feature(cell_feature,cell_line_name)
        label=drug_comb.at[i,score]
        if is_class:
            if label>5:
                label=1
            elif label<-33:
                label=0
            else:
                label=-1
        sample=np.hstack((drugA_feature,drugB_feature,feature,label))
        data[i]=sample
        if i%10000==0:
            print(i)
        # break
    return data[:,0:-1],data[:,-1]




def get_drug_feature(feature,drug_id):
    drug_feature=feature.loc[feature["cid"]==drug_id]
    drug_feature=np.array(drug_feature)
    drug_feature=drug_feature.reshape(drug_feature.shape[1])[1:]
    return drug_feature


def get_cell_feature(feature,cell_line_name):
    # print(feature.head())
    # print(cell_line_name)
    cell_feature=feature[str(cell_line_name)]
    cell_feature=np.array(cell_feature)
    return cell_feature


# load_unknown_data()
