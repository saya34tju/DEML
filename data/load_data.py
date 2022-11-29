import pandas as pd
import numpy as np
import os
import time
np.set_printoptions(suppress=True)



extract_file="BigDeepSynergy/data0704/drugdrug_extract28new.csv"
drugfeature_file="BigDeepSynergy/data0704/drugfeature.csv"
cell_line_feature="BigDeepSynergy/data0704/cell-line-feature_express_extract.csv"
# print(os.listdir('BigDeepSynergy/data0704'))
# print(os.getcwd())
def load_data(cell_line_name="all",score="S",is_class=True):
    sta_time=time.time()
    # extract=pd.read_csv(extract_file,usecols=[4,5,6,7,8,9,10,11,13])
    extract = pd.read_csv(extract_file)
    # hh
    drug_feature=pd.read_csv(drugfeature_file)
    cell_feature=pd.read_csv(cell_line_feature)
    drug_comb=extract
    
    n_sample=drug_comb.shape[0]
    n_feature=(drug_feature.shape[1]-1)*2+1+cell_feature.shape[0]+4+1
    drug_comb.index=range(n_sample)
    # n_sample=10000
    data=np.zeros((n_sample,n_feature))
    for i in range(n_sample):
        drugA_id=drug_comb.at[i,"drug_row_cid"]
        drugB_id=drug_comb.at[i,"drug_col_cid"]
        drugA_feature=get_drug_feature(drug_feature,drugA_id)
        drugB_feature=get_drug_feature(drug_feature,drugB_id)
        cell_line_name=drug_comb.at[i,"cell_line_name"]
        feature=get_cell_feature(cell_feature,cell_line_name)
        chr_array = ['synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'S','label']
        gate_array=[(6.8,-8),(8.4,-7.4),(4.6,-33),(7.8,-7.5),(23.3,-14)]
        pos_gate=[i[0] for i in gate_array]
        neg_gate=[i[-1] for i in gate_array]
        label=[drug_comb.at[i,name] for name in chr_array]
        if label[0]>pos_gate[0] or label[1]>pos_gate[1] or label[2]>pos_gate[2] or label[4]>pos_gate[4]:
            label[-1]=1
        elif label[0]<neg_gate[0] or label[1]<neg_gate[1] or label[2]<neg_gate[2]  or label[4]<neg_gate[4]:
            label[-1]=0
        else:
            label[-1]=-1


        # if label[-1]>=0:
        #     label[-1]=1 if label[-1]==1 or label[-1]==3 else 0
        # if label[2]>5:
        #     label[-1]=1
        # elif label[2]<-33:
        #     label[-1]=0
        # else:label[-1]=-1



        # label=drug_comb.at[i,score]
        # if is_class:
        #     if label>=1:
        #         label=1
        #     elif label<-19:
        #         label=0
        #     else:
        #         label=-1
        sample=np.hstack((drugA_feature,drugB_feature,feature,label))
        data[i]=sample
        if i%10000==0:
            print(i)
        # break
            print('this costs {:.2f} minutes'.format((time.time()-sta_time)/60))
    return data[:,0:-6],data[:,-6::]




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

# load_data()
# load_unknown_data()
