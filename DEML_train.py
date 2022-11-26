import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader,Dataset,TensorDataset
import numpy as np
import pandas as pd
from torch import nn, optim
import time
from scipy.stats import spearmanr, pearsonr
from torch.nn.modules.container import ModuleList
from sklearn.metrics import mean_squared_error,r2_score,roc_curve,roc_auc_score,precision_recall_curve, auc,confusion_matrix,accuracy_score
import math
import copy
import data_preprocess as dp
import util_tools
import models
# info: gate type 2 / label[0:7] / expert=5 / 2000 4000 2000 / 1e4 / loss_weight 1 10 10

sta_time=time.time()
data,label=dp.load_data()
data=util_tools.normalize(data)[0]
label=label[:,0:7]
length_data=len(data)
train_ind=np.random.choice(range(length_data),int(length_data*0.8),replace=False)
valid_ind=np.random.choice(train_ind,int(length_data*0.2),replace=False)
test_ind=np.array([i for i in range(length_data) if i not in train_ind])

lr=1e-4
criterion_r = nn.MSELoss()
criterion_c=util_tools.LabelSmoothingCrossEntropy()
archi=[2048,4096,2048]
epoch=2000
cuda_id=0
# model=models.DeepSynergy(len(data[0]),archi,2)

model=models.MMOE(input_size=len(data[0]),num_experts=5,experts_out=2000,experts_hidden=4000,towers_hidden=2000,towers_out=2,tasks=7,pre_type=False)
train_data=TensorDataset(torch.Tensor(data[train_ind]),torch.Tensor(label[train_ind]))
train_loader=DataLoader(
    train_data,
    batch_size=128,
    shuffle=True
)
data_test,label_test=torch.Tensor(data[test_ind]),torch.Tensor(label[test_ind])
optimizer = optim.Adam(model.parameters(), lr=lr)
model=model.cuda(cuda_id)
df=[]
for num_epoch in range(epoch):
    acc_epoch_1 = 0
    acc_epoch_2=0
    sum_loss = 0
    for data, label in train_loader:
        data = data.cuda(cuda_id)
        model = model.cuda(cuda_id)
        outreg = model(data)
        out_r=torch.stack([outreg[i][:,-1] for i in range(5)],axis=1).squeeze(1)
        pre=[torch.argmax(torch.softmax(outs,dim=-1),dim=-1) for outs in outreg[5::]]
        label = label.cuda(cuda_id)
        acc_epoch_1+=(label[:,5]==pre[0]).sum().item()/len(label)
        acc_epoch_2+=(label[:,6]==pre[1]).sum().item()/len(label)
        loss=criterion_r(out_r,label[:,0:5])+criterion_c(outreg[5],label[:,5])+criterion_c(outreg[6],label[:,6])
        print_loss = loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += print_loss
    # print()
    with torch.no_grad():
        model.eval()
        model=model.cpu()
        outreg=[outs.detach() for outs in model(data_test)]
        out_r=torch.stack([outreg[i][:,-1] for i in range(5)],axis=1).squeeze(1)
        loss=criterion_r(out_r,label_test[:,0:5])+10*criterion_c(outreg[5],label_test[:,5])+10*criterion_c(outreg[6],label_test[:,6])
        score_reg=[list(util_tools.reg_score(label_test[:,i],out_r[:,i])) for i in range(5)]
        score_1=util_tools.cal_score(label_test[:,5],torch.argmax(outreg[0],dim=-1),torch.softmax(outreg[0],dim=-1)[:,-1])
        score_2=util_tools.cal_score(label_test[:,6],torch.argmax(outreg[1],dim=-1),torch.softmax(outreg[1],dim=-1)[:,-1])
        score=score_reg+[[score[i] for i in [0,1,2,7,8,9]] for score in [score_1,score_2]]
        df.append(list(score))
    if num_epoch%10==0:
        print(num_epoch,sum_loss,acc_epoch_1/len(train_loader),acc_epoch_2/len(train_loader),'test',score)
df=pd.DataFrame(df)
df.to_csv('synergy_log_ddi.csv')
print((time.time()-sta_time)/3600)




