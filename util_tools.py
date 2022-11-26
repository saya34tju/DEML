import time
import pandas as pd
import numpy as np
import sys
import copy
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch import nn, optim
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error,r2_score,roc_curve,roc_auc_score,precision_recall_curve, \
    auc,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import math


def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df

def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1!=0
    X = X[:,feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X-means1)/std1[feat_filt]
    if norm == 'norm':
        return(X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X-means2)/std2
        X[:,std2==0]=0
        return(X, means1, std1, means2, std2, feat_filt)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.scores=[]
        self.models=[]
        # self.dic_test_score={
        #     'acc':[],
        #     'auc':[],
        #     'aupr':[],
        #     'tpr':[],
        #     'fpr':[],
        #     'precision':[],
        #     'recall':[],
        #     # 下面三个是一起的
        #     'Precision':[],
        #     'Recall':[],
        #     'f1':[],
        #     'cm':[]
        # }


    def __call__(self, val_loss, model,num,scores):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.scores=scores
            self.save_checkpoint(val_loss, model,num)
            # for i,j in enumerate(self.dic_test_score.items()):
            #     self.dic_test_score[j[0]]=scores[i]
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.scores=scores
            self.save_checkpoint(val_loss, model,num)
            self.counter = 0
            # for i,j in enumerate(self.dic_test_score.items()):
            #     self.dic_test_score[j[0]]=scores[i]

    def save_checkpoint(self, val_loss, model,num):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        # torch.save(model, 'finish_model'+str(num)+'.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss
        self.models=copy.deepcopy(model)

def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target.long().reshape(-1), reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

def reg_score(y_true,y_pred):
    y_pred=np.array(y_pred.detach().cpu())
    y_true=np.array(y_true.detach().cpu())
    mse=mean_squared_error(y_true,y_pred)
    r2=r2_score(y_true,y_pred)
    pear=pearsonr(y_true,y_pred)
    spea=spearmanr(y_true,y_pred)
    rmse=pow(mse,0.5)
    return mse,rmse,float(pear[0]),float(spea[0]),r2

def cal_score(label,pre,prob):
    label,pre,prob=[i.detach().numpy() for i in (label,pre,prob)]
    aucc = roc_auc_score(label, prob)
    fpr, tpr, _ = roc_curve(label, prob)
    precision, recall, _ = precision_recall_curve(label, prob)
    acc = accuracy_score(label,pre)
    aupr = auc(recall, precision)
    cm = confusion_matrix(label, pre)
    Precision, Recall, f1 = precision_score(label,pre),recall_score(label,pre),f1_score(label,pre)
    return acc,aucc,aupr,tpr,fpr,precision,recall,Precision,Recall,f1,cm

def prepare_data_MM(data,norm='tanh_norm',logger=True):
    data_drug1=data[:,0:541]
    data_drug2=data[:,541:2*541]
    data_drug=np.concatenate([data_drug1,data_drug2])
    data_cell=data[:,2*541::]
    data_drug,data_cell=[normalize(d,norm=norm)[0] for d in [data_drug,data_cell]]
    data_drug1,data_drug2=data_drug[0:len(data_drug1)],data_drug[len(data_drug1)::]
    if logger:
        print('one drug shape is '+str(data_drug1.shape[1]+data_cell.shape[1]))
        print('one drug shape is '+str(data_drug2.shape[1]+data_cell.shape[1]))
    return np.concatenate([data_drug1,data_cell,data_drug2,data_cell],axis=1),\
           np.concatenate([data_drug2,data_cell,data_drug1,data_cell],axis=1)

graph_data_file='28wdc_ddi_train.csv'
def prepare_data(data,label,norm='tanh_norm',logger=True):
    df_dds=pd.read_csv('28wdc_ddi_train.csv')[0:len(data)]
    data_graph = [(h, t, r) for h, t, r in
               zip(df_dds['d1'], df_dds['d2'], df_dds['type'])
               ]
    data_graph_r = [(t, h, r) for h, t, r in
               zip(df_dds['d1'], df_dds['d2'], df_dds['type'])
               ]
    data_drug1=data[:,0:541]
    data_drug2=data[:,541:2*541]
    data_drug=np.concatenate([data_drug1,data_drug2])
    data_cell=data[:,2*541::]
    data_drug,data_cell=[normalize(d,norm=norm)[0] for d in [data_drug,data_cell]]
    data_drug1,data_drug2=data_drug[0:len(data_drug1)],data_drug[len(data_drug1)::]
    data_MM,data_MM_r=prepare_data_MM(data,norm,logger)
    data_DS,data_DS_r=np.concatenate([data_drug1,data_drug2,data_cell],axis=1),\
            np.concatenate([data_drug2,data_drug1,data_cell],axis=1)
    data={
        'C':[data_cell,data_cell],
        'G':[data_graph,data_graph_r],
        'M':[data_MM,data_MM_r],
        'B': [data_MM, data_MM_r],
        'S':[data_DS,data_DS_r],
        'F':[[(data_graph[i],data_DS[i],data_MM[i],data_cell[i],label[i]) for i in range(len(data))],
             [(data_graph_r[i],data_DS_r[i],data_MM_r[i],data_cell[i],label[i]) for i in range(len(data))]]
    }
    data_input_dim={
        'C':len(data_cell[0]),
        'M':len(data_MM[0]),
        'F':len(data_MM[0]),
        'S':len(data_DS[0])
    }
    return data,label,data_input_dim

