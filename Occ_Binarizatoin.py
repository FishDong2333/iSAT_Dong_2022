# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:13:23 2022

@author: Xiyu
"""

import numpy as np
import mrcfile
import pandas as pd
import seaborn as sns
import networkx as nx
import pickle
from networkx.algorithms.components.connected import connected_components
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
#%%
def sns_occ_cluster(test_map, seg, metric = "euclidean", norm = False, norm_col = 0, transpose  = False):
    if not transpose:
        if norm:
            occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
            occ = occ/occ[:, norm_col].reshape(-1,1)
            sns.clustermap(occ, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
        else:
            occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
            sns.clustermap(occ, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    else:
        if norm:
            occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
            occ = occ/occ[:, norm_col].reshape(-1,1)
            sns.clustermap(occ.T, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
        else:
            occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
            sns.clustermap(occ.T, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    
    return occ
#%%
def qua_analysis(seg, seg_name, test_map, dp, save_pic = True, sigma_num=1, norm_seg = 0, threshold=0.5, step = 3):
    seg_num = seg.shape[0]
    lower_list = np.zeros(seg_num)
    upper_list = np.zeros(seg_num)
    s = sns_occ_cluster(test_map, seg, transpose = True, norm = True, norm_col = norm_seg)
    for i in range(seg_num):
        s_temp = s[:,i][s[:,i].argsort()]
        
        y_temp = s_temp[step:]-s_temp[:-step]
        
        y = y_temp.argmax()
        
        upper = s_temp[y:].mean() - sigma_num*s_temp[y:].std()
        print(y,s_temp[y:].mean(),  sigma_num*s_temp[y:].std() )
        if y == 0 or y == 1:
            lower = upper/2
        else:
            lower = s_temp[:y].mean() + sigma_num*s_temp[:y].std()
        
        if lower > upper:
            lower = (lower+upper)/2
            upper = lower
        if upper <0.4:
            upper = s_temp.max()/2
            lower = upper/2
        
        if sum(s_temp>threshold) == len(s_temp):
            lower = threshold
            upper = threshold
            
            
            
        lower_list[i] = lower
        upper_list[i] = upper
        
        
        if save_pic:
            plt.figure()
            plt.plot(s_temp)
            plt.title(seg_name[i])
            plt.plot(y_temp)
            plt.hlines([upper], 0 , test_map.shape[0], colors = "red")
            plt.hlines([lower], 0 , test_map.shape[0], colors = "blue")
            plt.savefig(dp + "/" + seg_name[i].split(".")[0])
            plt.close()
        
    s_upper = s>=upper_list
    s_lower = s<lower_list

    columns = ["i", "j", "0_0", "0_1", "1_0", "1_1"]
    
    qua = np.zeros((int(seg_num*(seg_num-1)/2),6))

    count = 0
    for i in range(seg_num-1):
        for j in range(i+1, seg_num):
            qua[count] = solve_uull(s_upper[:,i],s_upper[:,j],s_lower[:,i],s_lower[:,j], i,j)
            count += 1
    dependency = qua[np.logical_and(qua[:,3]==0, qua[:,4]!=0)][:,0:2] 
    dependency = np.vstack((dependency,qua[np.logical_and(qua[:,4]==0, qua[:,3]!=0)][:,0:2][:,::-1]))
    cor = qua[np.logical_and(qua[:,3]==0, qua[:,4]==0)][:,0:2]
    
    mass = np.zeros(seg_num)
    for i in range(seg_num):
        mass[i] = np.max(s[s_upper[:,0], i]) * seg[i].sum()
        
    return (s, upper_list, lower_list, qua, dependency, cor, mass)

#%%
def solve_qua(d, i, j):
    a = [[2,3],[4,5]]
    temp = [i,j,0,0,0,0]
    for i, t in enumerate(d[0]):
        temp[a[t[0]][t[1]]] = d[1][i] 
    return temp

def solve_uull(u1, u2, l1, l2, i, j):
    t00 = np.logical_and(l1, l2).sum()
    t01 = np.logical_and(l1, u2).sum()
    t10 = np.logical_and(u1, l2).sum()
    t11 = np.logical_and(u1, u2).sum()
    return((i,j,t00,t01,t10,t11))

#%% load mrc file
def load_mrc(dp, box):
    if dp[-1]!= "/": dp = dp + "/"
    name_list = [i for i in os.listdir(dp) if i.split(".")[-1]=="mrc"]
    name_list.sort()
    
    num = len(name_list)
    temp = np.zeros((num, box**3))
    for i, name in enumerate(name_list):
        temp[i] = mrcfile.open(dp + name).data.reshape(-1)
    return (name_list, temp)

#%%
Big_name, Big_mrc = load_mrc("BigClasses/", 160)
Big_bi = Big_mrc > 1
s, upper_list, lower_list, q, d, c, mass = qua_analysis(matrix, matrix_name, Big_bi, 'seg_sort/', norm_seg = 108)
df = pd.DataFrame({"seg": matrix_name, "threshold": upper_list}) 
df.to_csv("seg_sort/seg_sort.csv")
occ_mat = pd.read_csv('PathwayMaps14_occ_mat.csv')
occ_mat = pd.DataFrame.to_numpy(occ_mat)
occ_bi = np.zeros(occ_mat.shape)
for i,occ in enumerate(occ_mat):
    for j, value in enumerate(occ):
        if occ[j] > upper_list[i]:
            occ_bi[i][j] = 1
        elif occ[j] <= lower_list[i]:
            occ_bi[i][j] = 0
        else:
            occ_bi[i][j] = 0.5
pathway_maps_name, pathway_maps_mrc = load_mrc("pathway_maps_with_label/", 160)
df_occ_bi = pd.DataFrame(data=occ_bi,index=[matrix_name],columns=[pathway_maps_name])
df_occ_bi.to_csv("occ_bi_0p5.csv")