# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:41:57 2022

@author: Xiyu
"""
#%% occ_norm
import numpy as np
import mrcfile
import os
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import seaborn as sns
#%%
matrix = np.load(r"D:\Scripps\Python tutorial\occ_origin_matrix.npy\occ_origin_matrix.npy")
#%% save density
def save_density(data, grid_spacing, outfilename, origin=None):
    """
    Save the density to an mrc file. The origin of the grid will be (0,0,0)
    • outfilename: the mrc file name for the output
    """
    print("Saving mrc file ...")
    data = data.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = grid_spacing
        if origin is not None:
            mrc.header['origin']['x'] = origin[0]
            mrc.header['origin']['y'] = origin[1]
            mrc.header['origin']['z'] = origin[2]
        mrc.update_header_from_data()
        mrc.update_header_stats()
    print("done")

# for j in ["L17","L19","L28"]: print(len([i for i in os.listdir(j) if i.split(".")[1] =="mrc"]))
# 42 30 33
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

#%% occupancy analysis
def sns_occ_cluster(test_map, seg, metric = "euclidean", norm = False, norm_col = 0):
    if norm:
        occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
        occ = occ/occ[:, norm_col].reshape(-1,1)
        cl=sns.clustermap(occ, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    else:
        occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
        cl=sns.clustermap(occ, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    return(cl)

#%%
def sns_occ_cluster_label(test_map, seg, metric = "euclidean", norm = False, norm_col = 0, transpose  = False, give_name = False, test_map_name = 0, seg_name = 0):
    
    occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
    if norm:
        occ = occ/occ[:, norm_col].reshape(-1,1)
    col_name = seg_name
    row_name = test_map_name
    
    if transpose:
        occ = occ.T
        col_name = test_map_name
        row_name = seg_name
    
    if give_name:
        df = pd.DataFrame(occ, columns= col_name, index = row_name)
        cl = sns.clustermap(df, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    else:    
        cl = sns.clustermap(occ, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    
    return occ, cl, df

#%% no cluster
def sns_occ_label(test_map, seg, metric = "euclidean", norm = False, norm_col = 0, transpose  = False, give_name = False, test_map_name = 0, seg_name = 0):
    
    occ = np.matmul(test_map, seg.T)/seg.sum(1).reshape(1,-1)
    if norm:
        occ = occ/occ[:, norm_col].reshape(-1,1)
    col_name = seg_name
    row_name = test_map_name
    
    if transpose:
        occ = occ.T
        col_name = test_map_name
        row_name = seg_name
    
    if give_name:
        df = pd.DataFrame(occ, columns= col_name, index = row_name)
        cl = sns.clustermap(df, row_cluster = False, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    else:    
        cl = sns.clustermap(occ, xticklabels=True, yticklabels=True, cmap = "rocket_r", metric = metric)
    
    return occ, cl, df
#%%
matrix_name = np.array(['h24', 'h18', 'uL14', 'uL15', 'uL29', 'h19', 'h25', 'h31', 'h27',
       'h33', 'h8', 'uL16', 'h9', 'h32', 'h26', 'h22', 'h36', 'h109',
       'uL13', 'h108', 'h37', 'h23', 'h35', 'h21', 'uL11', 'uL10', 'h20',
       'h34', 'h90', 'h84', 'h53', 'h47', 'bL27', 'bL33', 'bL32', 'h46',
       'h52', 'h85', 'h91', 'h87', 'h93', 'h44', 'h50', 'h78', 'bL25',
       'bL19', 'h79', 'h51', 'h45', 'h92', 'h86', 'h82', 'h96', 'h69',
       'h41', 'h55', 'bL35', 'bL21', 'bL20', 'bL34', 'h54', 'h40', 'h68',
       'h97', 'h83', 'h95', 'h81', 'h56', 'h42', 'bL36', 'h43', 'h57',
       'h80', 'h94', 'h99', 'h72', 'h66', 'uL4', 'bL9', 'uL5', 'h67',
       'h73', 'h98', 'h65', 'h71', 'h59', 'uL6', 'h58', 'h70', 'h64',
       'h48', 'h60', 'h74', 'bL28', 'uL2', 'uL3', 'h75', 'h61', 'h49',
       'h88', 'h77', 'h63', 'bL17', 'h62', 'h76', 'h89', 'h11', 'h39',
       'h2', 'h106', 'h112', 'h107', 'h3', 'h38', 'h10', 'h12', 'h1',
       'h111', 'h105', 'uL22', 'uL23', 'h104', 'h110', 'h13', 'h4',
       'h100', 'h101', 'h5', 'h16', 'h28', 'h14', 'h7', 'uL30', 'uL24',
       'h103', 'uL18', 'h102', 'h6', 'h29'])

#%%

radius = np.zeros((160,160,160))
for i in range(160):
    for j in range(160):
        for k in range(160):
            radius[i,j,k] = np.sqrt((i-79.5)**2 + (j-79.5)**2 + (k-79.5)**2)
radius = radius.reshape(-1)  

#%%
def shell_std(test_map, radius, bins= 2 , rmax = 70):
    bin_num = len(list(range(0, rmax)[0: rmax: bins])) 
    std = np.zeros((test_map.shape[0], bin_num))
    
    for t, shell in enumerate(list(range(0, 70))[0:70:bins]):
        mask = np.logical_and(radius >= shell, radius < shell + bins)
        for i in range(test_map.shape[0]): 
            std[i, t] = np.std(test_map[i, mask == 1])
    return std
#%%

def shell_mean(test_map, radius, bins= 2 , rmax = 70):
    bin_num = len(list(range(0, rmax)[0: rmax: bins])) 
    m = np.zeros((test_map.shape[0], bin_num))
    
    for t, shell in enumerate(list(range(0, 70))[0:70:bins]):
        mask = np.logical_and(radius >= shell, radius < shell + bins)
        for i in range(test_map.shape[0]): 
            m[i, t] = np.mean(test_map[i, mask == 1])
    return m

#%%
def count_zero(test_map, radius, bins= 2 , rmax = 70):
    bin_num = len(list(range(0, rmax)[0: rmax: bins])) 
    zero_num = np.zeros((test_map.shape[0], bin_num))
    
    for t, shell in enumerate(list(range(0, 70))[0:70:bins]):
        mask = np.logical_and(radius >= shell, radius < shell + bins)
        mask_len = np.sum(mask*1)
        for i in range(test_map.shape[0]): 
            z_temp = (test_map[i, mask == 1] == 0)*1
            zero_num[i, t] = z_temp.sum()/mask_len
    return zero_num

#%%
def get_3sigma(std, cr, m, threshold):
    gt_len = std.shape[0]
    bin_num = len(std[0])
    gt = np.zeros((gt_len,2))
    for i in range(gt_len):
        for j in list(range(bin_num))[::-1]:
            if cr[i][j] < threshold: break
        gt[i, 0] = m[i][j]
        gt[i, 1] = std[i][j]
    return gt
#%% OCC of CryoSPARC (threshold = 1) No cluster on Row, norm to h2
#np.where(np.array(matrix_name) == "h2")
cryosparc_name, cryosparc_mrc = load_mrc(r"D:\Scripps\Williamson\iSAT\Structrues\FrealignandCryoSPARC_after_hierarchical\pathway_maps_with_label", 160)
cryosparc_bi = cryosparc_mrc > 1
cryosparc_occ, cryosparc_cl, occ_norm = sns_occ_label(cryosparc_bi, matrix, norm = True, norm_col = 108, give_name = True, test_map_name = cryosparc_name, seg_name = matrix_name)

#%%pass_seg

#%%dead

#%%
plt.rcParams.update({'font.size': 12})
padding = 30
img_size = 160-(padding*2)

#occ_norm = occ_norm.values

for i in range(len(matrix_name)):
#for i in [0]:
    temp = np.zeros((img_size*13,img_size))
    for j,k in enumerate(occ_norm[:,i].argsort()[::-1]):
        temp[j*img_size:(j+1)*img_size] = (matrix[i] * cryosparc_mrc[k]>1).reshape((160,160,160)).sum(2)[padding:-padding,padding:-padding]
    occ_temp = occ_norm[:,i][occ_norm[:,i].argsort()[::-1]]
    plt.figure(figsize = (16,2))
    plt.imshow(temp.T,cmap="Blues")
    for j,k in enumerate(occ_temp):
        plt.annotate("%.2f"%k, (img_size*(j+0.3),img_size/2), color = "grey")
    plt.axis("off")
    plt.savefig(r"D:\Scripps\Williamson\iSAT\Paper\Seg_projection\%s.png"%matrix_name[i].split(".")[0])
    plt.close()

#%%
plt.rcParams.update({'font.size': 12})
padding = 30
img_size = 160-(padding*2)

temp_all = np.zeros((img_size*13,1))

for i in range(len(pass_seg_name)):
#for i in [5]:
    
    for j,k in enumerate(occ_norm[:,i].argsort()[::-1]):
        temp[j*img_size:(j+1)*img_size] = (matrix[i] * cryosparc_mrc[k]>1).reshape((160,160,160)).sum(2)[padding:-padding,padding:-padding].T
    occ_temp = occ_norm[:,i][occ_norm[:,i].argsort()[::-1]]
    plt.figure(figsize = (1,16))
    plt.imshow(temp,cmap="Blues")
    for j,k in enumerate(occ_temp):
        plt.annotate("%.2f"%k, (img_size/2,img_size*(j+0.3)), color = "grey")
    plt.axis("off")
    plt.savefig("/Users/shengkai/Desktop/paper/deaD/illustrator/block_occupancy/%s_T.png"%pass_seg_name[i].split(".")[0])
    plt.close()