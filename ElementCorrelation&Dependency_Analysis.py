# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:13:39 2022

@author: Xiyu
"""

import pandas as pd
import numpy as np
import pickle
from itertools import compress
import random
import copy
import networkx as nx
from networkx.algorithms.components.connected import connected_components
import matplotlib.pyplot as plt
#%%

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
    
#%%

def to_graph(l):
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current
        
def prune_DG(DG):
    ls_suc = []
    
    for i in list(DG.nodes):
        ls_suc += [list(DG.successors(i))]
    
    for t1, i in enumerate(DG.nodes):
        for t2, j in enumerate(DG.nodes):
            if j == i: continue
            if sum(np.isin(np.array(ls_suc[t1]+[i]), np.array(ls_suc[t2])))==len(ls_suc[t1])+1:
                for k in ls_suc[t1]:
                    ls_suc[t2].remove(k)
            print(ls_suc)
    DG_prune = nx.DiGraph()           
    for i, j in enumerate(ls_suc):
        if len(j)>0:
            for k in j:
                DG_prune.add_edge(list(DG.nodes)[i],k)
    return DG_prune

def subgraph(H):
    sink_nodes = [node for node in H.nodes if H.out_degree(node) == 0]
    source_nodes = [node for node in H.nodes if H.in_degree(node) == 0]
    sub_list = []
    for source in source_nodes:
        for sink in sink_nodes:
            path = nx.all_simple_paths(H, source, sink)
            nodes = []
            for p in path:
                nodes  = nodes + p
            nodes = set(nodes)
            sub_list += [H.subgraph(nodes)]
    return sub_list
       
#%%     
def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})

def draw_hierachical(H, root="core"):
    pos = hierarchy_pos(H, root)
    plt.figure()
    plt.xlim([-1,2])
    nx.draw(H,pos,with_labels = True, edge_color='black', width = 1,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))




def solve_uull(u1, u2, l1, l2, i, j):
    t00 = np.logical_and(l1, l2).sum()
    t01 = np.logical_and(l1, u2).sum()
    t10 = np.logical_and(u1, l2).sum()
    t11 = np.logical_and(u1, u2).sum()
    return((i,j,t00,t01,t10,t11))

#%%

def merge_connected(seg, seg_name, connected_list):
    seg_name = [i.split(".")[0] for i in seg_name]
    connected_list = [list(i) for i in connected_list]
    ind = list(range(seg.shape[0]))
    merge_list = []
    for i in connected_list:
        merge_list += i
    merge_list =  [int(i) for i in merge_list]
    retain_ind = np.delete(np.array(ind), merge_list)
    retain_seg = seg[retain_ind]
    retain_seg_name = np.array(seg_name)[retain_ind]
    
    merge_seg = np.zeros((len(connected_list), 160**3))
    merge_name = []
    for i, m in enumerate(connected_list):
        m = [int(n) for n in m]
        merge_seg[i] = seg[m].sum(0)
        merge_name.append("_m_".join(np.array(seg_name)[m]))
    return (np.vstack((retain_seg, merge_seg)), list(retain_seg_name) + merge_name)

def contact_graph(seg, PDB_seg, threshold, mass):
    seg_occ = sns_occ_cluster(seg, PDB_seg)
    contact_occ = seg_occ>threshold
    contact = np.zeros((int(seg.shape[0]*(seg.shape[0]-1)/2),4))
    count = 0
    for i in range(seg.shape[0]-1):
        for j in range(i+1,seg.shape[0]):
            temp_c = np.logical_and(contact_occ[i],contact_occ[j]).sum()
            if temp_c ==0:
                contact[count] = [i,j,0,0]
                count+=1
                continue
            mass_cal = 1/(1/mass[i] + 1/mass[j])
            contact[count] = [i,j,seg_occ[[i,j]][:, np.logical_and(contact_occ[i],contact_occ[j])].sum(), mass_cal]
            count+=1                      

    contact_weight = contact[contact[:,2]!=0]
    contact_weight[:,0:2] = contact_weight[:,0:2] 
    contact_weight[:,2]= np.sqrt((1/contact_weight[:,2])*contact_weight[:,3])
    return(contact_weight)

#%%


def cloud_seg(seg, box, cloud_size):
    seg_temp = np.zeros(box)
    seg_temp = seg_temp.reshape(-1)
    seg_temp[seg==1] = 1
    seg_temp = seg_temp.reshape(box)
    x, y, z = np.where(seg_temp == 1) 
    for i in range(len(x)):
        seg_temp[x[i]-cloud_size:x[i]+cloud_size+1, y[i]-cloud_size:y[i]+cloud_size+1, z[i]-cloud_size:z[i]+cloud_size+1] = 1
    return seg_temp.reshape(-1)

#%%

def contact_cloud(seg, box, cloud_size, mass, norm = True):
    seg_temp = np.zeros(seg.shape)
    for i in range(seg.shape[0]):
        seg_temp[i] = cloud_seg(seg[i], box, cloud_size)
  
    contact = np.zeros((int(seg.shape[0]*(seg.shape[0]-1)/2),4))
    count = 0
    for i in range(seg.shape[0]-1):
        for j in range(i+1,seg.shape[0]):
            temp_c = np.logical_and(seg_temp[i]==1,seg_temp[j]==1).sum()
            if temp_c ==0:
                contact[count] = [i,j,0,0]
                count+=1
                continue
            if norm == True:
                mass_cal = 1/(1/mass[i] + 1/mass[j])
            else:
                mass_cal = 1
            temp_c = mass_cal/temp_c
            contact[count] = [i,j, temp_c, mass_cal]
            count += 1         

    return seg_temp, contact

#%%

def contact_prune_DG(contact_G, DG):
    contact = set(contact_G.edges())
    contact = set.union(contact, (set(map(lambda x: x[::-1], contact_G.edges()))))
    DG_temp = nx.DiGraph()
    for i in DG.edges():
        if i in contact:
            DG_temp.add_edge(i[0],i[1], weight = contact_G.edges[i[0],i[1]]["weight"])
    return prune_DG(DG_temp)
    
#%%

def prune_DG2(DG):
    DG_temp = list(DG.edges())
    source = [node for node in DG if DG.in_degree(node)==0][0]
    for i in DG[source].keys():
        if DG.in_degree(i)>1:
            DG_temp.remove((source, i))
    print(DG_temp)
    xx = nx.DiGraph()
    xx.add_edges_from(DG_temp)
    return xx

#%%

def prune_DG3(DG):
    DG_temp = list(DG.edges())
    for i in DG.nodes():
        for j in DG.successors(i):
            for k in set(DG.successors(i)) & set(DG.successors(j)):
                DG_temp.remove((i,k))
    
    print(DG_temp)
    xx = nx.DiGraph()
    xx.add_edges_from(DG_temp)
    return xx

#%%

seg = np.load("occ_origin_matrix/occ_origin_matrix.npy")
#unmasked_map = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/unmasked_map_wo70S.npy")

#bin_name = np.load("/Users/shengkai/Desktop/data/PDBs_forAlignment_box160/bin_name.npy")
#bin_map = np.load("/Users/shengkai/Desktop/data/Biopaper_unmaskedmaps/bin_map_wo70S.npy")

map_namelist = np.array(['h24', 'h18', 'uL14', 'uL15', 'uL29', 'h19', 'h25', 'h31', 'h27',
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
seg_name = map_namelist
#%%

s = pd.read_csv('occ_bi_forCluster', index_col=0)
seg_num = 140

#%%

s_order = [np.where(np.array(seg_name) == i)[0] for i in s.index.values]
s_order[54] = [113]
s_order[55] = [113]
s_order = [i[0] for i in s_order]

#%%

s=s.T
s_upper = s.values>=0.6
s_lower = s.values<0.4

columns = ["i", "j", "0_0", "0_1", "1_0", "1_1"]
    
qua = np.zeros((int(seg_num*(seg_num-1)/2),6))
count = 0
for i in range(seg_num-1):
    for j in range(i+1, seg_num):
        qua[count] = solve_uull(s_upper[:,i],s_upper[:,j],s_lower[:,i],s_lower[:,j], i,j)
        count += 1
    dependency = qua[np.logical_and(qua[:,3]==0, qua[:,4]>0)][:,0:2] 
    dependency = np.vstack((dependency,qua[np.logical_and(qua[:,4]<=1, qua[:,3]>=1)][:,0:2][:,::-1]))
    cor = qua[np.logical_and(qua[:,3]==0, qua[:,4]==0)][:,0:2]

#%%
seg_num = 140
seg140 = seg[s_order]
seg140_name = s.columns.values
G = to_graph(cor)
#%%
new_seg, new_seg_name = merge_connected(seg140, seg140_name, list(connected_components(G)))

#%%
s_represent = [i.split("_")[0] for i in new_seg_name]

#%%
s_2 = s[s_represent]
seg_num = s_2.shape[1]
s_upper = s_2.values>=0.6
s_lower = s_2.values<0.4

columns = ["i", "j", "0_0", "0_1", "1_0", "1_1"]
    
qua = np.zeros((int(seg_num*(seg_num-1)/2),6))
count = 0
for i in range(seg_num-1):
    for j in range(i+1, seg_num):
        qua[count] = solve_uull(s_upper[:,i],s_upper[:,j],s_lower[:,i],s_lower[:,j], i,j)
        count += 1
    dependency = qua[np.logical_and(qua[:,3]==0, qua[:,4]>0)][:,0:2] 
    dependency = np.vstack((dependency,qua[np.logical_and(qua[:,4]<1, qua[:,3]>=1)][:,0:2][:,::-1]))
    cor = qua[np.logical_and(qua[:,3]==0, qua[:,4]==0)][:,0:2]

#%%
DG=nx.DiGraph()
DG.add_edges_from(dependency.astype(int))
DG_prune = prune_DG(DG)
plt.figure()
nx.draw_kamada_kawai(DG_prune, with_labels = True)
