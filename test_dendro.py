#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 07:12:18 2022

@author: jrwill
"""


# from OccMat_Dendro import DrawOccMat

import OccMat_Dendro as omd

file = r'occ_bi_forCluster.csv'

occmat,row_labels, col_labels = omd.ReadCSVmatrix(file)



occmat_ordered, rows_ordered, cols_ordered = omd.DrawOccMat(occmat = occmat, row_labels = row_labels, col_labels = col_labels, font = 'Arial', row_label_size = 0.9, col_label_size = 0.9, border_color = 'w', save_ordered_csv=True)

print("occmat",occmat_ordered)
print("rows_ordered", rows_ordered)
print("cols_ordered", cols_ordered)




