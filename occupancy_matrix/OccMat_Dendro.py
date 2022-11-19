#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 07:10:50 2022

@author: jrwill
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
from scipy.cluster import hierarchy
import sys
import csv
import argparse
import copy

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
    
def PaintPixel(value,npixels = 40,value_hue = 0.62, border_col = 'w'):
    hsv = np.zeros((npixels,npixels,3))
    hsv[...,0] = value_hue # blue center
    hsv[...,1] = value
    hsv[...,2] = 1
    if border_col == 'w':
        border_hue = [0,0,1]
    else:
        border_hue = [0.16,1,1]
    hsv[0,:] = border_hue # yellow border
    hsv[-1,:] = border_hue
    hsv[:,0] = border_hue
    hsv[:,-1] = border_hue
    rgb = hsv_to_rgb(hsv)
    return(rgb)

def ClusterData(mat):
    clust = hierarchy.linkage(mat, 'ward')
    dendro = hierarchy.dendrogram(clust, no_plot = True)
    xdendro = np.array(dendro["icoord"])
    ydendro = np.array(dendro["dcoord"])
    order = [int(l) for l in dendro["ivl"]]
    return(xdendro,ydendro,order)

def ReadCSVmatrix(csvfile):
    data = np.array(list(csv.reader(open(csvfile))))
    col_labels = list(data[0,1:])
    row_labels = list(data[1:,0])
    mat =np.asarray(data[1:,1:], dtype=float)
    return(mat,row_labels,col_labels)

def WriteCSVmatrix(csvout,occmat,row_labels, col_labels):
    occmatlist = occmatlist = occmat.tolist()
    row_labels.reverse()
    nr,nc = occmat.shape
    csvmat =[[""] + col_labels]
    for i in range(nr):
        row = [row_labels[i]]
        for j in range(nc):
            row.append(str(occmatlist[i][j]))
        csvmat.append(row)
    with open(csvout,"w+") as occmat_csv:
        csvWriter = csv.writer(occmat_csv,delimiter=',')
        csvWriter.writerows(csvmat)
        
    

def DrawOccMat(csvfile = None,
               occmat = None,
               row_labels = None,
               col_labels = None,
               cluster_rows = True,
               cluster_cols = True,
               log_row_dendro = True, 
               log_col_dendro = True,
               show_row_labels = True,
               show_col_labels = True,
               xsize = 8,
               ysize = 8,
               plotscale = 40,
               dendro_width = 1.0,
               font = 'Helvetica', 
               font_path_width = 0.02,
               row_label_x_offset = 0.2,
               row_label_y_offset = 0.3,
               col_label_x_offset = 0.3,
               col_label_y_offset = -0.2,
               row_label_size = 0.6,
               col_label_size = 0.6,
               show_pdf = True,
               hue = 0.62,
               border_color = 'y',
               save_ordered_csv = False):
    

    # input from .csv file, or directly from optional arguments
    
    if csvfile:
        occmat, row_labels, col_labels = ReadCSVmatrix(csvfile)
        nr, nc = occmat.shape
    else:
        nr, nc = occmat.shape
        if nr == 0 or nc == 0:
            print('Occupancy matrix must be read from csvfile or passed as command line option --occmat var')
            sys.exit()
        if len(row_labels) != nr:
            print('Dimensions of occmat and row labels do not match')
            sys.exit()
        if len(col_labels) != nc:
            print('Dimensions of occmat and col labels do not match')
            sys.exit()
        
    
    occmat_save = copy.deepcopy(occmat)
    # set up plot
    fig, ax = plt.subplots(figsize = (nc,nr))  # plot internal coords are 1 unit/row-col
    fig.set_size_inches(xsize, ysize) # absolute plot size for display in inches
    plt.rcParams["figure.autolayout"] = True
    plt.axis('off')
    fp = FontProperties(family=font) # plot label font
    
       
    # clustering 
    if cluster_rows:
        row_yscale = 0.1  # dendrograms from scipy cluster are scaled by 10X
        row_xscale = 0.1
        row_dendro_x, row_dendro_y, row_order = ClusterData(occmat)  # row dendrogram
        if log_row_dendro:
            row_dendro_y = np.log2(row_dendro_y + 1) # log transform  
            row_yscale = float(np.log2(nc))/np.amax(row_dendro_y)
        row_dendro_coords = np.array([-row_dendro_y * row_yscale, row_dendro_x * row_xscale]).transpose(1,0,2)
        row_nd,npt,ns = row_dendro_coords.shape
    
        # shuffle labels to dendrogram order
        row_labels_ordered = []
        for i in range(nr):
            row_labels_ordered.append(row_labels[row_order[i]])
        row_labels_ordered.reverse()
        occmat = occmat[row_order,:]
        # imgmat = imgmat[row_order,:,:,:,:] # shuffle row order
        
    else:
        row_labels_ordered = row_labels
        row_labels.reverse()
        
    if cluster_cols:
        col_yscale = 0.1
        col_xscale = 0.1
        col_dendro_x, col_dendro_y, col_order = ClusterData(occmat.transpose()) # column dendrogram
        if log_col_dendro:
            col_dendro_y = np.log2(col_dendro_y+1)
            col_yscale = float(np.log2(nr)/np.amax(col_dendro_y)) 
        col_dendro_coords = np.array([col_dendro_x * col_xscale, nr + col_dendro_y * col_yscale]).transpose(1,0,2)
        col_nd,npt,ns = col_dendro_coords.shape
        col_labels_ordered = []
        for i in range(nc):
            col_labels_ordered.append(col_labels[col_order[i]])
        occmat = occmat.transpose(1,0)[col_order,:].transpose(1,0)
        # imgmat = imgmat.transpose(1,0,2,3,4)[col_order,:,:,:,:].transpose(1,0,2,3,4) # shuffle column order
     
    else:
        col_labels_ordered = col_labels
    
    # build occmat image
    imgmat = np.zeros((nr,nc,plotscale,plotscale,3))
    for row in range(nr):
        for col in range(nc):
            imgmat[row,col] = PaintPixel(occmat[row,col], npixels = plotscale, value_hue = hue, border_col = border_color)
    # reshape array of images (nx,ny,plotscale,plotscale,3) to single image (nx*plotscale,ny*plotscale,3)
    imgmat = imgmat.transpose([0,2,1,3,4])
    imgmat = imgmat.reshape(nr*plotscale,nc*plotscale,3)
    
    # display occmat
    ax.imshow(imgmat, extent=[0, nc, 0, nr])
    
    # plot dendrogram segments
    if cluster_cols:
        for i in range(col_nd):
            ax.plot(col_dendro_coords[i,0,:],col_dendro_coords[i,1,:], color = 'k', lw = dendro_width)
        
    if cluster_rows:
        for i in range(row_nd):
            ax.plot(row_dendro_coords[i,0,:],row_dendro_coords[i,1,:], color = 'k', lw = dendro_width)
    
    # plot labels
    
    if show_row_labels:
        for i in range(nr):
            labx = nc + row_label_x_offset
            laby = i + row_label_y_offset
            tp = TextPath((labx , laby), str(row_labels_ordered[i]), size=row_label_size, prop = fp)
            ax.add_patch(PathPatch(tp, color="black",lw = font_path_width))
        
    if show_col_labels:
        for i in range(nc):
            labx = i + col_label_x_offset
            laby = col_label_y_offset
            tp = TextPath((labx, laby), str(col_labels_ordered[i]), size=col_label_size, prop = fp)
            rtp = Affine2D().rotate_around(labx,laby,-np.pi/2).transform_path(tp) # rotate label
            ax.add_patch(PathPatch(rtp, color="black",lw = font_path_width))
    

    if show_pdf:
        if csvfile:
            pdffile = csvfile.split(".")[0] + ".pdf"
        else:
            pdffile = "occmat.pdf"
        fig.savefig(pdffile, format='pdf', dpi=300)
        
    if save_ordered_csv:
        if csvfile:
            csvout = csvfile.split(".")[0] + "_ordered.csv"
        else:
            csvout = "occmat_ordered.csv"
        WriteCSVmatrix(csvout,occmat,row_labels_ordered, col_labels_ordered)
    
    plt.show()
    
    return(occmat, row_labels_ordered, col_labels_ordered)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Occupancy Matrix Clustering Arguments')
    parser.add_argument('--csvfile', dest = 'csvfile', default = None, help= 'csv file containing occupancy matrix')
    parser.add_argument('--occmat', dest = 'occmat', default = None, help = ' numpy array containing values to be clustered')
    parser.add_argument('--row_labels', dest = 'row_labels', default = None, help = ' list of row labels')
    parser.add_argument('--col_labels', dest = 'col_labels', default = None, help = ' list of col labels')
    parser.add_argument('--cluster_rows', dest='cluster_rows', type=str_to_bool, nargs='?', default=True, help= ' Flag to cluster Rows, default = True')
    parser.add_argument('--cluster_cols', dest='cluster_cols', type=str_to_bool, nargs='?', default=True, help= ' Flag to cluster Columns, default = True')
    parser.add_argument('--log_row_dendro', dest='log_row_dendro', type=str_to_bool, nargs='?', default=True, help= ' Flag to take log2 of row dendrogram, default = True')
    parser.add_argument('--log_col_dendro', dest='log_col_dendro', type=str_to_bool, nargs='?', default=True, help= ' Flag to take log2 of column dendrogram, default = True')
    parser.add_argument('--show_row_labels', dest='show_row_labels', type=str_to_bool, nargs='?', default=True, help= ' Flag to show row labels, default = True')
    parser.add_argument('--show_col_labels', dest='show_col_labels', type=str_to_bool, nargs='?', default=True, help= ' Flag to show column labels, default = True')
    parser.add_argument('--xsize', dest='xsize', default = 8,type = float, help= 'X size of output plot in inches, default = 8')
    parser.add_argument('--ysize', dest='ysize', default = 8,type = float, help= 'Y size of output plot in inches, default = 8')
    parser.add_argument('--pixel_size', dest='pixel_size', default = 40, type = int, help= 'number of pixels per element of the occupancy matrix, default = 40')
    parser.add_argument('--dendro_linewidth', dest='dendro_linewidth', default = 1.0, type = float, help= 'linewidth to use for drawing dendrograms, default = 1.0')
    parser.add_argument('--font', dest='font', default = 'Helvetica', help= 'font for dendrogram labels, default = ''Helvetica')
    parser.add_argument('--font_path_width', dest='font_path_width', default = 0.02, type = float, help= 'linewidth to use for drawing dendrograms, default = 0.02')
    parser.add_argument('--row_label_x_offset', dest='row_label_x_offset', default = 0.2, type = float, help= 'x offset for labels in fraction of row, default = 0.2')
    parser.add_argument('--row_label_y_offset', dest='row_label_y_offset', default = 0.3, type = float, help= 'y offset for labels in fraction of row, default = 0.3')
    parser.add_argument('--col_label_x_offset', dest='col_label_x_offset', default = 0.3, type = float, help= 'x offset for labels in fraction of column, default = 0.3')
    parser.add_argument('--col_label_y_offset', dest='col_label_y_offset', default = -0.2, type = float, help= 'y offset for labels in fraction of column, default = -0.2')
    parser.add_argument('--row_label_size', dest='row_label_size', default = 0.6, type = float, help= 'row label size as fraction of row, default = 0.6')
    parser.add_argument('--col_label_size', dest='col_label_size', default = 0.6, type = float, help= 'column label size as fraction of column, default = 0.6')
    parser.add_argument('--show_pdf', dest = 'show_pdf', type=str_to_bool, nargs='?', default=True, help = ' Flag to supress pdf output, default = True')
    parser.add_argument('--hue', dest='hue', default = 0.62, type = float, help= 'hue (for HSV) for pixels, default = 0.62 (blue)')
    parser.add_argument('--border_color', dest='border_color', default = 'y', type = str, help= 'border color between pixels (y or w), default = y (yellow)')
    parser.add_argument('--save_ordered_csv', dest = 'save_ordered_csv', type = str_to_bool, nargs = '?', default = False, help = 'Flag to save ordered output in .csv')


    args = parser.parse_args()
    
    DrawOccMat(csvfile = args.csvfile,
                   occmat = args.occmat,
                   row_labels = args.row_labels,
                   col_labels = args.col_labels,
                   cluster_rows = args.cluster_rows,
                   cluster_cols = args.cluster_cols,
                   log_row_dendro = args.log_row_dendro, 
                   log_col_dendro = args.log_col_dendro,
                   show_row_labels = args.show_row_labels,
                   show_col_labels = args.show_col_labels,
                   xsize = args.xsize,
                   ysize = args.ysize,
                   plotscale = args.pixel_size,
                   dendro_width = args.dendro_linewidth,
                   font = args.font, 
                   font_path_width = args.font_path_width,
                   row_label_x_offset = args.row_label_x_offset,
                   row_label_y_offset = args.row_label_y_offset,
                   col_label_x_offset = args.col_label_x_offset,
                   col_label_y_offset = args.col_label_y_offset,
                   row_label_size = args.row_label_size,
                   col_label_size = args.col_label_size,
                   show_pdf = args.show_pdf,
                   hue = args.hue,
                   border_color = args.border_color,
                   save_ordered_csv = args.save_ordered_csv)
