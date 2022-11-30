# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 21:37:00 2021

@author: ebert
"""
import numpy as np
import cv2
from Fenster_RS import Hamming

def read_image_optris_pi450(file_name, norm = True):
    """
    read image file and norm
    """
    if file_name[-4:] == ".npy":
        gray = np.load(file_name)
    else:
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = (gray - 1000) / 10 # optris image format
    gmax, gmin = gray.max(), gray.min()
    gray_norm = (gray - gmin)/(gmax- gmin)
    if file_name[-5] == 'H': # ...wenn Kante horizontal
        gray_norm = cv2.rotate(gray_norm, cv2.ROTATE_90_CLOCKWISE)
    if norm == True:
        return gray_norm
    else:
        return gray

def get_roi_limit(img, file_name, distance_position = 1):
    """
    get roi limits based on file name
    """
    data_group = file_name.split('_')
    print()
    distance = data_group[distance_position]
    print(distance)
    # ROI Grenzen definieren
    if distance == "Distance1":
        x_start, x_end = 130, 220
        y_start, y_end = 200, 250

    elif distance == "Distance2":
        x_start, x_end = 130, 215
        y_start, y_end = 100, 200

    elif distance == "Distance3":
        x_start, x_end = 160, 240
        y_start, y_end = 100, 200
        
    else:
        x_start, x_end = 1, img.shape[1]-2
        y_start, y_end = 0, img.shape[0]-1
        print("Achtung: Es konnte keine ROI auf Basis des Dateinamens definiert werden!")

    return x_start, x_end, y_start, y_end
        

def get_roi_droi(img,p_start,p_end):
    """
    get roi and it's differentiation dROI
    """
    x_start, y_start = p_start
    x_end, y_end = p_end
    w, h = x_end-x_start+1, y_end-y_start+1
    roi = np.zeros((h,w))
    droi = np.zeros((h,w))
    for lin in range(h):
        for col in range(w):
            roi[lin][col] = img[lin+y_start,col+x_start] # define roi
            droi[lin][col] = 0.5*(img[lin+y_start,col+x_start+1]-img[lin+y_start,col+x_start-1]) # differentiation
    return roi, droi

def get_edge_position(droi, *args):
    """
    calculate edge position
    """
    win, _, _ = Hamming(droi.shape[1])
    win_cen = np.zeros((droi.shape[0],droi.shape[1]))
    pos = np.zeros(droi.shape[0])
    for lin in range(droi.shape[0]):
        for col in range(droi.shape[1]):
            delta = col # if no fit -> center window
            for fit in args: # if fit in input
                delta = col-int(round(fit[lin]+droi.shape[1]/2)) # center on fit coordinates
                if delta <= -droi.shape[1]:
                    delta = delta+droi.shape[1]
            win_cen[lin][col] = win[delta] # window positioning
            droi[lin][col] = droi[lin][col]*win_cen[lin][col] # window multiplication
        pos[lin] = sum([col * droi[lin][col] for col in range(np.shape(droi)[1])])/sum(droi[lin]) # center of gravity
    return pos

def horizontal_fit(x,y):
    """
    linear polyfit for fixed y values
    """
    m, b = np.polyfit(x,y,1)
    fit = (y-b)/m # x coordinate
    return fit

def get_ss_esf(roi, fit, ss):
    """
    calculate supersampled esf
    """
    h, w = roi.shape
    
    # Supersampling
    roi_ss = np.zeros((h,w*ss))
    delta = np.zeros(h)

    # move pixel to bin        
    for lin in range(h):
            for col in range(w):
                delta[lin] = int(round((fit[round(len(fit)/2)]-fit[lin])*ss)) # get center
                roi_ss[lin][col*ss] = roi[lin][col]   
                
    # align to center
    roi_ss_cen = np.zeros((h,w*ss))
    for lin in range(h):
        for col in range(w*ss):
            shift = -int(delta[lin])
            if 0 < col+shift < w*ss: 
                roi_ss_cen[lin][col] = roi_ss[lin][col+shift]

    esf = np.zeros(w*ss)
    k = np.zeros(w*ss)
    for lin in range(roi_ss_cen.shape[0]):
        for col in range(w*ss):
            if roi_ss_cen[lin][col] != 0:
                k[col] = k[col] + 1
                esf[col] = esf[col] + roi_ss_cen[lin][col]
    esf = esf[np.argwhere(k)]/k[np.argwhere(k)]
    
    roi_ss_cen[roi_ss_cen == 0] = np.nan
    roi_ss_cen = np.ma.array(roi_ss_cen, mask=np.isnan(roi_ss_cen))
    #roi_ss[roi_ss == 0] = np.nan
    #roi_ss = np.ma.array(roi_ss, mask=np.isnan(roi_ss))
    
    return esf, roi_ss_cen

def get_windowed_lsf(esf):
    """
    calculate the lsf
    """
    win, _, _ = Hamming(len(esf)) 
    lsf = np.zeros(len(esf))
    for n in range(len(esf)-1):
        if n > 0:
            lsf[n] = 0.5*(esf[n+1]-esf[n-1]) # differentiation
    # align to center
    cen = sum([n * lsf[n] for n in range(len(lsf))])/sum(lsf)
    lsf_cen = np.zeros(len(lsf))
    for n in range(len(lsf)):
        delta = int(round(cen-n+len(lsf)/2))
        if delta >= len(lsf):
            delta = delta - len(lsf)
        lsf_cen[n] = lsf[delta]
    lsf = lsf_cen*win # window

    return lsf