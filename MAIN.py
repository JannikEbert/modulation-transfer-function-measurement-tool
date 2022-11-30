# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 16:03:42 2021

@author: ebert
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import MTF_functions as mf

def main():
    # settings
    file_names = glob.glob('Input/*') # file name: MTF_<Distance1/2/3>_<slope>_<H/V>.npy (V=vertical edge)
    plot_mtf = True # show plot
    save_mtf = False # save mtf
    auto_roi = True # define roi limits based on file name
    x_start, x_end = 130, 215 # otherwise this roi
    y_start, y_end = 100, 200 # otherwise this roi
    ss = 4 # supersampling size (int(90-tan^-1(slope)[deg]))
    
    # for all images...
    for file_idx, file_name in enumerate(file_names):
        print('{:>5} of {:>5}'.format(file_idx+1,len(file_names)))
        
        # read image
        img = mf.read_image_optris_pi450(file_name)
        
        # define roi limits
        if auto_roi == True:
            x_start, x_end, y_start, y_end = mf.get_roi_limit(img, file_name)
        else:
            print("Warning: Auto-ROI deactivated!")
            
        # get roi and it's differentiation dROI
        roi, droi = mf.get_roi_droi(img,(x_start, y_start),(x_end, y_end))
        
        # calculate edge position
        if file_name[-8] == '0': # edge without slope (Only for Optris PI450)
            fit = np.ones(np.shape(roi)[0])*(-x_start+x_end)/2
        else:
            pos = mf.get_edge_position(droi)
            fit = mf.horizontal_fit(pos,range(np.shape(roi)[0])) # linear best fit edge position
            pos = mf.get_edge_position(droi, fit) # refine edge position with centered window
            fit = mf.horizontal_fit(pos,range(np.shape(roi)[0])) # linear best fit refined edge position
        
        # calculate supersampled esf
        esf, roi_ss_cen = mf.get_ss_esf(roi, fit, ss)
        
        # calculate lsf
        lsf = mf.get_windowed_lsf(esf)
        # lsf = lsf/lsf[int(len(lsf)/2)-1] # norm

        # calculate mtf
        mtf = np.abs(np.fft.fft(lsf))
        mtf = mtf[0:int(len(mtf)/(len(esf)/roi.shape[1]))] # cut mtf because of supersampling
        
        # norm mtf
        if mtf[0] != 0:
            mtf = mtf/mtf[0]
        else:
            print("MTF Normierung Fehlgeschlagen")      
 
        # norm frequency
        fx = []
        for n in range(len(mtf)):
            fx.append(n/(len(mtf))) # x_end-x_start
        
        # plot
        if plot_mtf == True:
            fig, _ = plt.subplots(figsize=(12,9))
            title = str("[Nr. "+str(file_names.index(file_name))+"] "+file_name)
            #fig.canvas.set_window_title(title)
            
            ax1 = plt.subplot(231)      
            ax1.set_title(file_name[-11:])
            ax1.imshow(img)
            ax1.plot([x_start,x_end], [y_start,y_start], color = 'red')
            ax1.plot([x_start,x_end], [y_end,y_end], color = 'red')
            ax1.plot([x_start,x_start], [y_start,y_end], color = 'red')
            ax1.plot([x_end,x_end], [y_end,y_start], color = 'red')
            
            ax2 = plt.subplot(232)          
            ax2.set_title('ROI')
            ax2.imshow(roi)
            ax2.plot(fit,np.arange(0,roi.shape[0]),color = 'red')
            
            ax3 = plt.subplot(233)
            ax3.set_title('SS ESF (centered)')
            ax3.set(xlabel='')
            ax3.set(ylabel='')
            ax3.imshow(roi_ss_cen, interpolation='nearest')

            ax4 = plt.subplot(234)
            ax4.grid(True)
            ax4.set_title('SS ESF (mean)')
            ax4.plot(esf)
            
            ax5 = plt.subplot(235)
            ax5.grid(True)
            ax5.set_title('SS LSF')
            ax5.set(xlabel='')
            ax5.set(ylabel='')
            ax5.plot(lsf, marker='x')
            
            ax6 = plt.subplot(236)
            ax6.grid(True)
            ax6.set_title('MTF')
            ax6.set(xlabel='$a$ $f_x$ in $-$')
            ax6.set(ylabel='')
            ax6.plot(fx,mtf)
            
            plt.show()
        
        # Ergebnis speichern
        if save_mtf == True:
            mtf_dat = np.zeros([2,len(mtf)])
            for i in range(len(mtf)):
                mtf_dat[0][i]=fx[i]
                mtf_dat[1][i]=mtf[i]
            np.save('Out'+file_name[2:],mtf_dat)

  
if __name__ == '__main__':
    main()