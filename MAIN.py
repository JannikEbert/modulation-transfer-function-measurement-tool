# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 16:03:42 2021

@author: ebert
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import MTF_functions as mf
from MTF_Classes import ROI_Selector

def main():
    

    ############################## HOW TO USE ##############################

    #  1.   adjust input settings below if you want
    #  2.   run the script
    #  3.   a plot shows the first image in the input folder
    #  4.   if you want to calculate a mtf with that image select a region of 
    #       interest (ROI) with your mouse (press, draw, release)
    #       if you want to skip the image simply close the plot window
    #  5.   if you choose a ROI the mtf will be calculated
    #  6.   if "save_mtf==True" a plot window will show the result
    #  7.   if "plot_mtf==True" the mtf is saved as npy and the plot as svg
    #  8.   step 3 to 7 will repeat for all images in the input folder

    # GENERAL TIPS

    # Be sure that every part of the ROI is withing the measuring range
    # otherwise the mtf will not converge to 0 because of a sharp bend
    # due to over or under saturation

    # make sure the slanted edge is sharp in the selected ROI
    # for small distances the image can be blurry towards the edge
    # small changes in the focus setting can result in large inaccuracies

    # Aim for a continous mtf with no sharp bends at low spatial frequencies
    #     if mtf has a sharp bend pointing DOWNWARDS at low spatial frequency:
    #         --> ROI is to wide (ESF: _______/°°°°°°°°°°)
    #     if mtf has a sharp bend pointing UPWARDS at low spatial frequency:
    #          --> ROI is to narrow (ESF: _/°)

    # Calculate spatial frequency: fx = fx_norm/a 
    #                              with fx_norm = fx*a and pixel width a


    ############################ INPUT SETTINGS ############################

    strahler = "Infrared_Systems" # select strahler (used in: data_in, path_out)
    camera = "VarioCam" # select camera (used in: data_in, path_out)

    plot_mtf = True # plot result at the end
    save_mtf = True # save calculated mtf and plot at the end
    auto_roi = False # define roi limits based on file name (advanced option)
    ss = 4 # supersampling size (calculate: ss = int(90-tan^-1(slope[°])) --> ss=4 for slope=15°)

    # input and output path structure
    data_in = 'Input/' + strahler + '/' + camera + '/*' # path of input data (star (*) later creates a list of all images in the input folder)
    path_out = 'Output/' + strahler + '/' + camera + '/' # path of output folder


    ############################ PROGRAMM STARTS ############################

    file_names = glob.glob(data_in)
    # for all images...
    for file_idx, file_name in enumerate(file_names):
        print('{:>5} of {:>5}'.format(file_idx+1,len(file_names)))
        
        # read image                                                        <--- put MTF
        if camera == "OptrisPI450":
            img = mf.read_image_optris_pi450(file_name)
        else:
            img = mf.read_image_npy(file_name)
        
        # define roi limits
        if auto_roi == True:
            x_start, x_end, y_start, y_end = mf.get_roi_limit(img, file_name)
        else:
            print("Please select ROI. Close plot window to confirm selection.")
            Selector = ROI_Selector(img)
            plt.show()
            x_start, x_end, y_start, y_end = Selector.get_roi()
            if np.array([Selector.get_roi()]).all() == None:
                print("Nothing selected. Cancel Mission!")
                continue
            
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
            ax1.set_title("IR Image")
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
            if save_mtf == True:
                plt.savefig(path_out+'Ergebnis_'+os.path.basename(file_name)[:-5]+'_ROI_'+str(x_start)+'-'+str(x_end)+'-'+str(y_start)+'-'+str(y_end)+'.svg')
            plt.show()
        
        # Ergebnis speichern
        if save_mtf == True:
            mtf_dat = np.zeros([2,len(mtf)])
            for i in range(len(mtf)):
                mtf_dat[0][i]=fx[i]
                mtf_dat[1][i]=mtf[i]
            np.save(path_out+'Ergebnis_'+os.path.basename(file_name)[:-4]+'_ROI_'+str(x_start)+'-'+str(x_end)+'-'+str(y_start)+'-'+str(y_end),mtf_dat)
            

  
if __name__ == '__main__':
    main()