import numpy as np
import pandas as pd
import os
from os.path import join as opj
from PIL import Image
import torch
import torch.nn as nn

from dataset.data_preprocessing import pad_todesire_2D, crop_center_2D


def loading_data(args):
    nifti_list = pd.read_csv(opj(args.path,args.img_file + '.csv'))
    imgs = []
    for nii_file in nifti_list.to_numpy():
        filename = os.path.split(nii_file[0])[1]
        img = sitk.ReadImage(nii_file)
        img_arr = np.squeeze(sitk.GetArrayFromImage(img)) #reg:181*217*181 / mvps:91*109*91 / jacobian:181*217*181
        print(img_arr.shape)
        for i, slice_number in enumerate(range(80,95)):        ##30-45
            new_filename= "third_"+str(slice_number)+"_"+filename     #For FreeSurfer reconstructed image
            img_2D = img_arr[slice_number, :, :].astype(np.float32)
            # save_path = opj(filename +"_2D", new_filename)
            # nib.save(nib.Nifti1Image(fdata_2D, affine = np.eye(4)), save_path)
            min_shape = min(img_2D.shape)
            cropped_shape = list((min_shape,min_shape))
            cropped_img = crop_center_2D(img_2D, cropped_shape)
            for j in range (2):
                if args.desired_shape[j] < cropped_shape[j]:
                    #final_img = crop_center_2D(cropped_img, args.desired_shape)
                    final_img =  resize(cropped_img, args.desired_shape, order=3, mode='reflect', anti_aliasing=True)
                else:
                    final_img = pad_todesire_2D(cropped_img, args.desired_shape)
            processed_img = np.array(final_img).astype(float)
            #print(processed_img.shape)
            # if i % 10 == 0:
            #     mid_slice_x_after = processed_img
            #     plt.imshow(mid_slice_x_after, cmap='gray', origin='lower')
            #     plt.xlabel('First axis')
            #     plt.ylabel('Second axis')
            #     plt.colorbar(label='Signal intensity')
            #     plt.show()
            imgs.append(processed_img)
    ##
    imgs = np.squeeze(np.array(imgs))
    imgs = imgs[:, np.newaxis, :, :]
    processed_img = np.repeat(imgs,3,axis=1)
    ##
    y = pd.read_csv(opj(args.path, args.label_file + '.csv')).iloc[:,1].values
    y = np.repeat(y, 15)

    return processed_img, y


def load_test_data(args):
    nifti_list = pd.read_csv(opj(args.path,args.test_img_file + '.csv'))
    imgs = []
    for nii_file in nifti_list.to_numpy():
        filename = os.path.split(nii_file[0])[1]
        img = sitk.ReadImage(nii_file)
        img_arr = np.squeeze(sitk.GetArrayFromImage(img)) #reg:181*217*181 / mvps:91*109*91
        ##
        for i, slice_number in enumerate(range(80,95)):
            new_filename= "third_"+str(slice_number)+"_"+filename     #For FreeSurfer reconstructed image
            img_2D = img_arr[slice_number, :, :].astype(np.float32)
            # save_path = opj(filename +"_2D", new_filename)
            # nib.save(nib.Nifti1Image(fdata_2D, affine = np.eye(4)), save_path)
            min_shape = min(img_2D.shape)
            cropped_shape = list((min_shape,min_shape))
            cropped_img = crop_center_2D(img_2D, cropped_shape)
            for j in range (2):
                if args.desired_shape[j] < cropped_shape[j]:
                    #final_img = crop_center_2D(cropped_img, args.desired_shape)
                    final_img =  resize(cropped_img, args.desired_shape, order=3, mode='reflect', anti_aliasing=True)
                else:
                    final_img = pad_todesire_2D(cropped_img, args.desired_shape)
            #print(final_img.shape)
            processed_img = np.array(final_img).astype(float)
            imgs.append(processed_img)
    ##
    processed_img = np.squeeze(np.array(imgs))
    processed_img = processed_img[:, np.newaxis, :, :]
    processed_img = np.repeat(processed_img,3,axis=1)
    ##
    y = pd.read_csv(opj(args.path, args.test_label_file + '.csv')).iloc[:,1].values
    y = np.repeat(y, 15)
    #y = np.eye(args.num_classes)[y]
    return processed_img, y
