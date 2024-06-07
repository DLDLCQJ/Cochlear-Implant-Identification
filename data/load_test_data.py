import numpy as np
import pandas as pd
import os
from os.path import join as opj
from PIL import Image

def load_test_data(args):
    nifti_list = pd.read_csv(opj(args.path,args.test_img_file + '.csv'))
    imgs = []
    for nii_file in nifti_list.to_numpy():
        filename = os.path.split(nii_file[0])[1]
        img = sitk.ReadImage(nii_file)
        img_arr = np.squeeze(sitk.GetArrayFromImage(img)) 
        ##
        for i, slice_number in enumerate(range(80,95)):
            new_filename= "third_"+str(slice_number)+"_"+filename     
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