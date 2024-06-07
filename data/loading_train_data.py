import os
from os.path import join as opj
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import SimpleITK as sitk
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, norm_type, X,y,meta=None,augment=None):
        super(MyDataset, self).__init__()
        self.X = X
        self.y = y
        self.norm_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
        self.augment = augment
        self.meta = meta
        self.norm_type = norm_type

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X, label = self.X[idx], self.y[idx]
        if self.meta is not None:
            meta = norm_meta(X, self.norm_type)
            return (meta, label)
        else:
            if self.augment is not None:
                image = self.augment(X)
            else:
                image = torch.tensor(X, dtype=torch.float32)
                image = self.norm_img(image)
            return (image,torch.tensor(label, dtype=torch.float32))

def loading_data(args):
    nifti_list = pd.read_csv(opj(args.path,args.img_file + '.csv'))
    imgs = []
    for nii_file in nifti_list.to_numpy():
        filename = os.path.split(nii_file[0])[1]
        img = sitk.ReadImage(nii_file)
        img_arr = np.squeeze(sitk.GetArrayFromImage(img)) 
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
            processed_img = np.array(final_img).astype(float)
            if i % 10 == 0:
                  plot_2D_MRI(processed_img)
            imgs.append(processed_img)
    ##
    imgs = np.squeeze(np.array(imgs))
    imgs = imgs[:, np.newaxis, :, :]
    processed_img = np.repeat(imgs,3,axis=1)
    ##
    y = pd.read_csv(opj(args.path, args.label_file + '.csv')).iloc[:,1].values
    y = np.repeat(y, 15)

    return processed_img, y
