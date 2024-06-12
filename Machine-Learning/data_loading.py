import os
import numpy as np
import pandas as pd
import nilearn
from nilearn import plotting
from os.path import join as opj


def load_data(path):

    nifti_list = pd.read_csv(opj(path,'CI_brain_jacobian_withoutcbl_lure_eng.csv'))
    imgs = []
    for f, file in enumerate(nifti_list.to_numpy()):
        img = nilearn.image.load_img(file)
        org_img = np.squeeze(nilearn.image.get_data(img))
        org_shape = list(org_img.shape)
        desired_shape = org_shape
        crop_shape = org_shape
        for i in range (3):
            if desired_shape[i] < org_shape[i]:
                crop_shape[i] = desired_shape[i]  
        cropped_img = crop_center(org_img, crop_shape)
        final_img = pad_todesire(cropped_img, desired_shape)
        processed_img = np.array(final_img).astype(float)
        print(processed_img.shape)
        if f % 10 == 0:
            mid_slice_x_after = processed_img
            plt.imshow(mid_slice_x_after[:,60,:], cmap='gray', origin='lower')
            plt.xlabel('First axis')
            plt.ylabel('Second axis')
            plt.colorbar(label='Signal intensity')
            plt.show()
        imgs.append(processed_img)
    ##
    processed_img = np.squeeze(np.array(imgs))
    processed_img = processed_img[:, np.newaxis, :, :, :]
    y = pd.read_csv(opj(path, "CI_Meta_lure_eng.csv")).iloc[:,1].values
    return processed_img,y

def Float_MRI(ims):
    flatmap = np.array([im.flatten() for im in ims]) 
    evox = ((flatmap**2).sum(axis=0)!=0)
    flatmap = flatmap[:,evox] # only analyze voxels with values > 0
    X = flatmap-flatmap.mean(axis=0) # center each voxel at zero
    return X
