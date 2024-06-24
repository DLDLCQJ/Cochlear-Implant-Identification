"""
@author: yuandi
"""

import ants
import antspynet
import os
import pandas as pd
import numpy as np

import nibabel as nib
import platform

if platform.system()=='Windows':
    print('current platform is windows')
    datapath='XXX'
    exlpath='XXX'
    tpmpath='XXX'
else:
    print('current platform is MAC')
    datapath='XXX'
    exlpath='XXX'
    tpmpath='XXX'


exlfile=os.path.join(exlpath,"XXX.csv")
behdata=pd.read_csv(exlfile)

sub_list=behdata["SUB_ID"]
sub_num=sub_list.size

tpmname="infant-1yr-withCerebellum.nii.gz"
tpmfile=os.path.join(tpmpath,tpmname)

for idx in range(sub_num):
    name=sub_list[idx]
    print('>>>>>>>>>>>>>>>>>loading image of subject',str(name),'...')
    
    filename="r_" + str(name) + ".nii.gz"
    imgfile=os.path.join(datapath,filename)
    
    image=ants.image_read(imgfile)
    
    ##############
    preprocessed_image=antspynet.preprocess_brain_image(image,brain_extraction_modality=None,verbose=False)
    
    brain=preprocessed_image["preprocessed_image"] 
    mask = antspynet.brain_extraction(brain, modality="t1")
    
    p_mask=mask>0.3
    extracted_brain = brain * p_mask
    
    brainfilename="r_" + str(name) + "_brain.nii.gz"
    maskfilename="r_" + str(name) + "_mask.nii.gz"
    extractedbrainfilename="r_" + str(name) + "_extractedbrain.nii.gz"
    
    brainimgfile=os.path.join(datapath,brainfilename)
    maskimgfile=os.path.join(datapath,maskfilename)
    extractedbrainimgfile=os.path.join(datapath,extractedbrainfilename)
    
    ants.image_write(mask,maskimgfile)
    ants.image_write(brain,brainimgfile)
    ants.image_write(extracted_brain,extractedbrainimgfile)
    
    savefile=os.path.join(datapath,str(name) + "overlay.jpg")
    ants.plot(brain,overlay=extracted_brain,overlay_alpha=0.9,axis=2,nslices=36,slice_buffer=15,title=str(name),filename=savefile)
    
    fi = ants.image_read(tpmfile)
    mi = extracted_brain
    mytx = ants.registration(fixed=fi , moving=mi, type_of_transform = ('SyN') )
    jac = ants.create_jacobian_determinant_image(fi,mytx['fwdtransforms'][0],1)
    
    jacfilename="r_" + str(name) + "_jacobian.nii.gz"
    jacimgfile=os.path.join(datapath,jacfilename)
    ants.image_write(jac,jacimgfile)
    
