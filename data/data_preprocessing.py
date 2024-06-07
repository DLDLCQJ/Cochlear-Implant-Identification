import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder

def pad_todesire_2D(img, desired_shape):
    X_before = int((desired_shape[0]-img.shape[0])/2)
    Y_before = int((desired_shape[1]-img.shape[1])/2)
    X_after = desired_shape[0]-img.shape[0]-X_before
    Y_after = desired_shape[1]-img.shape[1]-Y_before
    npad = ((X_before, X_after),
            (Y_before, Y_after))
    padded = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
    return padded

def crop_center_2D(img,crop_shape):
    x,y = img.shape
    cropx = crop_shape[0]
    cropy = crop_shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[startx:startx+cropx,starty:starty+cropy]

def augment():
    return transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize(image_resize=128,image_resize=128),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(45,),
    transforms.RandomVerticalFlip(p = 0.5,),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def norm_meta(X, norm_type='demean_std'):
    if norm_type == 'demean_std':
        X_o = np.float32(X.copy())
        m = np.mean(X_o)
        s = np.std(X_o)
        normalized_X = np.divide((X_o - m), s)
    elif norm_type == 'minmax':
        perc1 = np.percentile(X, 1)
        perc99 = np.percentile(X, 99)
        normalized_X = np.divide((X - perc1), (perc99 - perc1))
        normalized_X[normalized_X < 0] = 0.0
        normalized_X[normalized_X > 1] = 1.0
    return torch.tensor(normalized_X,dtype=torch.float32)

def meta_engineer(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object]).columns
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna('Unknown', inplace=True)

    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df
