import torch
import torch.nn as nn
from torchvision import transforms

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
