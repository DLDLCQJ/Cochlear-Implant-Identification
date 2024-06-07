import torch
import torch.nn as nn

def linear_block(input_size,output_size,hidden_size=256,transformer=True,activation_func=nn.ReLU,dropout=0.5):
    layers = []
    if transformer:
        layers.append(nn.LayerNorm(normalized_shape=input_size))
        layers.append(nn.Linear(input_size, output_size, bias=True))
    else:
        layers.append(nn.Linear(input_size, hidden_size, bias=True))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(activation_func())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, output_size, bias=True))

    return nn.Sequential(*layers)

class adapter_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = linear_block(args.clinical_features,args.num_classes,args.hidden_size,False,nn.SELU)
    def forward(self, x):
        out = self.fc(x)
        return out
