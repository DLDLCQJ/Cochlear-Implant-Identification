import os
import sys
import numpy as np
import pandas as pd
from os.path import join as opj
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Transfer Learning implementation in MRI using PyTorch.')
    parser.add_argument('--path', type=str, default='/direct to your MRI doctory/', help='Path of data files')
    parser.add_argument('--img_file', type=str, default='XXX', help='Path of data files')
    parser.add_argument('--label_file', type=str, default="XXX", help='Path of label files')
    parser.add_argument('--test_img_file', type=str, default='XXX', help='Path of data files')
    parser.add_argument('--test_label_file', type=str, default="XXX", help='Path of label files')
    parser.add_argument('--desired_shape', type=list, default=[128,128], help='target shape of input image')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--num_epochs_pre', type=int, default=200,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='The dimension of hidden_size in adapter_layer')
    parser.add_argument('--clinical_features', type=int, default=19, metavar='N',
                        help='The number of the clincial features for classify?')
    parser.add_argument('--num_classes', type=int, default=1, metavar='N',
                        help='What would you like to classify?')
    parser.add_argument('--embedding_dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--c', type=float, default=0.1,
                        help='Regularization parameter')
    parser.add_argument('--k', type=float, default=5,
                        help='the number of nfold in Cross_validation')
    parser.add_argument('--betas', type=float, default=(0.9,0.999),
                        help='Optimizer parameters')
    parser.add_argument('--epsilon', type=float, default=1e-08,
                        help='Optimizer parameters')
    parser.add_argument('--norm_type', type=str, default='demean_std', choices=["demean_std", "minmax","None"],
                        help='help=How to preprocess data: demean_std, minmax or None')
    # parser.add_argument('--rg_type', type=str, default='KM', choices=['L1', 'L2', 'L1L2','HybridReg'],
    #                     help='Regularization type to use: L1 (LASSO), L2 (Ridge), Elastic net (beta*L2 + L1) or HybridReg')
    parser.add_argument('--kernel', type=int, default=16, metavar='N',
                        help='the kernel size (default: 16')
    parser.add_argument('--network', type=str, default='Mobilenet', metavar='str',
                        help='the network name (default: [Googlenet, Alexnet, Mobilenet,VGG, Resnet, Densenet, Inception])')
    parser.add_argument('--pretrained', type=str, default=True, metavar='str',
                        help='Loading pretrained model (default: False)')
    parser.add_argument('--optim', type=str, default='ADAM', metavar='str',
                        help='the optimizer (default: Adam)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers to use in data loading')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of workers to use in data loading')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate for training')
    # parser.add_argument('--lr_clinical', type=float, default=1e-4,
    #                     help='Initial learning rate for clincial training')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping')
    parser.add_argument('--lr_scheduler_warmup_ratio', type=int, default=0.1,
                        help='Number of workers to use in data loading')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                        help='help=Type of learning rate')
    parser.add_argument('--save_dir', type=str, default='XXX', help='Location of model temporarily')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    return parser.parse_args()  # Running from command line

def main(args):
    set_seed(123)
    index_train, index_test = train_test_split(list(range(X.shape[0])), train_size=0.8, test_size=0.2, shuffle=True)
    nifti_list = pd.read_csv(opj(args.path,args.img_file + '.csv'))
    y = pd.read_csv(opj(args.path, args.label_file + '.csv')).iloc[:,1].values
    X_train_list, y_train_list = nifti_list[index_train],y[index_train]
    X_test_list,y_test_list = nifti_list[index_test],y[index_test]
    X_train, y_train = loading_data(X_train_list,y_train_list)
    X_test,y_test = loading_data(X_test_list,y_test_list)
    ##
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    test_dataset = MyDataset(args.norm_type, X_test, y_test)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    # Cross Validation
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        print(f"X_train.shape: {X_train.shape}", f"y_train.shape: {y_train.shape}")
        train_dataset = MyDataset(args.norm_type,X_train, y_train)
        val_dataset = MyDataset(args.norm_type,X_val, y_val)
        train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)  #45 - recommended value for batchsize
        val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False)

        # Initialized the model to be trained
        if args.pretrained:
            model = Loading_pretrained(args.network,
                                       args.num_classes,
                                       args.hidden_size,
                                       pretrained=args.pretrained)
        else:
            print('without network')
        #model.apply(reset_weights)
        model.to(args.device)
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        #criterion = nn.CrossEntropyLoss()
        if args.optim == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=args.betas, eps=args.epsilon)
        if args.optim == 'ADAM_W':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0)
        else:
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        # Calculate the number of total steps
        num_training_steps = len(train_dataloader) * args.num_epochs
        num_warmup_steps = int(num_training_steps * args.lr_scheduler_warmup_ratio)
        # Define the scheduler
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        ## Training
        results_val = train_epoch(args,model, train_dataloader, val_dataloader, criterion, optimizer, lr_scheduler, fold)
        ## Testing
        results_te = test_epoch(args,model, test_dataloader, criterion, fold)
        # Save the scores of model
        with open(opj(args.save_dir, f"model_fold{fold+1}_cv_predicts_and_labels_val_{args.network}_{args.img_file}.pkl"), "wb") as f:
            pickle.dump(results_val, f)
        with open(opj(args.save_dir, f"model_fold{fold+1}_cv_predicts_and_labels_te_{args.network}_{args.img_file}.pkl"), "wb") as f:
            pickle.dump(results_te, f)

if __name__ == "__main__":
    args = parse_arguments()  # You could provide default args here
    main(args)
