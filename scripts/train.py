import sys
import numpy as np
import pandas as pd
import os
from os.path import join as opj

import torch
import torch.nn as nn

from scripts.run_epoch import run_epoch

def train_epoch(args, model, train_dataloader, valid_dataloader, criterion, optimizer, lr_scheduler, fold):
    best_epoch = 0
    best_loss = float('inf')
    # Train the model
    results_tr,results_val = [], []
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch+1}")
        # Run training epoch and compute training loss
        tr_loss, tr_predicts,tr_truths = run_epoch(args, model, train_dataloader, criterion,optimizer,  lr_scheduler,  fold, epoch, is_training=True)
        # Run validation epoch and compute validation loss
        with torch.no_grad():
            val_loss, val_predicts,val_truths = run_epoch(args, model, valid_dataloader, criterion, None,  None, fold, epoch, is_training=False)
        
        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            best_epoch = epoch
            best_epoch_predicts_tr = tr_predicts
            best_epoch_labels_tr = tr_truths
            best_epoch_predicts_val = val_predicts
            best_epoch_labels_val = val_truths
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping triggered on fold {fold + 1}, epoch {epoch + 1}")
                break
        # Save the model temporarily after each epoch
        try:
            torch.save(model.state_dict(), f'{args.save_dir}/model_fold{fold + 1}_epoch{epoch + 1}_{args.network}_{args.img_file}.pth')
        except Exception as e:
            print(f"Error occurred while saving model: {e}")
        # Store results for the fold
        results_tr.append({
                "fold": fold + 1,
                "epoch": best_epoch + 1,
                "predictions": best_epoch_predicts_tr,
                "labels": best_epoch_labels_tr
            })
        results_val.append({
                "fold": fold + 1,
                "epoch": best_epoch + 1,
                "predictions": best_epoch_predicts_val,
                "labels": best_epoch_labels_val
            })
    # Retain the best epoch file and remove the rest for the fold
    all_model_files = [opj(args.save_dir, f'model_fold{fold + 1}_epoch{i + 1}_{args.network}_{args.img_file}.pth') for i in range(args.num_epochs)]
    best_model_file = opj(args.save_dir, f'model_fold{fold + 1}_epoch{best_epoch + 1}_{args.network}_{args.img_file}.pth')
    for model_file in all_model_files:
        if model_file != best_model_file:
            try:
                os.remove(model_file)
            except FileNotFoundError:
                print(f"File not found, skipping: {model_file}")
    print(f"Best epoch for fold {fold + 1}: {best_epoch + 1}")
    return results_tr, results_val
