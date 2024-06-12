import sys
import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from utils.matrix_scores import compute_scores

results =[]
for i in range(args.k):
    with open(opj(args.save_dir,f"model_fold{i+1}_cv_predicts_and_labels_val_{args.network}_{args.img_file}.pkl"), "rb") as f:
        result = pickle.load(f)
    results.append(result)
confusion_matrices = []
# Iterate over the results to compute confusion matrices for each fold
for fold_result in results:
    predictions = fold_result[-1]["predictions"]
    # predictions = [tensor.detach().cpu().numpy() for tensor in predictions]
    # predictions = np.concatenate(predictions).tolist()
    predictions_ = (1 / (1 + np.exp(-np.array(predictions))) > 0.5).astype(float)
    labels = fold_result[-1]["labels"]

    # Compute confusion matrix
    # labels = [tensor.detach().cpu().numpy() for tensor in labels]
    # labels = np.concatenate(labels).tolist()
    labels = np.array(labels)
    predictions_ = np.array(predictions_)
    print(labels.shape, predictions_.shape)
    cm = confusion_matrix(labels, predictions_,labels=[0, 1])
    confusion_matrices.append(cm)
    # Print the confusion matrix for the fold
    print(f"Confusion Matrix for Fold {fold_result[-1]['fold']}:")
    print(cm)
    # Print other evaluation indexs
    scores_dict = compute_scores(labels, predictions_)
    print ("Sensitivity = {}".format(scores_dict['sensitivity']))
    print ("Specificity = {}".format(scores_dict['specificity']))
    print ("ROC_AUC = {}".format(scores_dict['roc_auc']))
    print("Accuracy = {}".format(scores_dict['accuracy']))

## Optional: Compute the mean confusion matrix over all folds
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
print("Mean Confusion Matrix over all folds:")
print(mean_confusion_matrix)
# Print other evaluation indexs
tn, fp, fn, tp = mean_confusion_matrix.ravel()
sen = tp / (tp + fn)
spe = tn / (fp + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
F1score = 2 * (precision * sen) / (precision + sen)
fpr, tpr, thresholds = roc_curve(labels, predictions)
roc_auc = auc(fpr, tpr)
acc = (tp+tn) / (tp + tn + fp + fn)
pos_num = (labels==1).sum().item()
neg_num = (labels==0).sum().item()
print ("Sensitivity = {}".format(sen))
print ("Specificity = {}".format(spe))
print ("Precision = {}".format(precision))
print ("F1score = {}".format(F1score))
print ("Recall = {}".format(recall))
print ("ROC_AUC = {}".format(roc_auc))
print("Accuracy = {}".format(acc))
