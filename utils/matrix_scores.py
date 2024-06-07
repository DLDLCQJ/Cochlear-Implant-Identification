import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def compute_scores(ground_truth, prediction):
    """Computes the accuracy, sensitivity, specificity and roc"""
    tp = np.sum((prediction == 1) & (ground_truth == 1))
    tn = np.sum((prediction == 0) & (ground_truth == 0))
    fp = np.sum((prediction == 1) & (ground_truth == 0))
    fn = np.sum((prediction == 0) & (ground_truth == 1))
    # cm = confusion_matrix(labels, predictions_,labels=[0, 1])
    # tn, fp, fn, tp = cm.ravel()
    scores_dict = dict()
    scores_dict['sensitivity'] = tp / (tp + fn)
    scores_dict['specificity'] = tn / (fp + tn)
    scores_dict['precision'] = tp / (tp + fp)
    scores_dict['recall'] = tp / (tp + fn)
    #scores_dict['F1score'] = 2 * (precision * sen) / (precision + sen)
    fpr, tpr, thresholds = roc_curve(ground_truth, prediction)
    scores_dict['roc_auc'] = auc(fpr, tpr)
    scores_dict['accuracy'] = (tp+tn) / (tp + tn + fp + fn)
    return scores_dict

def compute_mean_metrics(confusion_matrices, all_truths, all_predictions):
    # Compute the mean confusion matrix
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
    print("Mean Confusion Matrix over all folds:")
    print(mean_confusion_matrix)

    # Flatten the confusion matrix
    tn, fp, fn, tp = mean_confusion_matrix.ravel()

    # Compute metrics
    sen = tp / (tp + fn)
    spe = tn / (fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1score = 2 * (precision * sen) / (precision + sen)
    
    fpr, tpr, thresholds = roc_curve(all_truths, all_predictions)
    roc_auc = auc(fpr, tpr)
    acc = (tp + tn) / (tp + tn + fp + fn)
    pos_num = (all_truths == 1).sum()
    neg_num = (all_truths == 0).sum()

    print("Sensitivity = {}".format(sen))
    print("Specificity = {}".format(spe))
    print("Precision = {}".format(precision))
    print("F1score = {}".format(F1score))
    print("Recall = {}".format(recall))
    print("ROC_AUC = {}".format(roc_auc))
    print("Accuracy = {}".format(acc))
