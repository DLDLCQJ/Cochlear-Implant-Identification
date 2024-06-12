import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import roc_curve, auc

def Plot_Confusion_Matrix(conf_matrix, class_names=None,cmap=plt.cm.Blues):
    if class_names is None:
        class_names = ['Class1', 'Class2']
    ##
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

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


