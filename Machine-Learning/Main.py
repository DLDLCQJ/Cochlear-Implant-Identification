
'''
classification tasks
'''
import sys
import sklearn
import numpy as np
import pandas as pd
import os
from os.path import join as opj
import typing
import warnings
from textwrap import wrap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import torch
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr, kendalltau
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin

from Transfer-Learning.utils import set_seed
from Transfer-Learning.



class ScalerPCA(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler(with_std=False)
        self.pca = PCA()

    def fit(self, X, y=None):
        # Fit the scaler
        X_scaled = self.scaler.fit_transform(X)
        # Fit PCA to determine the number of components to retain the desired variance
        self.pca.fit(X_scaled)
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components_ = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        # Fit PCA again with the selected number of components
        self.pca = PCA(n_components=self.n_components_)
        self.pca.fit(X_scaled)
        return self

    def transform(self, X, y=None):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca
    
class Preprocessor:
    def __init__(self, preprocess: typing.Union[str, bool, None] = None,
                    **kwargs) -> None:
        from sklearn.pipeline import Pipeline

        preprocessor_classes = {
            'demean': StandardScaler(with_std=False),
            'demean_std': StandardScaler(with_std=True),
            'minmax': MinMaxScaler,
            # Create pipeline for pca
            'pca_auto': ScalerPCA(variance_threshold=0.95),
            'pca10': Pipeline([('scaler', StandardScaler(with_std=False)), ('pca', PCA(n_components=10))]),
            'pca100': Pipeline([('scaler', StandardScaler(with_std=False)), ('pca', PCA(n_components=100))]),
            
            None: None
        }

        if preprocess not in preprocessor_classes:
            raise ValueError(f'Preprocess setting {preprocess} does not exist in preprocessor_classes')

        self.unfitted_scaler = preprocessor_classes[preprocess]
        self.preprocess_name = preprocess

    def fit(self, A_raw: typing.Union[pd.DataFrame, np.ndarray] = None):
        """Fit based on the input data (A_raw), return scaler. Do not transform.
        
        If the scaler does not exist, return None
        """
        if self.unfitted_scaler is not None:
            print(f'\nFitting scaler {self.unfitted_scaler}')
            fitted_scaler = self.unfitted_scaler.fit(A_raw)  # demeans column-wise (i.e. per neuroid)
        else:
            fitted_scaler = None

        return fitted_scaler

    def transform(self, scaler: typing.Union[StandardScaler, MinMaxScaler] = None,
                    A_raw: typing.Union[pd.DataFrame, np.ndarray] = None):
        """Input an array/dataframe (A_raw) and scale based on the transform fitted supplied in scaler.
        If a dateframe is input, then add indexing back after scaling
        
        If scaler is None, then return A_raw.
        
        """

        if scaler is not None:
            print(f'\nTransforming on new data using scaler {scaler}')
            A_scaled = scaler.transform(A_raw)

            if type(A_raw) == pd.DataFrame:
                if self.preprocess_name.startswith(
                        'pca'):  # If PCA, we can't add back the column names because there are now fewer columns
                    A_scaled = pd.DataFrame(data=A_scaled, index=A_raw.index)
                else:
                    A_scaled = pd.DataFrame(A_scaled, index=A_raw.index, columns=A_raw.columns)

        else:
            print(f'Scaler is None, return A_raw')
            A_scaled = A_raw

        return A_scaled
    

def statis_metrics(ground_truth, prediction):
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
    return scores_dict['accuracy'],scores_dict['roc_auc'],scores_dict['sensitivity'],scores_dict['specificity']


def load_data(path):
    import nilearn
    from nilearn import plotting
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


# Loading data
data_folder = '/Users/simon/Desktop/Melb/Final-match-Melb/MRI-results/'
ims, y = load_data(data_folder)
X = Float_MRI(ims)
X = pd.DataFrame(X)
y = pd.DataFrame(y).squeeze()
print(X.shape, y.shape,y.head())

class Mapping:
    def __init__(self,
                X: X = None,
                y: y = None,
                mapping_class: typing.Union[str, typing.Any] = None,
                metric: statis_metrics = None,
                Preprocessor: Preprocessor = None,
                preprocess_X: bool = False,
                preprocess_y: bool = False,
                random_state: int =123,
                ) -> None:
        self.X = X
        self.y = y
        self.metric = metric
        self.preprocessor = Preprocessor
        self.preprocess_X = preprocess_X
        self.preprocess_y = preprocess_y  
        ## clf 
        mapping_classes = {
            'LR': (LogisticRegression, {'penalty': ['l1', 'l2', 'elasticnet'],
                                            'C' : [1e-1, 0.0, 1.0],
                                            }),
            'Lasso': (LogisticRegression, {'penalty': ['l1'],
                                            'C' : [1e-1, 0.0, 1.0],
                                            'solver':['liblinear'],
                                            }),
            'Ridge': (LogisticRegression, {'penalty': ['l2'],
                                            'C' : [1e-1, 0.0, 1.0],
                                            'solver': ['lbfgs'],
                                            }),
            'Elastic': (LogisticRegression, {'penalty': ['elasticnet'],
                                            'C' : [1e-1, 0.0, 1.0],
                                            }),
            'SVM':(SVC, {'C': [0.1, 1], 'kernel': ['linear','rbf'], 'gamma': [0.1, 1]}),
            'RF':(RandomForestClassifier, {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6]}),
            'DT':(DecisionTreeClassifier, {'criterion':['gini', 'entropy'],'max_depth':[3,4,5]}),
                                           #'splitter':['best', 'random'],'random_state': [0,42]}),
            'KNN':(KNeighborsClassifier, {'algorithm':['auto'],
                                        'n_neighbors':[3,5,7,9],#'weights':['uniform', 'distance'],
                                        'p':[2,3,5]}),
            'Xgboost':(XGBClassifier, {'min_child_weight': [1, 5, 10],'gamma': [0.5, 1, 1.5],
                                       #'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],
                                       'max_depth': [3, 4, 5]}),

            None: None}
        self.mapping_class_name = mapping_class
        self.mapping_class = mapping_classes[mapping_class]
        if not self.mapping_class:
            raise ValueError(f'Mapping class not specified')

    def permute_X(self,
                    X: pd.DataFrame = None,
                    method: str = 'shuffle_X_rows',
                    random_state: int = 0,
                    )-> pd.DataFrame:

        print(f' !!!!! Permuting X using: {method} !!!!!')
        # Shuffle the rows of X (=shuffle the sentences and create a mismatch between the sentence embeddings and target)
        if method == 'shuffle_X_rows':
            X = X.sample(frac=1, random_state=random_state)

        # Shuffle the columns of X (=shuffle the neuroids and destroy the sentence embeddings in the same way for each sentence)
        elif method == 'shuffle_X_cols':
            X_ndarray = X.values
            np.random.seed(random_state)
            # Permute columns of X
            X_ndarray = X_ndarray[:, np.random.permutation(X_ndarray.shape[1])]
            X = pd.DataFrame(X_ndarray, columns=X.columns, index=X.index)

        elif method == 'shuffle_each_X_col':
            np.random.seed(random_state)
            for col in X.columns:
                X[col] = np.random.permutation(X[col])

        else:
            raise ValueError(f'Invalid method: {method}')

        return X
    
    def CV_score(self,
                random_state: int = 123,
                k: int = 5,
                permute_X: typing.Union[str, None] = None,
                
                ):
        
        # Classifier
        clf = self.mapping_class[0]
        params = self.mapping_class[1]
        # Regressors (X) and targets (y)
        # If ann_layer is a string (i.e., an ROI), then we need to make sure we do not end up with a Series object:
        X = self.X
        y = self.y
        # Checks: perturbing the regressors (X)
        if permute_X is not None:
            X = self.permute_X(X=X,
                            method=permute_X,
                            random_state=random_state)

        # Train/test indices
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

        train_indices = []
        test_indices = []
        scores_tr_across_folds = []  # storing the score between y_test and y_pred in each fold
        scores_te_across_folds = [] 
        alpha_tr_across_folds = []  # storing the alpha value identified in the test split in each fold
        alpha_te_across_folds = []
        # y_tests = []  # storing the y_test values in each fold (for asserting that they match up with y in the end)
        # y_preds_cv = []  # storing the y_pred values in each fold (for storing them in a dict structure with keys "y"
        # # and "y_pred-CV-k-{k}" for each fold)

        d_cv_te_log = defaultdict()
        d_cv_tr_log = defaultdict()
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            test_indices.append(test_index)
            train_indices.append(train_index)

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            print(f"X_train.shape: {X_train.shape}", f"y_train.shape: {y_train.shape}")

            # Preprocessing
            if self.preprocess_X:
                X_scaler = self.preprocessor.fit(X_train)  # Fit transform on train set to avoid data leakage
                X_train = self.preprocessor.transform(scaler=X_scaler, A_raw=X_train)
                X_test = self.preprocessor.transform(scaler=X_scaler,
                                                        A_raw=X_test)  # use transform from training set on the test set
                print(f"X_train.shape: {X_train.shape}", f"y_train.shape: {y_train.shape}")
            if self.preprocess_y:
                y_scaler = self.preprocessor.fit(y_train)  # Fit transform on train set to avoid data leakage
                y_train = self.preprocessor.transform(scaler=y_scaler, A_raw=y_train)
                y_test = self.preprocessor.transform(scaler=y_scaler, A_raw=y_test)  # use transform from training set on the test set
            ##
            grid_search = GridSearchCV(clf(),params, cv=5, scoring='accuracy',verbose=5,return_train_score=True)
            grid_search.fit(X_train, y_train)
            print('Mean test score: {}'.format(grid_search.cv_results_['mean_test_score']))
            print('Mean train score: {}'.format(grid_search.cv_results_['mean_train_score']))
            best_params = grid_search.best_params_
            best_clf = clf(**best_params)
            best_clf.fit(X_train, y_train)
            #clf.fit(X_train, y_train)
            y_predictions_tr = best_clf.predict(X_train)
            y_predictions_te = best_clf.predict(X_test)
            # save best model
            # ...
            # evaluate validation performance
            print(f"Accuracy for the fold no. {fold_idx} on the test set: {accuracy_score(y_test, y_predictions_te)}")
            print("***Performance on Validation data***")
            print(y_train.shape, y_predictions_tr.shape)
            tr_acc, tr_auc, tr_sen, tr_spe = self.metric(y_train,y_predictions_tr)
            val_acc, val_auc, val_sen, val_spe = self.metric(y_test,y_predictions_te)
            # Append scores, p-vals, y_test, y_pred to lists
            scores_tr_across_folds.append(tr_acc)
            scores_te_across_folds.append(val_acc)

            if permute_X is None:  # only assert if we do not permute the X
                assert (y_train.index == X_train.index).all()
                assert (y_test.index == X_test.index).all()
  
            df_fold_tr_log = pd.DataFrame({'CV_fold_idx': fold_idx,
                                        'CV_fold_acc': [tr_acc],
                                        'CV_fold_auc': [tr_auc],
                                        'CV_fold_sensitivity': [tr_sen],
                                        'CV_fold_specificity' : [tr_spe]
                                        },
                                       )
            
            df_fold_te_log = pd.DataFrame({'CV_fold_idx': fold_idx,
                                        'CV_fold_acc': [val_acc],
                                        'CV_fold_auc': [val_auc],
                                        'CV_fold_sensitivity': [val_sen],
                                        'CV_fold_specificity' : [val_spe]
                                        },
                                       )

            if self.mapping_class_name.startswith('ridge'):
                df_fold_te_log['alpha'] = clf.alpha_
                alpha_te_across_folds.append(clf.alpha_)
                df_fold_tr_log['alpha'] = clf.alpha_
                alpha_tr_across_folds.append(clf.alpha_)

            d_cv_te_log[fold_idx] = df_fold_te_log
            d_cv_tr_log[fold_idx] = df_fold_tr_log

        print(f'\nFinished {k} CV folds!\n')

        # Convert scores into array
        scores_tr_arr = np.array(scores_tr_across_folds)
        scores_te_arr = np.array(scores_te_across_folds)
        df_tr_scores_across_folds = pd.concat(d_cv_tr_log)
        df_te_scores_across_folds = pd.concat(d_cv_te_log)
        df_te_scores = pd.DataFrame({'CV_score_te_mean': [np.mean(scores_te_arr, axis=0)]},)
        df_tr_scores = pd.DataFrame({'CV_score_tr_mean': [np.mean(scores_tr_arr, axis=0)]},)
        return df_tr_scores, df_tr_scores_across_folds, df_te_scores, df_te_scores_across_folds



set_seed(123)
preprocessor = Preprocessor(preprocess='pca_auto') 
mapping = Mapping(X=X,
                y=y,
                mapping_class='SVM', #[Linear, SVM, RF, DT, KNN, Xgboost]
                metric=statis_metrics,
                Preprocessor=preprocessor,
                preprocess_X=True,
                preprocess_y=False,)

df_tr_scores, df_tr_scores_across_folds,df_te_scores, df_te_scores_across_folds = mapping.CV_score(k = 5,
                                                    random_state=1234,
                                                    permute_X=None, # 'shuffle_X_rows',
                                                    )

df_tr_scores, df_te_scores
df_tr_scores_across_folds, df_te_scores_across_folds

np.mean(df_tr_scores_across_folds, axis=0), np.mean(df_te_scores_across_folds, axis=0)

