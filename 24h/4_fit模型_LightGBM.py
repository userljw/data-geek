# -*- coding: utf-8 -*-
# Based on excellent script by @olivier
# https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm
# StratifiedKFold instead of KFold
# LightGBM parameters found by Bayesian optimization ( https://github.com/fmfn/BayesianOptimization )

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from GLOBAL_CONF import G_TARGET,G_INDEX



def build_model_input(train_file,test_file):
    # data = pd.read_csv('../input/application_train.csv')
    # test = pd.read_csv('../input/application_test.csv')
    # print('Shapes : ', data.shape, test.shape)
    #
    # y = data[G_TARGET]
    # ids = data[G_INDEX]
    # del data[G_TARGET]
    #
    # categorical_feats = [f for f in data.columns if data[f].dtype == 'object']
    # for f_ in categorical_feats:
    #     data[f_], indexer = pd.factorize(data[f_])
    #     test[f_] = indexer.get_indexer(test[f_])
    # return data, test, y, ids

    data=pd.read_csv(train_file)
    test=pd.read_csv(test_file)


    y = data[G_TARGET]
    ids = data[G_INDEX]
    del data[G_TARGET]


    return data, test, y, ids


def train_model(data_, test_, y_, folds_):

    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in data_.columns if f not in [G_INDEX]]

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]

        # LightGBM parameters found by Bayesian optimization
        params_opt={'bagging_fraction': 0.8239310472305381,
             'feature_fraction': 0.7082683453739915,
             'lambda_l1': 2.0405624971470133,
             'lambda_l2': 0.17922086322684405,
             'max_depth': 5,
             'min_child_weight': 49.9478079532173,
             'min_split_gain': 0.07105551524342339,
             'num_leaves': 25,
             'application': 'binary',
             'num_iterations': 1000,
             'learning_rate': 0.05,
             'early_stopping_round': 100,
             'metric': 'auc'}

        clf = LGBMClassifier(**params_opt)

        clf.fit(
            trn_x,
            trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric='auc',
            verbose=100,
            early_stopping_rounds=100 , #30
        )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats],
            num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))

    test_[G_TARGET] = sub_preds

    df_oof_preds = pd.DataFrame({G_INDEX:ids, G_TARGET:y, 'PREDICTION':oof_preds})
    df_oof_preds = df_oof_preds[[G_INDEX, G_TARGET, 'PREDICTION']]

    return oof_preds, df_oof_preds, test_[[G_INDEX, G_TARGET]], feature_importance_df, roc_auc_score(y, oof_preds)

def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(
            by="importance", ascending=False)[:50].index

    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances-01.png')


def display_roc_curve(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot(
        [0, 1], [0, 1],
        linestyle='--',
        lw=2,
        color='r',
        label='Luck',
        alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(
        fpr,
        tpr,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig('roc_curve-01.png')


def display_precision_recall(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6, 6))

    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(
        precision,
        recall,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig('recall_precision_curve-01.png')


if __name__ == '__main__':
    gc.enable()
    # Build model inputs

    train_file='process_data/1_train_data.csv'
    test_file='process_data/1_predict_data.csv'


    data, test, y, ids = build_model_input(train_file=train_file,test_file=test_file)
    # Create Folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)
    # Train model and get oof and test predictions
    oof_preds, df_oof_preds, test_preds, importances, score = train_model(data, test, y, folds)
    # Save test predictions
    now = datetime.now()
    score = str(round(score, 6)).replace('.', '')
    sub_file = 'submission_5x-LGBM-average' + score + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
    test_preds.to_csv(sub_file, index=False)
    # #oof_file = 'train_5x-LGB-run-01-v1-oof_' + score + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
    # #df_oof_preds.to_csv(oof_file, index=False)

    # Display a few graphs
    folds_idx = [(trn_idx, val_idx)
                 for trn_idx, val_idx in folds.split(data, y)]
    display_importances(feature_importance_df_=importances)
    display_roc_curve(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)
    display_precision_recall(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)
