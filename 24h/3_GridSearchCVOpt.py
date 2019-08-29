# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split,GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from GLOBAL_CONF import G_TARGET,G_INDEX


train_data = pd.read_csv('process_data/1_train_data.csv')   # 读取数据

y = train_data[G_TARGET]
train_data = train_data.drop([G_TARGET, G_INDEX], axis=1) #删除 index 和 target

col = train_data.columns
x = train_data[col].values  # 剩下的列作为训练数据
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0,stratify=y)   # 分训练集和验证集，分层抽样
train = lgb.Dataset(train_x, train_y)
valid = lgb.Dataset(valid_x, valid_y, reference=train)


parameters = {
              'max_depth': [15, 20, 25, 30, 35,],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_freq': [2, 4, 5, 6, 8],
              'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
              'lambda_l2': [0, 10, 15, 35, 40],
}
gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'binary',
                         metric = 'auc',
                         learning_rate = 0.01,
                         num_leaves = 35,
                         feature_fraction=0.8,
                         bagging_fraction= 0.9,
                         bagging_freq= 8,
                         lambda_l1= 0.6,
                         lambda_l2= 0)
#有了gridsearch我们便不需要fit函数
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='roc_auc', cv=2)
gsearch.fit(train_x, train_y)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
