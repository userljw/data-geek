# -*- coding: utf-8 -*-


"""
特征工程
1.创建新的特征  --- 创建多项式特征
2.归一化  -- lgbm 会归一化
3.特征选择
"""

import pandas as pd
from GLOBAL_CONF import G_TARGET,G_INDEX
from sklearn.preprocessing import PolynomialFeatures
import copy
#空值需要事先填充
#需要做多项式的特征
poly_feature_name=["gender","SeniorCitizen","Partner"]



train = pd.read_csv('process_data/1_train_data.csv')
test =pd.read_csv('process_data/1_predict_data.csv')
feature_category = pd.read_csv('process_data/1_feature_category.csv')

print('训练集: ', train.shape)
print('测试集:  ', test.shape)


# Make a new dataframe for polynomial features
poly_feature_name_train=copy.deepcopy(poly_feature_name)
poly_feature_name_train.append(G_TARGET)
poly_features = train[poly_feature_name_train]
poly_features_test = test[poly_feature_name]

poly_target = poly_features[G_TARGET]

poly_features = poly_features.drop(columns = [G_TARGET])

poly_transformer = PolynomialFeatures(degree = 3)

# Train the polynomial features
poly_transformer.fit(poly_features)
# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)

poly_featuresName=poly_transformer.get_feature_names(input_features = poly_feature_name)

print('特征多项式产生的特征数: ', poly_features.shape)
print('产生的特征名称：',poly_featuresName)


# Create a dataframe of the features
poly_features = pd.DataFrame(poly_features,columns = poly_featuresName)

# Add in the target
poly_features[G_TARGET] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()[G_TARGET].sort_values()
print("特征与target的相关性：")
print(poly_corrs)

# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test, columns = poly_featuresName)


# Merge polynomial features into training dataframe
poly_features[G_INDEX] = train[G_INDEX]
app_train_poly = train.merge(poly_features, on = G_INDEX, how = 'left')

# Merge polnomial features into testing dataframe
poly_features_test[G_INDEX] = test[G_INDEX]
app_test_poly = test.merge(poly_features_test, on = G_INDEX, how = 'left')

# Align the dataframes
# 对齐数据
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)

app_train_poly[G_TARGET]=train[G_TARGET]

app_train_poly.to_csv('process_data/2_train_data.csv',index=None,encoding='utf-8',line_terminator='\r\n')
app_test_poly.to_csv('process_data/2_predict_data.csv',index=None,encoding='utf-8',line_terminator='\r\n')


# Print out the new shapes
print('训练集-特征多项式之后: ', app_train_poly.shape)
print('测试集-特征多项式之后:  ', app_test_poly.shape)




"""
import featuretools as ft
es=ft.EntitySet(id="churn")
es.entity_from_dataframe(entity_id="data",dataframe=train,index=G_INDEX)
# Deep Feature Synthesis
feature_matrix, feature_names = ft.dfs(entityset=es,
                                       target_entity = 'data',
                                       n_jobs=1,
                                       verbose=1,
                                       trans_primitives= ['add_numeric', 'multiply_numeric'])
"""
