# -*- coding: utf-8 -*-
"""
只是用来观察数据，不对数据进行修改
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from GLOBAL_CONF import G_TARGET,G_INDEX


"""
TARGET 字段名字
"""

TARGET=G_TARGET
app_train = pd.read_csv('process_data/1_train_data.csv')



#---------------------------------------- step 1 观察数据 ----------------------------------------#
print('数据 shape: ', app_train.shape)

#分析 TARGET 分布
#TODO
"""
判断是否 imbalanced class problem
"""
print("---------------正负样本分布--------------------")
distrubtion_0_1=app_train[TARGET].value_counts()
print (distrubtion_0_1)
app_train[TARGET].astype(int).plot.hist()
plt.show()

#分析 缺失值
def missing_values_table(df):
    """
    # 分析缺失值函数
    """
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: '缺失的个数', 1: '缺失百分比'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '缺失百分比', ascending=False).round(1)
    # Print some summary information
    print("数据一共 " + str(df.shape[1]) + " 维.\n"+
          "有 " + str(mis_val_table_ren_columns.shape[0]) +
          " 维有缺失值.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

print("---------------缺失值分布--------------------")
missing_values = missing_values_table(app_train)
print("缺失率最高的列")
print (missing_values.head(20))

#分析数据类型 Column Types
print("---------------特征的数据类型--------------------")
ColumnTypes=app_train.dtypes.value_counts()
print("tips:object为字符类型:")
print(ColumnTypes)


#---------------------------------------- step 2 探索数据 ----------------------------------------#
#相关性分析
print("--------------相关性分析--------------------")
# Find correlations with the target and sort
correlations = app_train.corr()[TARGET].sort_values()

# Display correlations
print('正相关性最高的特征：')
print(correlations.tail(10))
print('负相关性最高的特征：')
print(correlations.head(10))

#画出相关性最高的特征与样本分布的关系
import_fea=correlations.abs().sort_values().tail(11).index.values[:-1]
for i, source in enumerate(import_fea):
    # create a new subplot for each source
    # plot repaid loans
    #空值不能画图
    app_train.fillna(0)
    sns.kdeplot(app_train.loc[app_train[TARGET] == 0, source], label='target == 0')
    # plot loans that were not repaid
    sns.kdeplot(app_train.loc[app_train[TARGET] == 1, source], label='target == 1')
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source)
    plt.ylabel('Density')
    plt.show()
