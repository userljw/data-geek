# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import Imputer, LabelEncoder
from GLOBAL_CONF import G_TARGET,G_INDEX


"""
同时对训练数据和预测数据进行操作
简单先做onehot labencoding fillNan

step：
1.合并训练数据和测试数据集

2.日期格式处理  默认关闭

3.针对字符类型的缺失值填充

4.one hot & label encode  默认只做label_encode ，输出1_feature_category.csv 标记哪些是类别特征

5.针对数值型的缺失值的填充 默认关闭

6.最后拆分数据，拆成1_train_data.csv / 1_predict_data.csv 
"""


train = pd.read_csv('process_data/0_train_data.csv')   #训练数据--带target
predict=pd.read_csv('process_data/0_predict_data.csv')  #预测数据--不带target


TARGET=G_TARGET  #target 的列名
INDEX=G_INDEX  #主键名


##可选项
#1. 是否包含日期类型数据
IS_Has_Datetime_col=False
date_feature_formater={}
if IS_Has_Datetime_col:
    # key=日期类型的列名 value=CSV中与之对应的日期格式
    date_feature_formater = {"recorddate_key": "%m/%d/%Y %H:%M",
                             "orighiredate_key": "%m/%d/%Y",
                             "birthdate_key": "%m/%d/%Y", }
#2.是否包针对性采用ONE_HOT
# 默认只做label encode
IS_ONE_HOT=False
if IS_ONE_HOT:
    one_hot_fea = ["FLAG_OWN_CAR", "NAME_TYPE_SUITE"]  #做one hot的列
                                                       #其他的字符类型 作 label encode

#3.是否自动填充控制  默认不填充空值
IS_FILL_NAN=True
if IS_FILL_NAN:
    num_imputer_type={} #可以针对每一个特征进行自定义的填充方式，如果是 {} 那就默认填写 0
    # num_imputer_type={"YEARS_BEGINEXPLUATATION_AVG":"median",
    #                   "YEARS_BEGINEXPLUATATION_MEDI":"mean",} #其他的填0



print("原始数据维度，训练集：",train.shape)
print("原始数据维度，预测集：",predict.shape)

#---------------------------1.合并训练数据和测试数据集-------------------------------------
# 训练集和预测集 进行合并，避免label encode的时候出现问题
__PREDICR_FLAG=-100
predict[TARGET] = __PREDICR_FLAG
all_data = pd.concat([train,predict])



#---------------------2.日期格式处理  默认关闭-------------------------------
def datetime_processing(df,date_feature_formater={}):
    """
    转换为至今的天数 或者 秒数
    format
    %m/%d/%Y
    %Y-%m-%d %H:%M:%S
    """
    # 日期数据精确到日
    for (feature,format) in date_feature_formater.items():
        df[feature] = pd.to_datetime(df[feature], format=format)
        df[feature] = pd.to_datetime("2019-01-01") - df[feature]
        df[feature] = df[feature].astype("str")
        df[feature] = df[feature].apply(lambda x: x.replace("days 00:00:00.000000000", "").replace("NaT", "0"))
        df[feature] = df[feature].astype("int")
    return df

if IS_Has_Datetime_col:
    all_data=datetime_processing(all_data,date_feature_formater)


#--------------------3.针对字符类型的缺失值填充-----------------------
"""
填充规则：
1.只对字符数据进行填充
2.数值类型不做处理，boost算法会自动处理空值的情况
"""
def onlyfill_str_col(tmp_data):
    for i, col in enumerate(tmp_data):
        if tmp_data[col].dtype == 'object':
            # flll nan with "None_by_lujw3"
            tmp_data[col] = tmp_data[col].fillna("None_by_lujw3")
    return tmp_data

all_data=onlyfill_str_col(all_data)


#--------------------one hot & label encode  默认只做label encode-----------------------
"""
编码规则:
0.优先按照列指定的方式进行编码
1.两类--》label encoding
2.大于两类 --》one hot
"""

def label_encode(df):
    # import copy
    # df=copy.deepcopy(df)
    label_encoder = LabelEncoder()
    feature_category=[]
    for i, col in enumerate(df):
        if df[col].dtype == 'object' and  df[col].name not in (TARGET,INDEX):
            df[col] = label_encoder.fit_transform(np.array(df[col].astype(str)).reshape((-1,)))
            feature_category.append(col)
    return df,feature_category


def label_and_oneHot(df,one_hot_fea=[]):
    """
    对指定列执行不同的编码方式
    """
    label_encoder = LabelEncoder()
    feature_category=[]
    if len(one_hot_fea) >=1:
        df=pd.get_dummies(df, columns=one_hot_fea)
    for i, col in enumerate(df):
        if df[col].dtype == 'object' and  df[col].name not in (TARGET,INDEX):
            df[col] = label_encoder.fit_transform(np.array(df[col].astype(str)).reshape((-1,)))
            feature_category.append(col)
    return df,feature_category


if IS_ONE_HOT:
    all_data,feature_category=label_and_oneHot(all_data,one_hot_fea=one_hot_fea)
else:
    all_data, feature_category = label_encode(all_data)




#--------------------5.针对数值型的缺失值的填充 默认关闭-----------------------
def fill_na_customize(df,num_imputer_type={}):
    """
    先根据num_imputer_type进行空值填充
    最后再对全部列空值填充，填0
    """
    if num_imputer_type:
        for (col,type) in num_imputer_type.items():
            imp =Imputer(strategy=type)
            df[col]=imp.fit_transform(df[[col]])
    df = df.fillna(0)
    return df

if IS_FILL_NAN:
    all_data=fill_na_customize(all_data,num_imputer_type=num_imputer_type)

#------------6.最后拆分数据，拆成1_train_data.csv / 1_predict_data.csv---------------
#根据第一步定义的__PREDICR_FLAG，进行拆分

data_train=all_data[all_data[TARGET]!=__PREDICR_FLAG]
data_predict=all_data[all_data[TARGET]==__PREDICR_FLAG]
data_predict=data_predict.drop([TARGET],axis=1)

data_train.to_csv('process_data/1_train_data.csv',index=None,encoding='utf-8',line_terminator='\r\n')
data_predict.to_csv('process_data/1_predict_data.csv',index=None,encoding='utf-8',line_terminator='\r\n')

# 输出--类别特征
feature_category=pd.DataFrame([feature_category],index=["fea_name"])
feature_category.to_csv('process_data/1_feature_category.csv',index=None,encoding='utf-8',line_terminator='\r\n')

print("处理后数据维度，训练集：",data_train.shape)
print("处理后数据维度，预测集：",data_predict.shape)
print("类别特征数组是：",feature_category.values)

#-----------------------是否删除该特征-----------------------------------
# TODO
IS_DEL=False
if IS_DEL:
    # 剔除含有缺失值的数据列
    all_data = all_data.dropna(axis=1)
