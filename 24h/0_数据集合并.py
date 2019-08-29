# -*- coding: utf-8 -*-
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
from GLOBAL_CONF import G_TARGET,G_INDEX

"""
数据集合并

输出
0_train_data.csv  * 包含 TARGET 列 *
0_perdict_data.csv
"""


#TODO
main = pd.read_csv('raw_data/main.csv')
append = pd.read_csv('raw_data/append.csv')
TARGET = pd.read_csv('data1/target.csv')


"""
外键 ： employee
统计范围 ： ['liked']==True
需要统计的字段 ：commentId
"""
index="employee"

#--------------------直接合并
# a=pd.merge(main,append,how="left")



#--------------------附表group by 后合并
# append[append['key']='value']
append_tmp=append[append['liked']==True].groupby(by=index,as_index=False)['commentId'].count()
append_tmp=append_tmp.rename(columns={"commentId":"append_tmp"})

data=pd.merge(main,append_tmp,how="left",on=index)



#TODO
# --------------------与TARGET 合并
# data=pd.merge(data,TARGET,how="left",on=index)

#--------------------导出合并后结果
data.to_csv('process_data/0_train_data.csv',index=None,encoding='utf-8',line_terminator='\r\n')



# TODO 对预测数据进行同样操作
data.to_csv('process_data/0_predict_data.csv',index=None,encoding='utf-8',line_terminator='\r\n')
