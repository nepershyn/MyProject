
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, mean_squared_error, r2_score
from copy import deepcopy
import matplotlib.pylab as plt
import cPickle as pickle


################################################

### ф-ция проверки отклонения доли налов и среднего по предикторам в разрезе срезов

def check_avg(data, period_field, delta=20, fields = None):
    """
    Check average value deviation throughout sections 

    Parameters
    ----------
    data : pandas.DataFrame
        data to check
    period_field : str
        Section date field name
    delta : float
        Percentage of difference between sections
    fields : array, optional, default 'None' 
        List witn columns names
    
    Returns
    -------
    DataFrame
        Initial dataframe excluding outlier fields
    DataFrame
        Result table for outliers

    """
    outliers = []
    table = pd.DataFrame()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if fields == None:
        fields = data.columns

    for field in data[fields].select_dtypes(include=numerics):
        c = pd.DataFrame(data={
                'Count': data[field].groupby(data[period_field]).size(), 
                'CountNull':data[field][pd.isnull(data[field])].groupby(data[period_field]).size(),
                'Mean' : (data[field].groupby(data[period_field]).agg('mean')).round(decimals=3)
            })
        c.insert(0, 'Field', field)
        c['MeanChange(%)'] = (c.Mean.pct_change()*100).round(decimals=1)
        if any(c['MeanChange(%)'].abs()>delta):
            outliers.append(field)
            table = pd.concat([table, c], axis = 0)
    return data[list(set(data.columns)-set(outliers))], table

def check_null(data, period_field, delta=20, fields = None):
    """
    Check null percentage deviation throughout sections 

    Parameters
    ----------
    data : pandas.DataFrame
        data to check
    period_field : str
        Section date field name
    delta : float
        Percentage of difference between sections
    fields : array, optional, default 'None' 
        List witn columns names
    
    Returns
    ------- 
    DataFrame
        Initial dataframe excluding outlier fields
    DataFrame
        Result table for outliers

    """
    outliers = []
    table = pd.DataFrame()
    if fields == None:
        fields = data.columns

    for field in set(data[fields])-{period_field}:
        c = pd.DataFrame(data={
                'Count': data[[field]].groupby(data[period_field]).size(), 
                'CountNull':data[field][pd.isnull(data[field])].groupby(data[period_field]).size()
            })
        c.insert(0, 'Field', field)
        c['NullPercent']=c['CountNull']/c['Count'] * 100
        c['NullPercentChange(%)'] = (c.NullPercent.pct_change()*100).round(decimals=1)
        if any(c['NullPercentChange(%)']>delta):
            outliers.append(field)
            table = pd.concat([table, c], axis = 0)
    return data[list(set(data.columns)-set(outliers))], table

##################################################

### Предпроцессинг
#Проверяю количество непустых значений в каждом поле (кроме тех, которые не явлются предикторами).

#Удаляю поле из базы:
#- количество НЕпустых == 0
#- имеют одно уникальное значение
#- заполнены менее чем на 2%


def predproc_data(data, features, emptiness=0.02):
    """
    Check data emptiness throughout fields 

    Parameters
    ----------
    data : pandas.DataFrame
        data to check
    features : list
        List with features names
    emptiness : float, default = 0.02
        Percentage of data emptiness in range [0,1]
    
    Returns
    ------- 
    DataFrame
        Initial dataframe excluding deleted fields
    list
        List with deleted fields names

    """
    if emptiness > 1 or emptiness < 0:
         raise ValueError('emptiness must be in range [0,1]')

    items = {'del_items_NAs':[], 'del_items_les2':[], 'del_items_1uniq':[]}
    for field in data[features]:
        if (data[field].nunique(dropna=True)==0) :
            items['del_items_NAs'].append(field)
        elif data[field].nunique(dropna=False)==1:
            items['del_items_1uniq'].append(field) #исключаем  поля имеющие одно уникальное значение
        elif float(data[field].count())/data.shape[0]<emptiness:
            items['del_items_les2'].append(field) #исключаем  поля заполненные менее чем на 2% 

    print("Deleted %s empty fields."%(sum(len(v) for v in items.itervalues())))
    print "Details about delated item:"
    if len(items['del_items_NAs']) > 0:
        print ("    Deleted because all items are NAS - %s"%(len(items['del_items_NAs'])))
        print items['del_items_NAs']
    if len(items['del_items_les2']) > 0:
        print ("    Deleted because items filled with less than %.2f percent - %s"%(emptiness*100, len(items['del_items_les2'])))
        print items['del_items_les2']
    if len(items['del_items_1uniq']) > 0:
        print ("    Deleted because items have 1 unique value - %s"%(len(items['del_items_1uniq'])))
        print items['del_items_1uniq']

    del_items = np.concatenate(np.array(items.values()), axis=0)

    return data[list(set(features) - set(del_items))], del_items



################################################

### ф-ция проверки отклонения доли налов и среднего по предикторам для нового среза

def check_period_avg(data_old, data_new, period_field, delta=20, fields = None):
    """
    Check average value percentage deviation between initial data and new period 

    Parameters
    ----------
    data_old : pandas.DataFrame
        initial daata for model
    data_new : pandas.Series
        new section to check
    period_field : str
        Section date field name
    delta : float
        Percentage of difference between sections
    fields : array, optional, default 'None' 
        List witn columns names
    
    Returns
    ------- 
    DataFrame
        Result table for outliers

    """
    table = pd.DataFrame()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if fields == None:
        fields = data_old.columns
    count = data_old.groupby(data_old[period_field]).size().mean()
    count_null = data_old.isnull().groupby(data_old[period_field]).sum().mean()
    initial_avg = data_old.groupby(data_old[period_field]).mean().mean()

    for field in data_old[fields].select_dtypes(include=numerics):
        c = pd.DataFrame(data={
                'Field': field,
                'Count': data_new[field].shape[0], 
                'CountNull':data_new[field].isnull().sum(),
                'Mean' : data_new[field].mean()
            }, index=['new'])
        #c['MeanChange'] = ((c.New_Mean-c.Initial_Mean)/c.Initial_Mean)*100
        f = pd.DataFrame(data={
                'Field': field,
                'Count': count, 
                'CountNull':count_null[field],
                'Mean': initial_avg[field]
            }, index = ['old'])
        c['MeanChange'] = (c.Mean[0]-f.Mean[0])/f.Mean[0]*100
        f['MeanChange'] = None
        if abs(c.MeanChange[0])>delta:
            table = pd.concat([table, f, c], axis = 0)
    return table

def check_period_null(data_old, data_new, period_field, delta=20, fields = None):
    """
    Check null percentage deviation between initial data and new period 

    Parameters
    ----------
    data_old : pandas.DataFrame
        initial daata for model
    data_new : pandas.Series
        new section to check
    period_field : str
        Section date field name
    delta : float
        Percentage of difference between sections
    fields : array, optional, default 'None' 
        List witn columns names
    
    Returns
    ------- 
    DataFrame
        Result table for outliers

    """
    table = pd.DataFrame()
    if fields == None:
        fields = data_old.columns
    count = data_old.groupby(data_old[period_field]).size().mean()
    count_null = data_old.isnull().groupby(data_old[period_field]).sum().mean()
    null_rate = (data_old.isnull().groupby(data_old[period_field]).sum()).divide(data_old.groupby(data_old[period_field]).size(), axis=0).mean()
  
    for field in fields: # set(null_rate[fields])-{period_field}:
        c = pd.DataFrame(data={
                'Field': field,
                'Count': data_new[field].shape[0], 
                'CountNull':data_new[field].isnull().sum()
            }, index = ['new'])
        c['Null_rate']=c['CountNull']/c['Count'] * 100
        f = pd.DataFrame(data={
                'Field': field,
                'Count': count, 
                'CountNull':count_null[field],
                'Null_rate': null_rate[field]*100
            }, index = ['old'])
        c['NullChange'] = (c.Null_rate[0]-f.Null_rate[0])/f.Null_rate[0]*100
        f['NullChange'] = None
        if (abs(c.NullChange[0])>delta) & ((c.Null_rate[0] > 1) | (f.Null_rate[0] > 1)):
            table = pd.concat([table, f, c], axis = 0)
    return table


def check_period(data_old, data_new, period_field, delta_null=20, delta_avg=20, fields = None):
    """
    Check null and mean percentage deviation between initial data and new period 

    Parameters
    ----------
    data_old : pandas.DataFrame
        initial daata for model
    data_new : pandas.Series
        new section to check
    period_field : str
        Section date field name
    delta_null : float
        Percentage of difference in null rate between sections
    delta_avg : float
        Percentage of difference in mean between sections
    fields : array, optional, default 'None' 
        List witn columns names
    
    Returns
    ------- 
    DataFrame
        Result table for outliers

    """
    null = check_period_null(data_old, data_new, period_field, delta=delta_null, fields = fields)
    avg = check_period_avg(data_old, data_new, period_field, delta=delta_avg, fields = fields)

    null.set_index(null.Field + null.index, drop=False, append=True, inplace=True)
    null.reset_index(0, drop=True, inplace=True)
    avg.set_index(avg.Field + avg.index, drop=False, append=True, inplace=True)
    avg.reset_index(0, drop=True, inplace=True)

    return pd.merge(avg, null, how='outer')
    





##############################################################
# Замена налов  по числовім переменным
#Делаем замену пропусков по следующему алгоритму:

#- если область значений переменных >=0 заменяем на -1
#- если область значений переменных <= 0 заменяем на 1
#- если переменные принимают замения от - до 0 заменяем на среднее значение по выборке

def fillna_num_fit(data, num_fields):
    all_upto_zero=[]
    two_sites_zero=[]
    all_more_zero=[]
    error_list=[]
    for field in data[num_fields]:
        if (np.min(data[field]) <0) & (np.max(data[field]) <=0) :
            data[field] = data[field].fillna(1)
            all_upto_zero.append(field)
        elif (np.min(data[field]) <0) & (np.max(data[field]) >0) :
            data[field] = data[field].fillna(data[field].mean())
            two_sites_zero.append(field)
        elif (np.min(data[field]) >=0) & (np.max(data[field]) >0) :
            data[field] = data[field].fillna(-1)
            all_more_zero.append(field)
        elif (np.min(data[field])==0) & (np.max(data[field]) ==0) :
            data[field] = data[field].fillna(-1)
            all_more_zero.append(field)
        else: 
            error_list.append(field)

    file='fillna_num_info.txt'
    with open(file,'wb') as output:
        pickle.dump((all_upto_zero, two_sites_zero, all_more_zero),output, 2)


    print "Кол-во предикторов, все значения которых меньше нуля =", len(all_upto_zero)
    print "Кол-во предикторов, которые имеют значения от - до + =", len(two_sites_zero)
    print "Кол-во предикторов, все значения которых больше нуля =", len(all_more_zero)
    print "Кол-во предикторов, которые не попали под интервалы =", len(error_list)
    print "Справочинки - all_more_zero, two_sites_zero, all_upto_zero"

    
def fillna_num_pred(data, num_fields):
    with open('fillna_num_info.txt','rb') as input:
        (all_upto_zero, two_sites_zero, all_more_zero) = pickle.load(input)
        
    data[all_upto_zero] = data[all_upto_zero].fillna(1)
    data[two_sites_zero] = data[two_sites_zero].fillna(data[two_sites_zero].mean())
    data[all_more_zero] = data[all_more_zero].fillna(-1)
    print "Справочинки - all_more_zero, two_sites_zero, all_upto_zero"