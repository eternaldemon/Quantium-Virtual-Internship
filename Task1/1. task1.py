# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:50:36 2020

@author: eternal_demon
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# data2 -> Transaction Dataset
# data1 -> Customer Dataset
data1 = pd.read_csv('QVI_purchase_behaviour.csv')
data2 = pd.read_excel('QVI_transaction_data.xlsx')

print(data1.head())
print(data2.head())

# Checking for null values
print(data1.isnull().sum())
print(data2.isnull().sum())


# taking a look at transaction dataset data2
# Getting Info regarding data2
print(data2.shape)
print(data2.columns)
print(data2.info())

# Converting date from integer to date format
def xldate_to_datetime(xldate):
   tempDate = datetime(1900, 1, 1)
   deltaDays =timedelta(days=int(xldate)-2)
   TheTime = (tempDate + deltaDays )
   return TheTime.strftime("%m/%d/%Y")
dates = data2['DATE']
dates = list(dates)
dates_new=[]
for date in dates:
  dates_new.append(xldate_to_datetime(date))
  
dates_new = np.reshape(np.array(dates_new),(-1,1))
#print(dates_new.shape)

data2['DATE'] = dates_new

print(data2.head())

# Checking for outliers
print(data2.describe())

# Outlier in Prod_qty and Tot_sales 

outlier = list((np.where(data2['PROD_QTY'] == 200.0)))

# Getting outlier index
#print(outlier)
#print(outlier[0].shape)
outlier = list(outlier[0])
print(outlier)

# Removing Outlier since there are only 2
#print(data2.shape)
data2 = data2.drop(outlier,axis=0)
#print(data2.shape)

# Data Analysis for column PROD_NAME
data2['PROD_NAME'].describe() 
sns.countplot(data2['PROD_NAME'],label='Count')

data2['PROD_NAME'].value_counts().sort_values(ascending=False)
# 114 unique values with highest frequency of 3304 

# Counting number of transactions by date
# 364 unique dates
len(data2['DATE'].unique())

# Plotting transactions by date 
data2.groupby(['DATE'])['TOT_SALES'].sum().plot(figsize=(10,10))

# Getting pack size
temp = pd.DataFrame(data2['PROD_NAME'])
temp = np.array(temp)
res = []
for value in temp:
  value = str(value)
  ans = int("".join(filter(str.isdigit, value)))
  res.append(ans)
#Adding PACK_SIZE TO DATASET
data2['PACK_SIZE'] = res
print(data2.head())

res = set(res)
#print(len(res))
#print(res)

# Smallest and largest pack size
min_packsize = min(res)
max_packsize = max(res)
# print(min_packsize)
# print(max_packsize)


# Plotting Pack size wrt to total sales
data2.groupby(data2['PACK_SIZE'])['TOT_SALES'].sum().plot(kind='bar')

# Getting Brand Name into the columns BRAND
data2['BRAND'] = data2['PROD_NAME'].str.split(' ').str[0]
#print(data2.head())

# Total brands and bar chart
print('Total Brands in dataset :',len(data2['BRAND'].unique()))
data2['BRAND'].describe()

data2.groupby(data2['BRAND'])['TOT_SALES'].count().plot(kind='bar')

# Date vs Sales Plot
data2.groupby([data2['DATE']])['TOT_SALES'].count().plot(figsize=(15,15))

# Examining Customer Dataset 

print(data1.head())
# Checking for Null values
data1.isnull().sum()

# Unique Lifestages, Loyalty Card numbers and Customer Types
print("Unique LIFESTAGES :", len(data1['LIFESTAGE'].unique()))
print(data1['LIFESTAGE'].unique())
print("Unique Loyalty Card Numbers :", len(data1['LYLTY_CARD_NBR'].unique()))
print(data1['LYLTY_CARD_NBR'].unique())
print("Unique Customer Types :", len(data1['PREMIUM_CUSTOMER'].unique()))
print(data1['PREMIUM_CUSTOMER'].unique())

# No of counts for different LIFESTAGES and Customer Types
print(data1['LIFESTAGE'].value_counts().sort_values(ascending=False))
print(data1['PREMIUM_CUSTOMER'].value_counts().sort_values(ascending=False))
print(data2.columns)

# Searching for Outliers
data1['LIFESTAGE'].describe()
data1.describe()
data1['PREMIUM_CUSTOMER'].describe()

# Merging the transaction dataset with customer dataset
data = pd.merge(data2,data1,on='LYLTY_CARD_NBR',how='outer',indicator=False)
print(data.head())
# Checking for null values
print(data.isnull().sum())
# Dropping na values since there is only one
data.dropna(axis=0,inplace=True)
print(data.isnull().sum())
print(data.shape)

data.groupby(['LIFESTAGE','PREMIUM_CUSTOMER'])['TOT_SALES'].count().plot(figsize=(50,10),kind='bar',color='green')

data.groupby(['LIFESTAGE','BRAND'])['TOT_SALES'].count().plot(figsize=(50,10),kind='bar')
# Premium Customer vs LIFESTAGE
data.groupby(['LIFESTAGE']).count().plot(kind='bar')
# brand purchase with respect to Cutomer Type
fig = data.groupby(['BRAND'])['PREMIUM_CUSTOMER'].count().plot(kind='bar')
fig.set_ylabel('CUSTOMER')
#No of customers in each category 
sns.countplot(data['PREMIUM_CUSTOMER'],label="No of customers in each category")
#sns.catplot(data=data,kind='bar')

#plt.plot(data['TOT_SALES'],data['LIFESTAGE'])
# Saving dataset
temp = pd.DataFrame()
temp['TOT_SALES'] = data['TOT_SALES']
temp['PREMIUM_CUSTOMER'] = data['PREMIUM_CUSTOMER']
temp['LIFESTAGE'] = data['LIFESTAGE']
print(temp.columns)
print(temp.shape)
print(temp.head())
'''
fig1 = temp['TOT_SALES'].groupby([temp['LIFESTAGE'],temp['PREMIUM_CUSTOMER']]).mean()
fig1.plot(kind='bar')
'''
temp.set_index('LIFESTAGE').plot(kind='bar',stacked=True)
data.to_csv('latest.csv',index=False)
temp[['PREMIUM_CUSTOMER','TOT_SALES']].plot(x="LIFESTAGE",kind='bar')

temp.groupby(['LIFESTAGE','PREMIUM_CUSTOMER'])['TOT_SALES'].mean().unstack().plot(kind='bar',figsize=(12,10))
plt.xticks(rotation=0)
plt.ylabel("AVG UNITS PER TRANSACTION")
plt.autoscale(True)
plt.legend()
plt.tight_layout()
plt.show()


df = temp.groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).size().unstack()
print(df)
df.plot(kind='bar',stacked=True,figsize=(15,10))
plt.xticks(rotation=0)
plt.ylabel("SUM OF TRANSACTIONS")
plt.autoscale(True)
plt.legend()
plt.tight_layout()
plt.show()

