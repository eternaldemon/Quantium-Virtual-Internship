# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 22:16:02 2020

@author: eternal_demon
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
data = pd.read_csv('qvi_data.csv')

print(data.head())
print(data.shape)
# Checking for null values
print(data.isnull().sum())

# Adding a new column DATE_NEW having YYYYMM Format
dates = data['DATE']
data['DATE_NEW'] = pd.to_datetime(data['DATE'])
data['DATE_NEW'] = data['DATE_NEW'].dt.strftime('%Y%m')

print(data.head())

stores = [77,86,88]

totsales = data.groupby(['STORE_NBR','DATE_NEW'],as_index=False)['TOT_SALES'].sum()
ncustomers = data.groupby(['STORE_NBR','DATE_NEW'],as_index=False)['LYLTY_CARD_NBR'].nunique()
ntxnpercust = data.groupby(['STORE_NBR','DATE_NEW'],as_index=False)['TXN_ID'].nunique()/data.groupby(['STORE_NBR','DATE_NEW'],as_index=False)['LYLTY_CARD_NBR'].nunique()
nchipspertxn = data.groupby(['STORE_NBR','DATE_NEW'],as_index=False)['PROD_QTY'].nunique()/data.groupby(['STORE_NBR','DATE_NEW'],as_index=False)['TXN_ID'].nunique()
avgprice = data.groupby(['STORE_NBR','DATE_NEW'],as_index=False)['TOT_SALES'].sum()
perunit = data.groupby(['STORE_NBR','DATE_NEW'],as_index=False)['PROD_QTY'].sum()
avgpriceperunit = avgprice['TOT_SALES']/perunit['PROD_QTY']


df = totsales
df['ncustomers'] = ncustomers
df['ntxnpercust'] = ntxnpercust
df['nchipspertxn'] = nchipspertxn
df['avgpriceperunit'] = avgpriceperunit
df['totsales'] = df['TOT_SALES']
df = df.drop('TOT_SALES',axis=1)

print(df.head())
measureovertime = df
period = list(df['DATE_NEW'].unique())

allstores = df['STORE_NBR'].value_counts().tolist()

storeswithfullobs = []  

for i in range(len(allstores)):
  if allstores[i] == 12:
    storeswithfullobs.append(i+1)
    
len(storeswithfullobs)

mask = (df['DATE_NEW'] < str(201902))

pretrialmeasures = df.loc[mask]

pretrialmeasures = pretrialmeasures.loc[pretrialmeasures['STORE_NBR'].isin(storeswithfullobs)]

storenumbers = df['STORE_NBR'].unique()

#trialstores = [77,86,88]

def calculatecorr(pretrialmeasures,tnum,column):
  columns = ['STORE1','STORE2','CORR']
  corrnsales = pd.DataFrame(columns=columns)
  corrncustomers = pd.DataFrame(columns=columns)
  storenumbers = df['STORE_NBR'].unique()
  for i in storenumbers:
    store = pretrialmeasures.loc[pretrialmeasures['STORE_NBR']==i]
    trialstore = pretrialmeasures.loc[pretrialmeasures['STORE_NBR']==tnum]
    col1 = store[column]
    col2 = trialstore[column]
    if(col1.shape==col2.shape):
      r = np.corrcoef(col1,col2)
      r = r[0][1]
      if(column=='totsales'):
        corrnsales.loc[len(corrnsales)] = [tnum,i,r]
      else:
        corrncustomers.loc[len(corrncustomers)] = [tnum,i,r]
    else:
      continue
  if(column=='totsales'):
    return corrnsales 
  else:   
    return corrncustomers

def getmagnitude(df):
  column='CORR'
  #df[column] = df[column].abs()
  df['magnitude'] = ( 1 - (df[column] - df[column].min())/(df[column].max() - df[column].min()))
  return df
# Creating correlation metrics
corrnsales = calculatecorr(pretrialmeasures,77,'totsales')
corrncustomers = calculatecorr(pretrialmeasures,77,'ncustomers')
# Normalizing between 0 and 1
corrnsales = getmagnitude(corrnsales)
corrncustomers = getmagnitude(corrncustomers)

print(corrnsales.head())
print(corrncustomers.head())

# Creating score columns 
corr_weight = 0.5
corrnsales['scorensales'] = corrnsales['CORR']*corr_weight + corrnsales['magnitude']*(1-corr_weight)
corrncustomers['scorensales'] = corrncustomers['CORR']*corr_weight + corrncustomers['magnitude']*(1-corr_weight)

scorecontrol = corrnsales
scorecontrol.rename(columns={'CORR':'CORRx'},inplace=True)
scorecontrol.rename(columns={'magnitude':'magnitudex'},inplace=True)
scorecontrol['CORRy'] = corrncustomers['CORR']
scorecontrol['magnitudey'] = corrncustomers['magnitude']
scorecontrol['scorencustomers'] = corrncustomers['scorensales']
scorecontrol['controlscore'] = 0.5*scorecontrol['scorensales'] + 0.5*scorecontrol['scorencustomers']

# Top 10 Largest Values
scorecontrol['controlscore'].nlargest(10)

print(scorecontrol.loc[221])
# Choosing store 233 and 119 as the control store

# Setting control and trial stores
trialstore = 77
store1= 233
store2=119

datatrialstore = df.loc[df['STORE_NBR']==trialstore]
datacontrolstore1 = df.loc[df['STORE_NBR'] == store1]
datacontrolstore2 = df.loc[df['STORE_NBR'] == store2]

mask = datatrialstore['DATE_NEW'] < str(201903)
datatrialstore= datatrialstore.loc[mask]
mask2 = datacontrolstore1['DATE_NEW']<str(201903)
mask3 = datacontrolstore2['DATE_NEW']<str(201903)
datacontrolstore1 = datacontrolstore1.loc[mask2]
datacontrolstore2 = datacontrolstore2.loc[mask3]
print(datatrialstore.shape)
print(datacontrolstore1.shape)
print(datacontrolstore2.shape)  

# Plot for trial store vs control store 1 vs control store 2 for totsales
plt.figure(figsize=(10,10))
plt.plot(datatrialstore['DATE_NEW'],datatrialstore['totsales'],label="Trial Store 77")
plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['totsales'],label="Control Store  : 233")
plt.plot(datacontrolstore2['DATE_NEW'],datacontrolstore2['totsales'],label="Other ")
plt.ylabel('Monthly Sale')
plt.xlabel('Dates before Trial Period')
plt.legend()
plt.show()


# Plot for trial store vs control store 1 vs control store 2 for ncustomers
plt.plot(datatrialstore['DATE_NEW'],datatrialstore['ncustomers'],label="Trial Store 77")
plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['ncustomers'],label="Control Store 1 : 233")
plt.plot(datacontrolstore2['DATE_NEW'],datacontrolstore2['ncustomers'],label="Control Store 2 : 119")
plt.ylabel('Monthly No of Customers')
plt.xlabel('Dates before Trial Period')
plt.legend()
plt.show()

# Assessment of trial from period March 2019 to June 2019

cal1 = pretrialmeasures.loc[pretrialmeasures['STORE_NBR']==trialstore]
cal2 = pretrialmeasures.loc[pretrialmeasures['STORE_NBR']==store1]
cal1 = cal1.loc[cal1['DATE_NEW'] < str(201902)]
cal2 = cal2.loc[cal2['DATE_NEW'] < str(201902)]
print(cal1.shape)
print(cal2.shape)

#Scale pre‐trial control sales to match pre‐trial trial store sales

scalingfactorforcontrolsales = (cal1['totsales'].sum())/(cal2['totsales'].sum())
print(scalingfactorforcontrolsales)
measureovertimesales = measureovertime
scaledcontrolsales = measureovertimesales.loc[measureovertimesales['STORE_NBR']==store1]
# Apply the scaling factor
scaledcontrolsales['controlsales'] = scaledcontrolsales['totsales']*scalingfactorforcontrolsales
print(scaledcontrolsales)
temp = scaledcontrolsales['controlsales'].tolist()
print(temp)
ed = measureovertime.loc[measureovertime['STORE_NBR']==trialstore]
#print(ed)
extra = pd.DataFrame()
extra['DATE_NEW'] = ed['DATE_NEW']
extra['totsales'] = ed['totsales']
extra['controlsales'] = temp
print(extra)
result  = (extra['controlsales']-extra['totsales'])/extra['controlsales']
result = result*100

#Calculate the percentage difference between scaled control sales and trial sales
percentagediff = pd.DataFrame()
percentagediff['diff'] = result
percentagediff['DATE_NEW'] = extra['DATE_NEW']
stdev = percentagediff.loc[percentagediff['DATE_NEW']<str(201902)]
stdev = percentagediff['diff'].std()
print(stdev)
a = percentagediff['diff']
a = a.divide(stdev)
percentagediff['tvalue'] = a
print(percentagediff)

'''
As our null hypothesis is that the trial period is the same as the
pre‐trial period, let's take the standard deviation based on the scaled
percentage difference in the pre‐trial period'''

tmonths = percentagediff.loc[percentagediff['DATE_NEW']>str(201901)]
tmonths = tmonths.loc[tmonths['DATE_NEW']<str(201905)]

print(tmonths)

print(tmonths['tvalue'].quantile(q=0.05))
# quantile of t value is less than 

print(measureovertimesales.head())


datatrialstore = measureovertimesales.loc[measureovertimesales['STORE_NBR']==trialstore]
datacontrolstore1 = measureovertimesales.loc[measureovertimesales['STORE_NBR'] == store1]

datacontrolstore1['95quantile'] = datacontrolstore1['totsales']*(1+stdev*2)
datacontrolstore1['5quantile'] = datacontrolstore1['totsales']*(1-stdev*2)

plt.figure(figsize=(15,10))
plt.plot(datatrialstore['DATE_NEW'],datatrialstore['ncustomers'],label='Trial Store')
plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['ncustomers'],label='Control Store')
plt.xlabel('TIMELINE')
plt.ylabel('Monthly Number of Customers')
plt.legend()
plt.show()

print(datacontrolstore1['5quantile'])


#################################3
# Trial Store 86

measureovertime = df
corrnsales = calculatecorr(pretrialmeasures,86,'totsales')
corrncustomers = calculatecorr(pretrialmeasures,86,'ncustomers')
# Normalizing between 0 and 1
corrnsales = getmagnitude(corrnsales)
corrncustomers = getmagnitude(corrncustomers)

print(corrnsales.head())
print(corrncustomers.head())

# Creating score columns 
corr_weight = 0.5
corrnsales['scorensales'] = corrnsales['CORR']*corr_weight + corrnsales['magnitude']*(1-corr_weight)
corrncustomers['scorensales'] = corrncustomers['CORR']*corr_weight + corrncustomers['magnitude']*(1-corr_weight)

scorecontrol = corrnsales
scorecontrol.rename(columns={'CORR':'CORRx'},inplace=True)
scorecontrol.rename(columns={'magnitude':'magnitudex'},inplace=True)
scorecontrol['CORRy'] = corrncustomers['CORR']
scorecontrol['magnitudey'] = corrncustomers['magnitude']
scorecontrol['scorencustomers'] = corrncustomers['scorensales']
scorecontrol['controlscore'] = 0.5*scorecontrol['scorensales'] + 0.5*scorecontrol['scorencustomers']

# Top 10 Largest Values
scorecontrol['controlscore'].nlargest(10)
print(scorecontrol.loc[147])
# Choosing store 155 as the control store

# Setting control and trial stores
trialstore = 86
store1= 155

datatrialstore = df.loc[df['STORE_NBR']==trialstore]
datacontrolstore1 = df.loc[df['STORE_NBR'] == store1]
mask = datatrialstore['DATE_NEW'] < str(201903)
datatrialstore= datatrialstore.loc[mask]
mask2 = datacontrolstore1['DATE_NEW']<str(201903)
datacontrolstore1 = datacontrolstore1.loc[mask2]
print(datatrialstore.shape)
print(datacontrolstore1.shape)

# Plot for trial store vs control store 1 vs control store 2 for totsales
plt.plot(datatrialstore['DATE_NEW'],datatrialstore['totsales'],label="Trial Store 86")
plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['totsales'],label="Control Store 1 : 155")
plt.ylabel('Monthly Sale')
plt.xlabel('Dates before Trial Period')
plt.legend()
plt.show()


# Plot for trial store vs control store 1 vs control store 2 for ncustomers
plt.plot(datatrialstore['DATE_NEW'],datatrialstore['ncustomers'],label="Trial Store 86")
plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['ncustomers'],label="Control Store 1 : 155")
plt.ylabel('Monthly No of Customers')
plt.xlabel('Dates before Trial Period')
plt.legend()
plt.show()

# Assessment of trial from period March 2019 to June 2019

cal1 = pretrialmeasures.loc[pretrialmeasures['STORE_NBR']==trialstore]
cal2 = pretrialmeasures.loc[pretrialmeasures['STORE_NBR']==store1]
cal1 = cal1.loc[cal1['DATE_NEW'] < str(201902)]
cal2 = cal2.loc[cal2['DATE_NEW'] < str(201902)]
print(cal1.shape)
print(cal2.shape)

#Scale pre‐trial control sales to match pre‐trial trial store sales

scalingfactorforcontrolsales = (cal1['totsales'].sum())/(cal2['totsales'].sum())
print(scalingfactorforcontrolsales)
measureovertimesales = measureovertime
scaledcontrolsales = measureovertimesales.loc[measureovertimesales['STORE_NBR']==store1]
# Apply the scaling factor
scaledcontrolsales['controlsales'] = scaledcontrolsales['totsales']*scalingfactorforcontrolsales
print(scaledcontrolsales)
temp = scaledcontrolsales['controlsales'].tolist()
print(temp)
ed = measureovertime.loc[measureovertime['STORE_NBR']==trialstore]
#print(ed)
extra = pd.DataFrame()
extra['DATE_NEW'] = ed['DATE_NEW']
extra['totsales'] = ed['totsales']
extra['controlsales'] = temp
print(extra)
result  = (extra['controlsales']-extra['totsales'])/extra['controlsales']
result = result*100

#Calculate the percentage difference between scaled control sales and trial sales
percentagediff = pd.DataFrame()
percentagediff['diff'] = result
percentagediff['DATE_NEW'] = extra['DATE_NEW']
stdev = percentagediff.loc[percentagediff['DATE_NEW']<str(201902)]
stdev = percentagediff['diff'].std()
print(stdev)
a = percentagediff['diff']
a = a.divide(stdev)
percentagediff['tvalue'] = a
print(percentagediff)

'''
As our null hypothesis is that the trial period is the same as the
pre‐trial period, let's take the standard deviation based on the scaled
percentage difference in the pre‐trial period'''

tmonths = percentagediff.loc[percentagediff['DATE_NEW']>str(201901)]
tmonths = tmonths.loc[tmonths['DATE_NEW']<str(201905)]

print(tmonths)

print(tmonths['tvalue'].quantile(q=0.05))
# quantile of t value is less than 

print(measureovertimesales.head())

datatrialstore = measureovertimesales.loc[measureovertimesales['STORE_NBR']==trialstore]
datacontrolstore1 = measureovertimesales.loc[measureovertimesales['STORE_NBR'] == store1]

datacontrolstore1['95quantile'] = datacontrolstore1['totsales']*(1+stdev*2)
datacontrolstore1['5quantile'] = datacontrolstore1['totsales']*(1-stdev*2)

plt.plot(datatrialstore['DATE_NEW'],datatrialstore['totsales'],label='Trial Store')
plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['totsales'],label='Control Store')
#plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['5quantile'].quantile(0.05),label='Control store 5th percetile')
#plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['95quantile'].quantile(0.95),label='Control store 95th percetile')
plt.ylabel('Monthly Sale')
plt.legend()
plt.show()

print(datacontrolstore1['5quantile'])


##################################################
# TRIAL STORE 88
###################################################


measureovertime = df
corrnsales = calculatecorr(pretrialmeasures,88,'totsales')
corrncustomers = calculatecorr(pretrialmeasures,88,'ncustomers')
# Normalizing between 0 and 1
corrnsales = getmagnitude(corrnsales)
corrncustomers = getmagnitude(corrncustomers)

print(corrnsales.head())
print(corrncustomers.head())

# Creating score columns 
corr_weight = 0.5
corrnsales['scorensales'] = corrnsales['CORR']*corr_weight + corrnsales['magnitude']*(1-corr_weight)
corrncustomers['scorensales'] = corrncustomers['CORR']*corr_weight + corrncustomers['magnitude']*(1-corr_weight)

scorecontrol = corrnsales
scorecontrol.rename(columns={'CORR':'CORRx'},inplace=True)
scorecontrol.rename(columns={'magnitude':'magnitudex'},inplace=True)
scorecontrol['CORRy'] = corrncustomers['CORR']
scorecontrol['magnitudey'] = corrncustomers['magnitude']
scorecontrol['scorencustomers'] = corrncustomers['scorensales']
scorecontrol['controlscore'] = 0.5*scorecontrol['scorensales'] + 0.5*scorecontrol['scorencustomers']

# Top 10 Largest Values
scorecontrol['controlscore'].nlargest(10)
print(scorecontrol.loc[170])  # Store 178
print(scorecontrol.loc[225])   # Store   
# Choosing store 178 and 237 as the control store

# Setting control and trial stores
trialstore = 86
store1= 178
store2 = 225

datatrialstore = df.loc[df['STORE_NBR']==trialstore]
datacontrolstore1 = df.loc[df['STORE_NBR'] == store1]
mask = datatrialstore['DATE_NEW'] < str(201903)
datatrialstore= datatrialstore.loc[mask]
mask2 = datacontrolstore1['DATE_NEW']<str(201903)
datacontrolstore1 = datacontrolstore1.loc[mask2]
mask3 = datacontrolstore2['DATE_NEW']<str(201903)
datacontrolstore2 = datacontrolstore2.loc[mask3]
print(datatrialstore.shape)
print(datacontrolstore1.shape)
print(datacontrolstore2.shape)

# Plot for trial store vs control store 1 vs control store 2 for totsales
plt.plot(datatrialstore['DATE_NEW'],datatrialstore['totsales'],label="Trial Store 86")
plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['totsales'],label="Control Store 1 : 178")
plt.plot(datacontrolstore2['DATE_NEW'],datacontrolstore2['totsales'],label="Control Store 2 : 237")
plt.ylabel('Monthly Sale')
plt.xlabel('Dates before Trial Period')
plt.legend()
plt.show()


# Plot for trial store vs control store 1 vs control store 2 for ncustomers
plt.plot(datatrialstore['DATE_NEW'],datatrialstore['ncustomers'],label="Trial Store 86")
plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['ncustomers'],label="Control Store 1 : 178")
plt.plot(datacontrolstore2['DATE_NEW'],datacontrolstore2['ncustomers'],label="Control Store 2 : 237")
plt.ylabel('Monthly No of Customers')
plt.xlabel('Dates before Trial Period')
plt.legend()
plt.show()

# Assessment of trial from period March 2019 to June 2019

cal1 = pretrialmeasures.loc[pretrialmeasures['STORE_NBR']==trialstore]
cal2 = pretrialmeasures.loc[pretrialmeasures['STORE_NBR']==store1]
cal1 = cal1.loc[cal1['DATE_NEW'] < str(201902)]
cal2 = cal2.loc[cal2['DATE_NEW'] < str(201902)]
print(cal1.shape)
print(cal2.shape)

#Scale pre‐trial control sales to match pre‐trial trial store sales

scalingfactorforcontrolsales = (cal1['totsales'].sum())/(cal2['totsales'].sum())
print(scalingfactorforcontrolsales)
measureovertimesales = measureovertime
scaledcontrolsales = measureovertimesales.loc[measureovertimesales['STORE_NBR']==store1]
# Apply the scaling factor
scaledcontrolsales['controlsales'] = scaledcontrolsales['totsales']*scalingfactorforcontrolsales
print(scaledcontrolsales)
temp = scaledcontrolsales['controlsales'].tolist()
print(temp)
ed = measureovertime.loc[measureovertime['STORE_NBR']==trialstore]
#print(ed)
extra = pd.DataFrame()
extra['DATE_NEW'] = ed['DATE_NEW']
extra['totsales'] = ed['totsales']
extra['controlsales'] = temp
print(extra)
result  = (extra['controlsales']-extra['totsales'])/extra['controlsales']
result = result*100

#Calculate the percentage difference between scaled control sales and trial sales
percentagediff = pd.DataFrame()
percentagediff['diff'] = result
percentagediff['DATE_NEW'] = extra['DATE_NEW']
stdev = percentagediff.loc[percentagediff['DATE_NEW']<str(201902)]
stdev = percentagediff['diff'].std()
print(stdev)
a = percentagediff['diff']
a = a.divide(stdev)
percentagediff['tvalue'] = a
print(percentagediff)

'''
As our null hypothesis is that the trial period is the same as the
pre‐trial period, let's take the standard deviation based on the scaled
percentage difference in the pre‐trial period'''

tmonths = percentagediff.loc[percentagediff['DATE_NEW']>str(201901)]
tmonths = tmonths.loc[tmonths['DATE_NEW']<str(201905)]

print(tmonths)

print(tmonths['tvalue'].quantile(q=0.95))
# quantile of t value is less than 

#print(measureovertimesales.head())

datatrialstore = measureovertimesales.loc[measureovertimesales['STORE_NBR']==trialstore]
datacontrolstore1 = measureovertimesales.loc[measureovertimesales['STORE_NBR'] == store1]
datacontrolstore2 = measureovertimesales.loc[measureovertimesales['STORE_NBR'] == store2]

datacontrolstore1['95quantile'] = datacontrolstore1['totsales']*(1+stdev*2)
datacontrolstore1['5quantile'] = datacontrolstore1['totsales']*(1-stdev*2)
datacontrolstore2['95quantile'] = datacontrolstore2['totsales']*(1+stdev*2)
datacontrolstore2['5quantile'] = datacontrolstore2['totsales']*(1-stdev*2)

plt.plot(datatrialstore['DATE_NEW'],datatrialstore['totsales'],label='Trial Store')
plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['totsales'],label='Control Store 1')
plt.plot(datacontrolstore2['DATE_NEW'],datacontrolstore2['totsales'],label='Control Store 2')
#plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['5quantile'].quantile(0.05),label='Control store 5th percetile')
#plt.plot(datacontrolstore1['DATE_NEW'],datacontrolstore1['95quantile'].quantile(0.95),label='Control store 95th percetile')
plt.ylabel('Monthly Sale')
plt.legend()
plt.show()

print(datacontrolstore1['5quantile'])
print(datacontrolstore2['5quantile'])

