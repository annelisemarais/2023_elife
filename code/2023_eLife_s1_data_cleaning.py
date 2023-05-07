"""
author = 'Anne-Lise Marais, see annelisemarais.github.io'
publication = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old. eLife'
corresponding author = 'nadege.roche@unicaen.fr'


This upsamples and cleans data.
"""

print(__doc__)


from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import random
import seaborn as sns

#################
##LOADING DATA
#################

df_rawdata = pd.read_csv("data/df_rawdata.csv", index_col=0, header=0)

string_cols = df_rawdata.columns[-4:].astype(str).tolist()
int_cols = df_rawdata.columns[:1000].astype(int).tolist()
df_rawdata.columns = int_cols + string_cols
#

#################
###UPSAMPLING
#################

#Resolve class imbalance  by upsampling six years old

#save data of two and four years old in a variable
df_data =df_rawdata.drop(df_rawdata[df_rawdata['age']==6].index)

#Random choice of subject number

mylist = list(range(25,31))
random.seed(1)
upsample = random.choices(mylist,k=12)
#output = [25, 30, 29, 26, 27, 27, 28, 29, 25, 25, 30, 27]

#Get data of the subject randomly chosen
sixyo_upsampled = pd.DataFrame(columns=df_rawdata.columns)
for sub in upsample:
  sixyo_upsampled = pd.concat([sixyo_upsampled,df_rawdata[df_rawdata['sub']==sub]])
  
#update subject number
sub = []
for i in range(25,37): 
    new_sub = [i] * 129* 7 #129 EEG channels and 7 conditions
    sub += new_sub

sixyo_upsampled['sub']=sub

#append upsampled six years old subjects to other subjects
df_data = df_data.append(sixyo_upsampled).reset_index(drop=True)

#save data
#df_data.to_csv("data/df_data_upsampled.csv", index=True, header=True)

#################
##Remove bad channels and eye channels
#################

###create a standardized df
df_data_std = df_data.drop(['condition', 'electrode', 'sub','age'], axis=1)
df_standardized = (df_data - np.mean(df_data)) / np.std(df_data)

#find channel with data > 3 std
df_badchan = df_standardized[df_standardized>3]
df_badchan = df_badchan.replace(np.nan,0)

#cumsum to easily find those channels in the last column
df_badchan = df_badchan.cumsum(axis=1)

#get the last column
df_mybadchan = df_badchan[999]

#find index of non zero (bad channel) rows
df_mybadchan = df_mybadchan.index[df_mybadchan>0]

#delete bad channels
df_nobadchannel = df_data.drop(index=df_mybadchan)

#delete eye channels
df_clean = df_nobadchannel.drop(df_nobadchannel[df_nobadchannel['electrode'] ==125].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==126].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==127].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==128].index)

#save
#df_clean.to_csv("data/df_typ_clean.csv", index=True, header=True)

#################
## Seperate omission from other condition as it is analyzed seperatly
#################

#discard omission data
df_ERPdata = df_clean.drop(df_clean[df_clean["condition"]==4].index)
df_ERPdata = df_ERPdata.reset_index(drop=True)

df_omi = df_clean[df_clean['condition']==4]
df_omi = df_omi.reset_index(drop=True)

#save new dfs
df_ERPdata.to_csv("data/df_ERP_typ.csv", index=True, header=True)
df_omi.to_csv("data/df_omi_typ.csv", index=True, header=True)