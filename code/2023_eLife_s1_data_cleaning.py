"""
author = 'Anne-Lise Marais, see annelisemarais.github.io'
publication = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical and atypical children at two and four. eLife'
corresponding author = 'nadege.roche@unicaen.fr'


This code cleans data: remove bad channels
"""

print(__doc__)

import pandas as pd
import numpy as np

#################
##Typical children
#################

df_data = pd.read_csv("data/data_typical.csv", index_col=0, header=0)

string_cols = df_data.columns[-4:].astype(str).tolist()
int_cols = df_data.columns[:1000].astype(int).tolist()
df_data.columns = int_cols + string_cols

#################
##Remove bad channels and eye channels
#################

###create a standardized df
df_data_std = df_data.drop(['condition', 'electrode', 'sub','age'], axis=1)
df_standardized = (df_data_std - np.mean(df_data_std)) / np.std(df_data_std)

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
typical_TS = df_clean.drop(df_clean[df_clean["condition"]==4].index)
typical_TS = typical_TS.reset_index(drop=True)

typical_omi = df_clean[df_clean['condition']==4]
typical_omi = typical_omi.reset_index(drop=True)

#save new dfs
typical_TS.to_csv("data/typical_TS.csv", index=True, header=True)
typical_omi.to_csv("data/typical_omi.csv", index=True, header=True)


#################
##Atypical children
#################

data_aty = pd.read_csv("data/data_atypical.csv", index_col=0, header=0)

string_cols = data_aty.columns[-4:].astype(str).tolist()
int_cols = data_aty.columns[:1000].astype(int).tolist()
data_aty.columns = int_cols + string_cols

values = data_aty.drop(['condition', 'electrode','sub','age'], axis=1)

###create a standardized df
standardized = (values - np.mean(values)) / np.std(values)

#find channel with data > 3 std
badchan = standardized[standardized>3]
badchan = badchan.replace(np.nan,0)

#cumsum to easily find those channels in the last column
badchan = badchan.cumsum(axis=1)

#get the last column
mybadchan = badchan[999]

#find index of non zero (bad channel) rwos
mybadchan = mybadchan.index[mybadchan>0]

#delete bad channels
nobadchannel = data_aty.drop(index=mybadchan)

#delete eye channels
clean_aty = nobadchannel.drop(nobadchannel[nobadchannel['electrode'] ==125].index)
clean_aty = clean_aty.drop(clean_aty[clean_aty['electrode'] ==126].index)
clean_aty = clean_aty.drop(clean_aty[clean_aty['electrode'] ==127].index)
clean_aty = clean_aty.drop(clean_aty[clean_aty['electrode'] ==128].index)

#save
#clean.to_csv("data/typical/df_clean.csv")

##############
## Third step : separate data for PCA1 (data without omission) and PCA2 (omission only) 
##############

#seperate omission data
atypical_TS = clean_aty.drop(clean_aty[clean_aty["condition"]==4].index)
atypical_TS = atypical_TS.reset_index(drop=True)

atypical_omi = clean_aty[clean_aty['condition']==4]
atypical_omi = atypical_omi.reset_index(drop=True)

#save new dfs
atypical_TS.to_csv("data/atypical_TS.csv", index=True, header=True)
atypical_omi.to_csv("data/atypical_omi.csv", index=True, header=True)
