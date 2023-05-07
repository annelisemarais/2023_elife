"""
author = 'Anne-Lise Marais, see annelisemarais.github.io'
publication = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old. eLife'
corresponding author = 'nadege.roche@unicaen.fr'

This code extract amplitudes at maximal mismatch amplitude's latency. 
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
##PEAK EXTRACT
#################

#load data
df_ERPdata = pd.read_csv("data/df_ERP_typ.csv", index_col=0)

string_cols = df_ERPdata.columns[-4:].astype(str).tolist()
int_cols = df_ERPdata.columns[:1000].astype(int).tolist()
df_ERPdata.columns = int_cols + string_cols

df_omi = pd.read_csv("data/df_omi_typ.csv", index_col=0)

string_cols = df_omi.columns[-4:].astype(str).tolist()
int_cols = df_omi.columns[:1000].astype(int).tolist()
df_omi.columns = int_cols + string_cols

#Choose electrodes for Region Of Interest (ROI)
somato = [28,29,35,36,41,42,47,52]
frontal = [5,6,7,12,13,106,112,129] 


def timewindow_ROI_extract(df,ROI,timewindow):
  time_series = df.iloc[:, timewindow[0]:timewindow[1]]
  time_series[['condition', 'electrode', 'sub', 'age']] = df[['condition', 'electrode', 'sub', 'age']]
  TS_ROI = time_series[time_series['electrode'].isin(ROI)]
  TS_ROI = TS_ROI.drop('electrode', axis=1)
  TS_ROI = TS_ROI.groupby(['condition', 'sub', 'age'], as_index=False).mean()
  return TS_ROI

df_N140 = timewindow_ROI_extract(df_ERPdata,somato,(199,449))
df_P300 = timewindow_ROI_extract(df_ERPdata,frontal,(399,899))

#divide time series by age
two_N140 = df_N140[df_N140['age']==2]
four_N140 = df_N140[df_N140['age']==4]
six_N140 = df_N140[df_N140['age']==6]

two_P300 = df_P300[df_P300['age']==2]
four_P300 = df_P300[df_P300['age']==4]
six_P300 = df_P300[df_P300['age']==6]

def condition_extract(df):
  fam = df[df['condition']==3].drop(['condition','sub','age'], axis=1).reset_index(drop=True)
  con = df[df['condition']==1].drop(['condition','sub','age'], axis=1).reset_index(drop=True)
  std = df[df['condition']==7].drop(['condition','sub','age'], axis=1).reset_index(drop=True)
  dev = df[df['condition']==2].drop(['condition','sub','age'], axis=1).reset_index(drop=True)
  pom = df[df['condition']==5].drop(['condition','sub','age'], axis=1).reset_index(drop=True)
  return fam, con, std, dev, pom

#Extract condition TS by age and ROI

two_N140_fam, two_N140_con, two_N140_std, two_N140_dev, two_N140_pom = condition_extract(two_N140)
two_P300_fam, two_P300_con, two_P300_std, two_P300_dev, two_P300_pom = condition_extract(two_P300)

four_N140_fam, four_N140_con, four_N140_std, four_N140_dev, four_N140_pom = condition_extract(four_N140)
four_P300_fam, four_P300_con, four_P300_std, four_P300_dev, four_P300_pom = condition_extract(four_P300)

six_N140_fam, six_N140_con, six_N140_std, six_N140_dev, six_N140_pom = condition_extract(six_N140)
six_P300_fam, six_P300_con, six_P300_std, six_P300_dev, six_P300_pom = condition_extract(six_P300)


#Create dfs in which data will be saved
two_stats = pd.DataFrame(columns=['age','fam_somato', 'con_somato', 'rs_latency_somato', 'fam_frontal', 'con_frontal', 'rs_latency_frontal', 'stddev_somato', 'dev_somato', 'dev_latency_somato', 'stddev_frontal', 'dev_frontal', 'dev_latency_frontal','stdpom_somato', 'pom_somato', 'pom_latency_somato', 'stdpom_frontal', 'pom_frontal', 'pom_latency_frontal', 'omi_amp', 'omi_lat', 'omi_base'], index=range(0,12))
two_stats.age = [2]*12

four_stats = pd.DataFrame(columns=['age', 'fam_somato', 'con_somato', 'rs_latency_somato', 'fam_frontal', 'con_frontal', 'rs_latency_frontal', 'stddev_somato', 'dev_somato', 'dev_latency_somato', 'stddev_frontal', 'dev_frontal', 'dev_latency_frontal','stdpom_somato', 'pom_somato', 'pom_latency_somato', 'stdpom_frontal', 'pom_frontal', 'pom_latency_frontal', 'omi_amp', 'omi_lat', 'omi_base'], index=range(0,12))
four_stats.age = [4]*12

six_stats = pd.DataFrame(columns=['age','fam_somato', 'con_somato', 'rs_latency_somato', 'fam_frontal', 'con_frontal', 'rs_latency_frontal', 'stddev_somato', 'dev_somato', 'dev_latency_somato', 'stddev_frontal', 'dev_frontal', 'dev_latency_frontal','stdpom_somato', 'pom_somato', 'pom_latency_somato', 'stdpom_frontal', 'pom_frontal', 'pom_latency_frontal', 'omi_amp', 'omi_lat', 'omi_base'], index=range(0,12))
six_stats.age = [6]*12


# Extract mismatch latencies

def mismatch_latency(fam, con, dev, std, pom, polarity):
  # Find latency of maximal mismatch
  if polarity == 'Neg':
      rs_latency = (fam-con).idxmin(axis=1).astype(int)
      dev_latency = (dev-std).idxmin(axis=1).astype(int)
      pom_latency = (pom-std).idxmin(axis=1).astype(int)
  else:
      rs_latency = (fam-con).idxmax(axis=1).astype(int)
      dev_latency = (dev-std).idxmax(axis=1).astype(int)
      pom_latency = (pom-std).idxmax(axis=1).astype(int)
  return rs_latency, dev_latency, pom_latency


two_stats.rs_latency_somato, two_stats.dev_latency_somato, two_stats.pom_latency_somato = mismatch_latency(two_N140_fam, two_N140_con, two_N140_dev, two_N140_std, two_N140_pom, polarity="Neg")
two_stats.rs_latency_frontal, two_stats.dev_latency_frontal, two_stats.pom_latency_frontal = mismatch_latency(two_P300_fam, two_P300_con, two_P300_dev, two_P300_std, two_P300_pom, polarity="Pos")

four_stats.rs_latency_somato, four_stats.dev_latency_somato, four_stats.pom_latency_somato = mismatch_latency(four_N140_fam, four_N140_con, four_N140_dev, four_N140_std, four_N140_pom, polarity="Neg")
four_stats.rs_latency_frontal, four_stats.dev_latency_frontal, four_stats.pom_latency_frontal = mismatch_latency(four_P300_fam, four_P300_con, four_P300_dev, four_P300_std, four_P300_pom, polarity="Pos")

six_stats.rs_latency_somato, six_stats.dev_latency_somato, six_stats.pom_latency_somato = mismatch_latency(six_N140_fam, six_N140_con, six_N140_dev, six_N140_std, six_N140_pom, polarity="Neg")
six_stats.rs_latency_frontal, six_stats.dev_latency_frontal, six_stats.pom_latency_frontal = mismatch_latency(six_P300_fam, six_P300_con, six_P300_dev, six_P300_std, six_P300_pom, polarity="Pos")

# Extract peaks based on mismatch latencies

def peak_extract(condition, latency):
  peak = []
  for sub, latency in enumerate(latency):
      amplitude = condition.loc[:, latency]
      peak.append(amplitude[sub])
  
  df = pd.DataFrame(peak)
  return df

two_stats.fam_somato = peak_extract(two_N140_fam,two_stats.rs_latency_somato)
two_stats.con_somato = peak_extract(two_N140_con,two_stats.rs_latency_somato)
two_stats.dev_somato = peak_extract(two_N140_dev,two_stats.dev_latency_somato)
two_stats.stddev_somato = peak_extract(two_N140_std,two_stats.dev_latency_somato)
two_stats.pom_somato = peak_extract(two_N140_pom,two_stats.pom_latency_somato)
two_stats.stdpom_somato = peak_extract(two_N140_std,two_stats.pom_latency_somato)

two_stats.fam_frontal = peak_extract(two_P300_fam,two_stats.rs_latency_frontal)
two_stats.con_frontal = peak_extract(two_P300_con,two_stats.rs_latency_frontal)
two_stats.dev_frontal = peak_extract(two_P300_dev,two_stats.dev_latency_frontal)
two_stats.stddev_frontal = peak_extract(two_P300_std,two_stats.dev_latency_frontal)
two_stats.pom_frontal = peak_extract(two_P300_pom,two_stats.pom_latency_frontal)
two_stats.stdpom_frontal = peak_extract(two_P300_std,two_stats.pom_latency_frontal)

four_stats.fam_somato = peak_extract(four_N140_fam,four_stats.rs_latency_somato)
four_stats.con_somato = peak_extract(four_N140_con,four_stats.rs_latency_somato)
four_stats.dev_somato = peak_extract(four_N140_dev,four_stats.dev_latency_somato)
four_stats.stddev_somato = peak_extract(four_N140_std,four_stats.dev_latency_somato)
four_stats.pom_somato = peak_extract(four_N140_pom,four_stats.pom_latency_somato)
four_stats.stdpom_somato = peak_extract(four_N140_std,four_stats.pom_latency_somato)

four_stats.fam_frontal = peak_extract(four_P300_fam,four_stats.rs_latency_frontal)
four_stats.con_frontal = peak_extract(four_P300_con,four_stats.rs_latency_frontal)
four_stats.dev_frontal = peak_extract(four_P300_dev,four_stats.dev_latency_frontal)
four_stats.stddev_frontal = peak_extract(four_P300_std,four_stats.dev_latency_frontal)
four_stats.pom_frontal = peak_extract(four_P300_pom,four_stats.pom_latency_frontal)
four_stats.stdpom_frontal = peak_extract(four_P300_std,four_stats.pom_latency_frontal)

six_stats.fam_somato = peak_extract(six_N140_fam,six_stats.rs_latency_somato)
six_stats.con_somato = peak_extract(six_N140_con,six_stats.rs_latency_somato)
six_stats.dev_somato = peak_extract(six_N140_dev,six_stats.dev_latency_somato)
six_stats.stddev_somato = peak_extract(six_N140_std,six_stats.dev_latency_somato)
six_stats.pom_somato = peak_extract(six_N140_pom,six_stats.pom_latency_somato)
six_stats.stdpom_somato = peak_extract(six_N140_std,six_stats.pom_latency_somato)

six_stats.fam_frontal = peak_extract(six_P300_fam,six_stats.rs_latency_frontal)
six_stats.con_frontal = peak_extract(six_P300_con,six_stats.rs_latency_frontal)
six_stats.dev_frontal = peak_extract(six_P300_dev,six_stats.dev_latency_frontal)
six_stats.stddev_frontal = peak_extract(six_P300_std,six_stats.dev_latency_frontal)
six_stats.pom_frontal = peak_extract(six_P300_pom,six_stats.pom_latency_frontal)
six_stats.stdpom_frontal = peak_extract(six_P300_std,six_stats.pom_latency_frontal)


######Omission

#Same thing for omission and omission baseline
N100_baseline = df_omi.iloc[:,0:99]
N100_baseline[['electrode', 'sub', 'age']] = df_omi[['electrode', 'sub', 'age']]
N100_baseline = N100_baseline[N100_baseline['electrode'].isin(somato)]
N100_baseline = N100_baseline.drop('electrode', axis=1)
N100_baseline = N100_baseline.groupby(['sub','age']).mean().reset_index()
N100_base = N100_baseline.drop(['age', 'sub'], axis=1).min(axis=1)


N100_TS = df_omi.iloc[:, 579:799]
N100_TS[['electrode', 'sub', 'age']] = df_omi[['electrode', 'sub', 'age']]
N100_TS = N100_TS[N100_TS['electrode'].isin(somato)]
N100_TS = N100_TS.drop('electrode', axis=1)
N100_TS = N100_TS.groupby(['sub','age']).mean().reset_index()
N100_lat = N100_TS.drop(['age', 'sub'], axis=1).idxmin(axis=1).astype(int)
N100_amp = N100_TS.drop(['age', 'sub'], axis=1).min(axis=1)
omi_stats = pd.concat([N100_lat, N100_amp,N100_base], axis=1)
omi_stats.columns = ['N100_latency', 'N100_amplitude', 'N100_base']
omi_stats['age']=N100_TS['age']


two_stats.omi_amp = omi_stats.loc[omi_stats['age']==2, 'N100_amplitude'].reset_index(drop=True)
four_stats.omi_amp = omi_stats.loc[omi_stats['age']==4, 'N100_amplitude'].reset_index(drop=True)
six_stats.omi_amp = omi_stats.loc[omi_stats['age']==6, 'N100_amplitude'].reset_index(drop=True)

two_stats.omi_lat = omi_stats.loc[omi_stats['age']==2, 'N100_latency'].reset_index(drop=True)
four_stats.omi_lat = omi_stats.loc[omi_stats['age']==4, 'N100_latency'].reset_index(drop=True)
six_stats.omi_lat = omi_stats.loc[omi_stats['age']==6, 'N100_latency'].reset_index(drop=True)

two_stats.omi_base = omi_stats.loc[omi_stats['age']==2, 'N100_base'].reset_index(drop=True)
four_stats.omi_base = omi_stats.loc[omi_stats['age']==4, 'N100_base'].reset_index(drop=True)
six_stats.omi_base = omi_stats.loc[omi_stats['age']==6, 'N100_base'].reset_index(drop=True)

#################
##CONCATE to export df
#################

all_age_stats = pd.concat([two_stats, four_stats, six_stats], axis=0).reset_index(drop=True)

all_age_stats.to_excel("data/all_age_stats.xlsx")



#################
##Statistics
#################


tests = [(two_stats.con_somato,two_stats.fam_somato),(two_stats.fam_frontal, two_stats.con_frontal),(two_stats.stddev_somato, two_stats.dev_somato),(two_stats.dev_frontal,two_stats.stddev_frontal),(two_stats.stdpom_somato, two_stats.pom_somato),(two_stats.pom_frontal, two_stats.stdpom_frontal), ( four_stats.con_somato, four_stats.fam_somato),(four_stats.fam_frontal, four_stats.con_frontal),(four_stats.stddev_somato, four_stats.dev_somato),(four_stats.dev_frontal, four_stats.stddev_frontal),(four_stats.stdpom_somato, four_stats.pom_somato),(four_stats.pom_frontal, four_stats.stdpom_frontal),( six_stats.con_somato, six_stats.fam_somato),(six_stats.fam_frontal, six_stats.con_frontal),(six_stats.stddev_somato, six_stats.dev_somato),(six_stats.dev_frontal, six_stats.stddev_frontal),(six_stats.stdpom_somato, six_stats.pom_somato),(six_stats.pom_frontal, six_stats.stdpom_frontal), (two_stats.omi_lat, four_stats.omi_lat),(two_stats.omi_lat,six_stats.omi_lat),(four_stats.omi_lat, six_stats.omi_lat),(two_stats.rs_latency_somato, four_stats.rs_latency_somato),(two_stats.rs_latency_somato, six_stats.rs_latency_somato),(four_stats.rs_latency_somato, six_stats.rs_latency_somato),(two_stats.rs_latency_frontal, four_stats.rs_latency_frontal),(two_stats.rs_latency_frontal, six_stats.rs_latency_frontal),(four_stats.rs_latency_frontal, six_stats.rs_latency_frontal),(two_stats.dev_latency_somato, four_stats.dev_latency_somato),(two_stats.dev_latency_somato, six_stats.dev_latency_somato),(four_stats.dev_latency_somato, six_stats.dev_latency_somato),(two_stats.dev_latency_frontal, four_stats.dev_latency_frontal),(two_stats.dev_latency_frontal, six_stats.dev_latency_frontal),(four_stats.dev_latency_frontal, six_stats.dev_latency_frontal),(two_stats.pom_latency_somato, four_stats.pom_latency_somato),(two_stats.pom_latency_somato, six_stats.pom_latency_somato),(four_stats.pom_latency_somato, six_stats.pom_latency_somato),(two_stats.pom_latency_frontal, four_stats.pom_latency_frontal),(two_stats.pom_latency_frontal, six_stats.pom_latency_frontal),(four_stats.pom_latency_frontal, six_stats.pom_latency_frontal),(two_stats.omi_base, two_stats.omi_amp),(four_stats.omi_base, four_stats.omi_amp),(six_stats.omi_base, six_stats.omi_amp)]


n_tests = len(tests)

df_p_value = pd.DataFrame(columns=['p_values', 'statistics'], index=['N140_RS_two','P300_RS_two','N140_MMN_two','P300_MMN_two','N140_MMNpom_two','P300_MMNpom_two','N140_RS_four','P300_RS_four','N140_MMN_four','P300_MMN_four','N140_MMNpom_four','P300_MMNpom_four', 'N140_RS_six','P300_RS_six','N140_MMN_six','P300_MMN_six','N140_MMNpom_six','P300_MMNpom_six','twofour_omi_lat','twosix_omi_lat','foursix_omi_lat', 'twofour_rs_somato_lat', 'twosix_rs_somato_lat','foursix_rs_somato_lat', 'twofour_rs_frontal_lat', 'twosix_rs_frontal_lat','foursix_rs_frontal_lat', 'twofour_dev_somato_lat', 'twosix_dev_somato_lat','foursix_dev_somato_lat', 'twofour_dev_frontal_lat', 'twosix_dev_frontal_lat','foursix_dev_frontal_lat', 'twofour_pom_somato_lat', 'twosix_pom_somato_lat','foursix_pom_somato_lat', 'twofour_pom_frontal_lat', 'twosix_pom_frontal_lat','foursix_pom_frontal_lat', 'omi_two', 'omi_four', 'omi_six'])
p_values = []
for ind, samples in enumerate(tests):
  sample1 = samples[0]
  sample2 = samples[1]
  stat, p = stats.wilcoxon(sample1, sample2, alternative="greater")
  df_p_value.iloc[ind] = p, stat
  p_values.append(p)


p_values = np.array(p_values)

significant, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

df_p_value['significant'] = significant
df_p_value['corrected'] = corrected_p_values
df_p_value

df_p_value.to_excel("data/p_value.xlsx")