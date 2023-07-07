"""
author = 'Anne-Lise Marais, see annelisemarais.github.io'
publication = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old. eLife'
corresponding author = 'nadege.roche@unicaen.fr'

This code extract amplitudes at maximal mismatch amplitude's latency. 
"""

print(__doc__)


import pandas as pd

#################
##PEAK EXTRACT
#################

#load data
typical_TS = pd.read_csv("data/typical_TS.csv", index_col=0)
string_cols = typical_TS.columns[-4:].astype(str).tolist()
int_cols = typical_TS.columns[:1000].astype(int).tolist()
typical_TS.columns = int_cols + string_cols

typical_omi = pd.read_csv("data/typical_omi.csv", index_col=0)
string_cols = typical_omi.columns[-4:].astype(str).tolist()
int_cols = typical_omi.columns[:1000].astype(int).tolist()
typical_omi.columns = int_cols + string_cols

atypical_TS = pd.read_csv("data/atypical_TS.csv", index_col=0)
string_cols = atypical_TS.columns[-4:].astype(str).tolist()
int_cols = atypical_TS.columns[:1000].astype(int).tolist()
atypical_TS.columns = int_cols + string_cols

atypical_omi = pd.read_csv("data/atypical_omi.csv", index_col=0)
string_cols = atypical_omi.columns[-4:].astype(str).tolist()
int_cols = atypical_omi.columns[:1000].astype(int).tolist()
atypical_omi.columns = int_cols + string_cols

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

typical_N140 = timewindow_ROI_extract(typical_TS,somato,(199,399))
typical_MMN = timewindow_ROI_extract(typical_TS,somato,(349,449))
typical_P300 = timewindow_ROI_extract(typical_TS,frontal,(449,549))

atypical_N140 = timewindow_ROI_extract(atypical_TS,somato,(249,349))
atypical_MMN = timewindow_ROI_extract(atypical_TS,somato,(349,449))
atypical_P300 = timewindow_ROI_extract(atypical_TS,frontal,(449,549))


#divide time series by age
two_N140 = typical_N140[typical_N140['age']==2]
four_N140 = typical_N140[typical_N140['age']==4]

two_MMN = typical_MMN[typical_MMN['age']==2]
four_MMN = typical_MMN[typical_MMN['age']==4]

two_P300 = typical_P300[typical_P300['age']==2]
four_P300 = typical_P300[typical_P300['age']==4]


def condition_extract(df):
  fam = df[df['condition']==3].drop(['condition','sub','age'], axis=1).reset_index(drop=True)
  con = df[df['condition']==1].drop(['condition','sub','age'], axis=1).reset_index(drop=True)
  std = df[df['condition']==7].drop(['condition','sub','age'], axis=1).reset_index(drop=True)
  dev = df[df['condition']==2].drop(['condition','sub','age'], axis=1).reset_index(drop=True)
  pom = df[df['condition']==5].drop(['condition','sub','age'], axis=1).reset_index(drop=True)
  return fam, con, std, dev, pom

#Extract condition TS by age and ROI

two_N140_fam, two_N140_con, two_N140_std, two_N140_dev, two_N140_pom = condition_extract(two_N140)
two_MMN_fam, two_MMN_con, two_MMN_std, two_MMN_dev, two_MMN_pom = condition_extract(two_MMN)
two_P300_fam, two_P300_con, two_P300_std, two_P300_dev, two_P300_pom = condition_extract(two_P300)

four_N140_fam, four_N140_con, four_N140_std, four_N140_dev, four_N140_pom = condition_extract(four_N140)
four_MMN_fam, four_MMN_con, four_MMN_std, four_MMN_dev, four_MMN_pom = condition_extract(four_MMN)
four_P300_fam, four_P300_con, four_P300_std, four_P300_dev, four_P300_pom = condition_extract(four_P300)

atypical_N140_fam, atypical_N140_con, atypical_N140_std, atypical_N140_dev, atypical_N140_pom = condition_extract(atypical_N140)
atypical_MMN_fam, atypical_MMN_con, atypical_MMN_std, atypical_MMN_dev, atypical_MMN_pom = condition_extract(atypical_MMN)
atypical_P300_fam, atypical_P300_con, atypical_P300_std, atypical_P300_dev, atypical_P300_pom = condition_extract(atypical_P300)


#Create dfs in which data will be saved
two_stats = pd.DataFrame(columns=['age','fam_somato', 'con_somato', 'rs_latency_somato', 'fam_frontal', 'con_frontal', 'rs_latency_frontal', 'stddev_somato', 'dev_somato', 'dev_latency_somato', 'stddev_frontal', 'dev_frontal', 'dev_latency_frontal','stdpom_somato', 'pom_somato', 'pom_latency_somato', 'stdpom_frontal', 'pom_frontal', 'pom_latency_frontal', 'omi_amp', 'omi_lat', 'omi_base'], index=range(0,12))
two_stats.age = [2]*12

four_stats = pd.DataFrame(columns=['age', 'fam_somato', 'con_somato', 'rs_latency_somato', 'fam_frontal', 'con_frontal', 'rs_latency_frontal', 'stddev_somato', 'dev_somato', 'dev_latency_somato', 'stddev_frontal', 'dev_frontal', 'dev_latency_frontal','stdpom_somato', 'pom_somato', 'pom_latency_somato', 'stdpom_frontal', 'pom_frontal', 'pom_latency_frontal', 'omi_amp', 'omi_lat', 'omi_base'], index=range(0,12))
four_stats.age = [4]*12


atypical_stats = pd.DataFrame(columns=['fam_somato', 'con_somato', 'rs_latency_somato', 'fam_frontal', 'con_frontal', 'rs_latency_frontal', 'stddev_somato', 'dev_somato', 'dev_latency_somato', 'stddev_frontal', 'dev_frontal', 'dev_latency_frontal','stdpom_somato', 'pom_somato', 'pom_latency_somato', 'stdpom_frontal', 'pom_frontal', 'pom_latency_frontal', 'omi_amp', 'omi_lat', 'omi_base'], index=range(0,9))

# Extract mismatch latencies


def mismatch_latency_rs(fam_somato, con_somato,fam_frontal,con_frontal):
  # Find latency of maximal mismatch
  rs_latency_somato = (fam_somato-con_somato).idxmin(axis=1).astype(int)
  rs_latency_frontal = fam_frontal.idxmax(axis=1).astype(int)
  return rs_latency_somato, rs_latency_frontal

two_stats.rs_latency_somato, two_stats.rs_latency_frontal = mismatch_latency_rs(two_N140_fam, two_N140_con, two_P300_fam, two_P300_con)
four_stats.rs_latency_somato, four_stats.rs_latency_frontal = mismatch_latency_rs(four_N140_fam, four_N140_con, four_P300_fam, four_P300_con)
atypical_stats.rs_latency_somato, atypical_stats.rs_latency_frontal = mismatch_latency_rs(atypical_N140_fam, atypical_N140_con, atypical_P300_fam, atypical_P300_con)


def mismatch_latency_somato(dev, std, pom):

  dev_mismatch = (dev-std)
  pom_mismatch = (pom-std)

  if dev.mean().mean() < std.mean().mean():
    dev_latency = dev_mismatch.idxmin(axis=1).astype(int)
  else:
    dev_latency = dev_mismatch.idxmax(axis=1).astype(int)

  if pom.mean().mean() < std.mean().mean():
    pom_latency = pom_mismatch.idxmin(axis=1).astype(int)
  else:
    pom_latency = pom_mismatch.idxmax(axis=1).astype(int)

  return dev_latency, pom_latency

two_stats.dev_latency_somato, two_stats.pom_latency_somato = mismatch_latency_somato(two_MMN_dev, two_MMN_std, two_MMN_pom)
four_stats.dev_latency_somato, four_stats.pom_latency_somato = mismatch_latency_somato(four_MMN_dev, four_MMN_std, four_MMN_pom)
atypical_stats.dev_latency_somato, atypical_stats.pom_latency_somato = mismatch_latency_somato(atypical_MMN_dev, atypical_MMN_std, atypical_MMN_pom)

def P300_latency(dev, std, pom):

  if dev.mean().mean() < std.mean().mean():
    dev_latency = dev.idxmin(axis=1).astype(int)
  else:
    dev_latency = dev.idxmax(axis=1).astype(int)

  if pom.mean().mean() < std.mean().mean():
    pom_latency = pom.idxmin(axis=1).astype(int)
  else:
    pom_latency = pom.idxmax(axis=1).astype(int)

  return dev_latency, pom_latency

two_stats.dev_latency_frontal, two_stats.pom_latency_frontal = P300_latency(two_P300_dev, two_P300_std, two_P300_pom)
four_stats.dev_latency_frontal, four_stats.pom_latency_frontal = P300_latency(four_P300_dev, four_P300_std, four_P300_pom)
atypical_stats.dev_latency_frontal, atypical_stats.pom_latency_frontal = P300_latency(atypical_P300_dev, atypical_P300_std, atypical_P300_pom)


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
two_stats.dev_somato = peak_extract(two_MMN_dev,two_stats.dev_latency_somato)
two_stats.stddev_somato = peak_extract(two_MMN_std,two_stats.dev_latency_somato)
two_stats.pom_somato = peak_extract(two_MMN_pom,two_stats.pom_latency_somato)
two_stats.stdpom_somato = peak_extract(two_MMN_std,two_stats.pom_latency_somato)

two_stats.fam_frontal = peak_extract(two_P300_fam,two_stats.rs_latency_frontal)
two_stats.con_frontal = peak_extract(two_P300_con,two_stats.rs_latency_frontal)
two_stats.dev_frontal = peak_extract(two_P300_dev,two_stats.dev_latency_frontal)
two_stats.stddev_frontal = peak_extract(two_P300_std,two_stats.dev_latency_frontal)
two_stats.pom_frontal = peak_extract(two_P300_pom,two_stats.pom_latency_frontal)
two_stats.stdpom_frontal = peak_extract(two_P300_std,two_stats.pom_latency_frontal)

four_stats.fam_somato = peak_extract(four_N140_fam,four_stats.rs_latency_somato)
four_stats.con_somato = peak_extract(four_N140_con,four_stats.rs_latency_somato)
four_stats.dev_somato = peak_extract(four_MMN_dev,four_stats.dev_latency_somato)
four_stats.stddev_somato = peak_extract(four_MMN_std,four_stats.dev_latency_somato)
four_stats.pom_somato = peak_extract(four_MMN_pom,four_stats.pom_latency_somato)
four_stats.stdpom_somato = peak_extract(four_MMN_std,four_stats.pom_latency_somato)

four_stats.fam_frontal = peak_extract(four_P300_fam,four_stats.rs_latency_frontal)
four_stats.con_frontal = peak_extract(four_P300_con,four_stats.rs_latency_frontal)
four_stats.dev_frontal = peak_extract(four_P300_dev,four_stats.dev_latency_frontal)
four_stats.stddev_frontal = peak_extract(four_P300_std,four_stats.dev_latency_frontal)
four_stats.pom_frontal = peak_extract(four_P300_pom,four_stats.pom_latency_frontal)
four_stats.stdpom_frontal = peak_extract(four_P300_std,four_stats.pom_latency_frontal)

atypical_stats.fam_somato = peak_extract(atypical_N140_fam,atypical_stats.rs_latency_somato)
atypical_stats.con_somato = peak_extract(atypical_N140_con,atypical_stats.rs_latency_somato)
atypical_stats.dev_somato = peak_extract(atypical_MMN_dev,atypical_stats.dev_latency_somato)
atypical_stats.stddev_somato = peak_extract(atypical_MMN_std,atypical_stats.dev_latency_somato)
atypical_stats.pom_somato = peak_extract(atypical_MMN_pom,atypical_stats.pom_latency_somato)
atypical_stats.stdpom_somato = peak_extract(atypical_MMN_std,atypical_stats.pom_latency_somato)

atypical_stats.fam_frontal = peak_extract(atypical_P300_fam,atypical_stats.rs_latency_frontal)
atypical_stats.con_frontal = peak_extract(atypical_P300_con,atypical_stats.rs_latency_frontal)
atypical_stats.dev_frontal = peak_extract(atypical_P300_dev,atypical_stats.dev_latency_frontal)
atypical_stats.stddev_frontal = peak_extract(atypical_P300_std,atypical_stats.dev_latency_frontal)
atypical_stats.pom_frontal = peak_extract(atypical_P300_pom,atypical_stats.pom_latency_frontal)
atypical_stats.stdpom_frontal = peak_extract(atypical_P300_std,atypical_stats.pom_latency_frontal)

######Omission

def omi_baseline(data,ROI):
  TS = data.iloc[:,0:99]
  TS[['electrode', 'sub', 'age']] = data[['electrode', 'sub', 'age']]
  TS_ROI = TS[TS['electrode'].isin(ROI)]
  TS_ROI = TS_ROI.drop('electrode', axis=1)
  baseline = TS_ROI.groupby(['sub','age']).mean().reset_index()
  baseline = baseline.drop(['age', 'sub'], axis=1).min(axis=1)
  return baseline

typical_omi_baseline = omi_baseline(typical_omi,somato)
atypical_omi_baseline = omi_baseline(atypical_omi,somato)

def omi_N1(data,ROI):
  TS = data.iloc[:, 579:799]
  TS[['electrode', 'sub', 'age']] = data[['electrode', 'sub', 'age']]
  TS_ROI = TS[TS['electrode'].isin(ROI)]
  TS_ROI = TS_ROI.drop('electrode', axis=1)
  TS_ROI = TS_ROI.groupby(['sub','age']).mean().reset_index()
  TS_ROI = TS_ROI.drop(['age', 'sub'], axis=1)
  N1_lat = TS_ROI.idxmin(axis=1).astype(int)
  N1_amp = TS_ROI.min(axis=1)
  return N1_lat, N1_amp

typical_omi_N1_lat, typical_omi_N1 = omi_N1(typical_omi,somato)
atypical_omi_N1_lat, atypical_omi_N1 = omi_N1(atypical_omi,somato)


two_stats.omi_base = typical_omi_baseline[:12].reset_index(drop=True)
four_stats.omi_base = typical_omi_baseline[12:].reset_index(drop=True)
atypical_stats.omi_base = atypical_omi_baseline

two_stats.omi_lat = typical_omi_N1_lat[:12].reset_index(drop=True)
four_stats.omi_lat = typical_omi_N1_lat[12:].reset_index(drop=True)
atypical_stats.omi_lat = atypical_omi_N1_lat

two_stats.omi_amp = typical_omi_N1[:12].reset_index(drop=True)
four_stats.omi_amp = typical_omi_N1[12:].reset_index(drop=True)
atypical_stats.omi_amp = atypical_omi_N1

#################
##Export df
#################

two_stats.to_csv("data/two_stats.csv")
four_stats.to_csv("data/four_stats.csv")

all_age_stats = pd.concat([two_stats, four_stats], axis=0).reset_index(drop=True)
all_age_stats.to_excel("data/all_age_stats.xlsx")

atypical_stats.to_excel("data/atypical_stats.xlsx")