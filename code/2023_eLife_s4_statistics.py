"""
author = 'Anne-Lise Marais, see annelisemarais.github.io'
publication = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old. eLife'
corresponding author = 'nadege.roche@unicaen.fr'

This code extract amplitudes at maximal mismatch amplitude's latency. 
"""

print(__doc__)


import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import seaborn
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

#################
##PEAK EXTRACT
#################

#load data
two_stats = pd.read_csv("data/two_stats.csv", index_col=0)
four_stats = pd.read_csv("data/four_stats.csv", index_col=0)

#######################
##Typical RS and SP by age
#######################

tests = [(two_stats.con_somato,two_stats.fam_somato),(two_stats.fam_frontal, two_stats.con_frontal),(two_stats.stddev_somato, two_stats.dev_somato),(two_stats.dev_frontal,two_stats.stddev_frontal),(two_stats.stdpom_somato, two_stats.pom_somato),(two_stats.pom_frontal, two_stats.stdpom_frontal), ( four_stats.con_somato, four_stats.fam_somato),(four_stats.fam_frontal, four_stats.con_frontal),(four_stats.stddev_somato, four_stats.dev_somato),(four_stats.dev_frontal, four_stats.stddev_frontal),(four_stats.stdpom_somato, four_stats.pom_somato),(four_stats.pom_frontal, four_stats.stdpom_frontal),(two_stats.omi_base, two_stats.omi_amp),(four_stats.omi_base, four_stats.omi_amp)]

n_tests = len(tests)

p_value_all_age = pd.DataFrame(columns=['p_values', 'statistics'], index=['two_RS_N140','two_RS_P300','two_dev_MMN','two_dev_P300','two_pom_MMN','two_pom_P300','four_RS_N140','four_RS_P300','four_dev_MMN','four_dev_P300','four_pom_MMN','four_pom_P300', 'two_omi', 'four_omi'])
p_values = []
for ind, samples in enumerate(tests):
  sample1 = samples[0]
  sample2 = samples[1]
  stat, p = stats.wilcoxon(sample1, sample2)
  p_value_all_age.iloc[ind] = p, stat
  p_values.append(p)


p_values = np.array(p_values)

significant, corrected_p_values, _, _ = multipletests(p_values, method='fdr_by')

p_value_all_age['significant'] = significant
p_value_all_age['corrected'] = corrected_p_values
p_value_all_age

p_value_all_age.to_excel("data/p_value_all_age.xlsx")

#######################
##Typical 2 yo VS 4 yo
#######################

tests = [(two_stats.rs_latency_somato, four_stats.rs_latency_somato),(two_stats.rs_latency_frontal, four_stats.rs_latency_frontal),(two_stats.dev_latency_somato, four_stats.dev_latency_somato),(two_stats.dev_latency_frontal, four_stats.dev_latency_frontal),(two_stats.pom_latency_somato, four_stats.pom_latency_somato),(two_stats.pom_latency_frontal, four_stats.pom_latency_frontal),(two_stats.omi_lat, four_stats.omi_lat)]

n_tests = len(tests)

p_value_all_age_vs = pd.DataFrame(columns=['p_values', 'statistics'], index=['vs_RS_N140_lat', 'vs_RS_P300_lat', 'vs_dev_MMN_lat', 'vs_dev_P300_lat', 'vs_pom_MMN_lat', 'vs_pom_P300_lat', 'vs_omi_lat'])
p_values = []
for ind, samples in enumerate(tests):
  sample1 = samples[0]
  sample2 = samples[1]
  stat, p = stats.mannwhitneyu(sample1, sample2)
  p_value_all_age_vs.iloc[ind] = p, stat
  p_values.append(p)


p_values = np.array(p_values)

significant, corrected_p_values, _, _ = multipletests(p_values, method='fdr_by')

p_value_all_age_vs['significant'] = significant
p_value_all_age_vs['corrected'] = corrected_p_values
p_value_all_age_vs

p_value_all_age_vs.to_excel("data/p_value_all_age_vs.xlsx")

#######################
##Atypical RS and SP
#######################

tests = [(atypical_stats.con_somato,atypical_stats.fam_somato),(atypical_stats.fam_frontal, atypical_stats.con_frontal),(atypical_stats.stddev_somato, atypical_stats.dev_somato),(atypical_stats.dev_frontal,atypical_stats.stddev_frontal),(atypical_stats.stdpom_somato, atypical_stats.pom_somato),(atypical_stats.pom_frontal, atypical_stats.stdpom_frontal), (atypical_stats.omi_base, atypical_stats.omi_amp)]

n_tests = len(tests)

p_value_atyp = pd.DataFrame(columns=['p_values', 'statistics'], index=['atyp_RS_N140','atyp_RS_P300','atyp_dev_MMN','atyp_dev_P300','atyp_pom_MMN','atyp_pom_P300','atyp_omi'])

p_values = []
for ind, samples in enumerate(tests):
  sample1 = samples[0]
  sample2 = samples[1]
  stat, p = stats.wilcoxon(sample1, sample2)
  p_value_atyp.iloc[ind] = p, stat
  p_values.append(p)


p_values = np.array(p_values)

significant, corrected_p_values, _, _ = multipletests(p_values, method='fdr_by')

p_value_atyp['significant'] = significant
p_value_atyp['corrected'] = corrected_p_values
p_value_atyp

p_value_atyp.to_excel("data/p_value_atyp.xlsx")

#######################
##Typical VS Atypical
#######################

mismatch_ty_stats = pd.DataFrame(columns=['RS_somato', 'RS_somato_lat', 'RS_frontal', 'RS_frontal_lat', 'dev_MMN', 'dev_MMN_lat', 'dev_P300', 'dev_P300_lat','pom_MMN', 'pom_MMN_lat', 'pom_P300', 'pom_P300_lat','omi_amp','omi_lat'], index=range(0,12))
mismatch_aty_stats = pd.DataFrame(columns=['RS_somato', 'RS_somato_lat', 'RS_frontal', 'RS_frontal_lat', 'dev_MMN', 'dev_MMN_lat', 'dev_P300', 'dev_P300_lat','pom_MMN', 'pom_MMN_lat', 'pom_P300', 'pom_P300_lat','omi_amp','omi_lat'], index=range(0,9))


mismatch_ty_stats.RS_somato = four_stats.fam_somato - four_stats.con_somato
mismatch_ty_stats.RS_somato_lat  = four_stats.rs_latency_somato
mismatch_ty_stats.dev_MMN = four_stats.dev_somato - four_stats.stddev_somato
mismatch_ty_stats.dev_MMN_lat  = four_stats.dev_latency_somato
mismatch_ty_stats.pom_MMN = four_stats.pom_somato - four_stats.stdpom_somato
mismatch_ty_stats.pom_MMN_lat  = four_stats.pom_latency_somato
mismatch_ty_stats.RS_frontal = four_stats.fam_frontal - four_stats.con_frontal
mismatch_ty_stats.RS_frontal_lat  = four_stats.rs_latency_frontal
mismatch_ty_stats.dev_P300 = four_stats.dev_frontal
mismatch_ty_stats.dev_P300_lat  = four_stats.dev_latency_frontal
mismatch_ty_stats.pom_P300 = four_stats.pom_frontal
mismatch_ty_stats.pom_P300_lat  = four_stats.pom_latency_frontal
mismatch_ty_stats.omi_amp = four_stats.omi_amp
mismatch_ty_stats.omi_lat = four_stats.omi_lat

mismatch_aty_stats.RS_somato = atypical_stats.fam_somato - atypical_stats.con_somato
mismatch_aty_stats.RS_somato_lat  = atypical_stats.rs_latency_somato
mismatch_aty_stats.dev_MMN = (atypical_stats.dev_somato - atypical_stats.stddev_somato)
mismatch_aty_stats.dev_MMN_lat  = atypical_stats.dev_latency_somato
mismatch_aty_stats.pom_MMN = atypical_stats.pom_somato - atypical_stats.stdpom_somato
mismatch_aty_stats.pom_MMN_lat  = atypical_stats.pom_latency_somato
mismatch_aty_stats.RS_frontal = atypical_stats.fam_frontal - atypical_stats.con_frontal
mismatch_aty_stats.RS_frontal_lat  = atypical_stats.rs_latency_frontal
mismatch_aty_stats.dev_P300 = atypical_stats.dev_frontal
mismatch_aty_stats.dev_P300_lat  = atypical_stats.dev_latency_frontal
mismatch_aty_stats.pom_P300 = atypical_stats.pom_frontal
mismatch_aty_stats.pom_P300_lat  = atypical_stats.pom_latency_frontal
mismatch_aty_stats.omi_amp = atypical_stats.omi_amp
mismatch_aty_stats.omi_lat = atypical_stats.omi_lat


tests = [(mismatch_ty_stats.RS_somato,mismatch_aty_stats.RS_somato),(mismatch_ty_stats.RS_frontal,mismatch_aty_stats.RS_frontal),(mismatch_ty_stats.RS_somato_lat,mismatch_aty_stats.RS_somato_lat),(mismatch_ty_stats.RS_frontal_lat,mismatch_aty_stats.RS_frontal_lat),(mismatch_ty_stats.dev_MMN,mismatch_aty_stats.dev_MMN),(mismatch_ty_stats.dev_P300,mismatch_aty_stats.dev_P300),(mismatch_ty_stats.dev_MMN_lat,mismatch_aty_stats.dev_MMN_lat),(mismatch_ty_stats.dev_P300_lat,mismatch_aty_stats.dev_P300_lat),(mismatch_ty_stats.pom_MMN,mismatch_aty_stats.pom_MMN),(mismatch_ty_stats.pom_P300,mismatch_aty_stats.pom_P300),(mismatch_ty_stats.pom_MMN_lat,mismatch_aty_stats.pom_MMN_lat),(mismatch_ty_stats.pom_P300_lat,mismatch_aty_stats.pom_P300_lat),(mismatch_ty_stats.omi_amp,mismatch_aty_stats.omi_amp),(mismatch_ty_stats.omi_lat,mismatch_aty_stats.omi_lat)]

n_tests = len(tests)

p_value_typvsatyp = pd.DataFrame(columns=['p_values', 'statistics'], index=['vs_RS_somato','vs_RS_frontal','vs_RS_somato_lat','vs_RS_frontal_lat','vs_dev_MMN','vs_dev_P300','vs_dev_MMN_lat','vs_dev_P300_lat','vs_pom_MMN','vs_pom_P300','vs_pom_MMN_lat','vs_pom_P300_lat', 'vs_omi_amp','vs_omi_lat'])

p_values = []
for ind, samples in enumerate(tests):
  sample1 = samples[0]
  sample2 = samples[1]
  stat, p = stats.mannwhitneyu(sample1, sample2)
  p_value_typvsatyp.iloc[ind] = p, stat
  p_values.append(p)


p_values = np.array(p_values)

significant, corrected_p_values, _, _ = multipletests(p_values, method='fdr_by')

p_value_typvsatyp['significant'] = significant
p_value_typvsatyp['corrected'] = corrected_p_values
p_value_typvsatyp

p_value_typvsatyp.to_excel("data/p_value_typvsatyp.xlsx")

#######################
##ERP correlated with psycho
#######################

psycho = pd.read_excel('data/data_psycho.xlsx')

scaled_psycho_both = psycho.reset_index(drop=True)

scaled_psycho_both[scaled_psycho_both.columns[:]] = StandardScaler().fit_transform(scaled_psycho_both.values)


PCA_both = PCA(n_components=2)
PCA_both.fit(scaled_psycho_both)
components_both = PCA_both.components_

scores_both = PCA_both.fit_transform(scaled_psycho_both) #time_series * n_comp

factor_scores_both = pd.DataFrame(scores_both, columns=['comp1', 'comp2'])

factor_scores_typ = factor_scores_both[:12].reset_index(drop=True)
factor_scores_atyp = factor_scores_both[12:].reset_index(drop=True)

df_EEG_typ = mismatch_ty_stats
df_EEG_typ[df_EEG_typ.columns[:]] = StandardScaler().fit_transform(df_EEG_typ.values)
columns = list(factor_scores_typ.columns) + list(df_EEG_typ.columns)
data_global_typ = pd.concat([factor_scores_typ, df_EEG_typ], axis=1)


df_EEG_atyp = mismatch_aty_stats
df_EEG_atyp[df_EEG_atyp.columns[:]] = StandardScaler().fit_transform(df_EEG_atyp.values)
columns = list(factor_scores_atyp.columns) + list(df_EEG_atyp.columns)
data_global_atyp = pd.concat([factor_scores_atyp, df_EEG_atyp], axis=1)


df_EEG_both = pd.concat([mismatch_ty_stats,mismatch_aty_stats], axis=0).reset_index(drop=True)
df_EEG_both[df_EEG_both.columns[:]] = StandardScaler().fit_transform(df_EEG_both.values)
columns = list(factor_scores_both.columns) + list(df_EEG_both.columns)
data_global_both = pd.concat([factor_scores_both, df_EEG_both], axis=1)
data_global_both['status'] = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, p = pearsonr(x, y)
    ax = ax or plt.gca()
    if p <= 0.05:
      ax.annotate(f'r = {r:.2f}, p = {p:.3f}', xy=(.1, .9), xycoords=ax.transAxes)
      ax.annotate('*', xy=(.9, .9), xycoords=ax.transAxes)
    elif p <= 0.08:
      ax.annotate(f'r = {r:.2f}, p = {p:.3f}', xy=(.1, .9), xycoords=ax.transAxes)
    else:
      ax.annotate(f'r = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)



ylim = (round(data_global_both.comp1.min())-1, round(data_global_both.comp1.max())+1)
xlim = (round(data_global_both.drop(['comp1', 'comp2'], axis=1).min().min())), (round(data_global_both.drop(['comp1', 'comp2'], axis=1).max().max()))

#plt.close('all')
#g = seaborn.pairplot(data_global_typ, x_vars=list(df_EEG_typ.columns), y_vars=list(factor_scores_typ.columns))
#g.map(corrfunc)
#for ax in g.axes.flat:
#    ax.set_xlim(xlim)
#plt.show()

#plt.close('all')
#g = seaborn.pairplot(data_global_atyp, x_vars=list(df_EEG_atyp.columns), y_vars=list(factor_scores_atyp.columns))
#g.map(corrfunc)
#plt.tight_layout()
#plt.show()

#plt.close('all')
#g = seaborn.pairplot(data_global_both, x_vars=list(df_EEG_both.columns), y_vars=list(factor_scores_both.columns), hue='status')
##g.map(corrfunc)
#plt.tight_layout()
#plt.show()




def corrfunc_both(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, p = pearsonr(x, y)
    ax = ax or plt.gca()
    if p <= 0.05:
        color = kws.get('color', 'black')
        ax.annotate(f'r = {r:.2f}, p = {p:.3f}', xy=(.1, .9), xycoords=ax.transAxes, color=color)
        ax.annotate('*', xy=(.9, .9), xycoords=ax.transAxes, color=color)
    elif p <= 0.08:
        color = kws.get('color', 'black')
        ax.annotate(f'r = {r:.2f}, p = {p:.3f}', xy=(.1, .9), xycoords=ax.transAxes, color=color)
    else:
        color = kws.get('color', 'black')
        ax.annotate(f'r = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes, color=color)



def add_regression_line(x, y, **kwargs):
    seaborn.regplot(x=x, y=y, scatter=False, ci=None, **kwargs)

plt.close('all')
g = seaborn.pairplot(data_global_both, x_vars=list(df_EEG_both.columns), y_vars=list(factor_scores_both.columns), hue='status', palette=['green', 'red'])
g.map(add_regression_line)
#g.map(corrfunc_both)
for ax in g.axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
plt.tight_layout()
plt.savefig("figures/correlations.png")
plt.show()

tests = [(df_EEG_typ.RS_somato,factor_scores_typ.comp1),(df_EEG_typ.RS_somato_lat,factor_scores_typ.comp1),(df_EEG_typ.dev_MMN,factor_scores_typ.comp1),(df_EEG_typ.dev_MMN_lat,factor_scores_typ.comp1),(df_EEG_typ.pom_MMN,factor_scores_typ.comp1),(df_EEG_typ.pom_MMN_lat,factor_scores_typ.comp1),(df_EEG_typ.RS_frontal,factor_scores_typ.comp1),(df_EEG_typ.RS_frontal_lat,factor_scores_typ.comp1),(df_EEG_typ.dev_P300,factor_scores_typ.comp1),(df_EEG_typ.dev_P300_lat,factor_scores_typ.comp1),(df_EEG_typ.pom_P300,factor_scores_typ.comp1),(df_EEG_typ.pom_P300_lat,factor_scores_typ.comp1),(df_EEG_typ.omi_amp,factor_scores_typ.comp1),(df_EEG_typ.omi_lat,factor_scores_typ.comp1), (df_EEG_typ.RS_somato,factor_scores_typ.comp2),(df_EEG_typ.RS_somato_lat,factor_scores_typ.comp2),(df_EEG_typ.dev_MMN,factor_scores_typ.comp2),(df_EEG_typ.dev_MMN_lat,factor_scores_typ.comp2),(df_EEG_typ.pom_MMN,factor_scores_typ.comp2),(df_EEG_typ.pom_MMN_lat,factor_scores_typ.comp2),(df_EEG_typ.RS_frontal,factor_scores_typ.comp2),(df_EEG_typ.RS_frontal_lat,factor_scores_typ.comp2),(df_EEG_typ.dev_P300,factor_scores_typ.comp2),(df_EEG_typ.dev_P300_lat,factor_scores_typ.comp2),(df_EEG_typ.pom_P300,factor_scores_typ.comp2),(df_EEG_typ.pom_P300_lat,factor_scores_typ.comp2),(df_EEG_typ.omi_amp,factor_scores_typ.comp2),(df_EEG_typ.omi_lat,factor_scores_typ.comp2),(df_EEG_atyp.RS_somato,factor_scores_atyp.comp1),(df_EEG_atyp.RS_somato_lat,factor_scores_atyp.comp1),(df_EEG_atyp.dev_MMN,factor_scores_atyp.comp1),(df_EEG_atyp.dev_MMN_lat,factor_scores_atyp.comp1),(df_EEG_atyp.pom_MMN,factor_scores_atyp.comp1),(df_EEG_atyp.pom_MMN_lat,factor_scores_atyp.comp1),(df_EEG_atyp.RS_frontal,factor_scores_atyp.comp1),(df_EEG_atyp.RS_frontal_lat,factor_scores_atyp.comp1),(df_EEG_atyp.dev_P300,factor_scores_atyp.comp1),(df_EEG_atyp.dev_P300_lat,factor_scores_atyp.comp1),(df_EEG_atyp.pom_P300,factor_scores_atyp.comp1),(df_EEG_atyp.pom_P300_lat,factor_scores_atyp.comp1),(df_EEG_atyp.omi_amp,factor_scores_atyp.comp1),(df_EEG_atyp.omi_lat,factor_scores_atyp.comp1), (df_EEG_atyp.RS_somato,factor_scores_atyp.comp2),(df_EEG_atyp.RS_somato_lat,factor_scores_atyp.comp2),(df_EEG_atyp.dev_MMN,factor_scores_atyp.comp2),(df_EEG_atyp.dev_MMN_lat,factor_scores_atyp.comp2),(df_EEG_atyp.pom_MMN,factor_scores_atyp.comp2),(df_EEG_atyp.pom_MMN_lat,factor_scores_atyp.comp2),(df_EEG_atyp.RS_frontal,factor_scores_atyp.comp2),(df_EEG_atyp.RS_frontal_lat,factor_scores_atyp.comp2),(df_EEG_atyp.dev_P300,factor_scores_atyp.comp2),(df_EEG_atyp.dev_P300_lat,factor_scores_atyp.comp2),(df_EEG_atyp.pom_P300,factor_scores_atyp.comp2),(df_EEG_atyp.pom_P300_lat,factor_scores_atyp.comp2),(df_EEG_atyp.omi_amp,factor_scores_atyp.comp2),(df_EEG_atyp.omi_lat,factor_scores_atyp.comp2)]


df_p_value_corr = pd.DataFrame(columns=['p_values', 'statistics'], index=['typ_RS_somato_comp1','typ_rs_latency_somato_comp1','typ_MMN_somato_comp1','typ_dev_latency_somato_comp1','typ_MMNpom_somato_comp1','typ_pom_latency_somato_comp1','typ_RS_frontal_comp1','typ_rs_latency_frontal_comp1','typ_MMN_frontal_comp1','typ_dev_latency_frontal_comp1','typ_MMNpom_frontal_comp1','typ_pom_latency_frontal_comp1', 'typ_om_amp_comp1','typ_om_lat_comp1','typ_RS_somato_comp2','typ_rs_latency_somato_comp2','typ_MMN_somato_comp2','typ_dev_latency_somato_comp2','typ_MMNpom_somato_comp2','typ_pom_latency_somato_comp2','typ_RS_frontal_comp2','typ_rs_latency_frontal_comp2','typ_MMN_frontal_comp2','typ_dev_latency_frontal_comp2','typ_MMNpom_frontal_comp2','typ_pom_latency_frontal_comp2','typ_om_amp_comp2','typ_om_lat_comp2','atyp_RS_somato_comp1','atyp_rs_latency_somato_comp1','atyp_MMN_somato_comp1','atyp_dev_latency_somato_comp1','atyp_MMNpom_somato_comp1','atyp_pom_latency_somato_comp1','atyp_RS_frontal_comp1','atyp_rs_latency_frontal_comp1','atyp_MMN_frontal_comp1','atyp_dev_latency_frontal_comp1','atyp_MMNpom_frontal_comp1','atyp_pom_latency_frontal_comp1','atyp_om_amp_comp1','atyp_om_lat_comp1', 'atyp_RS_somato_comp2','atyp_rs_latency_somato_comp2','atyp_MMN_somato_comp2','atyp_dev_latency_somato_comp2','atyp_MMNpom_somato_comp2','atyp_pom_latency_somato_comp2','atyp_RS_frontal_comp2','atyp_rs_latency_frontal_comp2','atyp_MMN_frontal_comp2','atyp_dev_latency_frontal_comp2','atyp_MMNpom_frontal_comp2','atyp_pom_latency_frontal_comp2','atyp_om_amp_comp2','atyp_om_lat_comp2',])

p_values_corr = []
for ind, samples in enumerate(tests):
  sample1 = samples[0]
  sample2 = samples[1]
  r, p = pearsonr(sample1, sample2)
  df_p_value_corr.iloc[ind] = p, r
  p_values_corr.append(p)


p_values_corr = np.array(p_values_corr)

significant, corrected_p_values_corr, _, _ = multipletests(p_values_corr, method='fdr_by')

df_p_value_corr['significant'] = significant
df_p_value_corr['corrected'] = corrected_p_values_corr
df_p_value_corr

df_p_value_corr.to_excel("data/p_value_corr.xlsx")
