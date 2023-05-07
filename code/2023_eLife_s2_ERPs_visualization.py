"""
author = 'Anne-Lise Marais, see annelisemarais.github.io'
publication = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old. eLife'
corresponding author = 'nadege.roche@unicaen.fr'


This code plots ERPs by age and condition. 
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
##ERPs visualization
#################

#load data
df_ERPdata = pd.read_csv("data/df_ERP_typ.csv", index_col=0, header=0)
df_omi = pd.read_csv("data/df_omi_typ.csv", index_col=0, header=0)

#Choose electrodes for Region Of Interest (ROI)
somato = [28,29,35,36,41,42,47,52]
frontal = [5,6,7,12,13,106,112,129] 


def df2plot(df,condition,electrode,age):
  data = df[df['condition']==condition][df['electrode'].isin(electrode)][df['age']==age]
  data = data.groupby('sub').mean()
  data = data.drop(['condition','electrode','age'], axis=1)
  return data

two_fam_somato = df2plot(df_ERPdata,3,somato,2)
two_con_somato = df2plot(df_ERPdata,1,somato,2)
two_dev_somato = df2plot(df_ERPdata,2,somato,2)
two_std_somato = df2plot(df_ERPdata,7,somato,2)
two_pom_somato = df2plot(df_ERPdata,5,somato,2)

two_fam_frontal = df2plot(df_ERPdata,3,frontal,2)
two_con_frontal = df2plot(df_ERPdata,1,frontal,2)
two_dev_frontal = df2plot(df_ERPdata,2,frontal,2)
two_std_frontal = df2plot(df_ERPdata,7,frontal,2)
two_pom_frontal = df2plot(df_ERPdata,5,frontal,2)

four_fam_somato = df2plot(df_ERPdata,3,somato,4)
four_con_somato = df2plot(df_ERPdata,1,somato,4)
four_dev_somato = df2plot(df_ERPdata,2,somato,4)
four_std_somato = df2plot(df_ERPdata,7,somato,4)
four_pom_somato = df2plot(df_ERPdata,5,somato,4)

four_fam_frontal = df2plot(df_ERPdata,3,frontal,4)
four_con_frontal = df2plot(df_ERPdata,1,frontal,4)
four_dev_frontal = df2plot(df_ERPdata,2,frontal,4)
four_std_frontal = df2plot(df_ERPdata,7,frontal,4)
four_pom_frontal = df2plot(df_ERPdata,5,frontal,4)

six_fam_somato = df2plot(df_ERPdata,3,somato,6)
six_con_somato = df2plot(df_ERPdata,1,somato,6)
six_dev_somato = df2plot(df_ERPdata,2,somato,6)
six_std_somato = df2plot(df_ERPdata,7,somato,6)
six_pom_somato = df2plot(df_ERPdata,5,somato,6)
 
six_fam_frontal = df2plot(df_ERPdata,3,frontal,6)
six_con_frontal = df2plot(df_ERPdata,1,frontal,6)
six_dev_frontal = df2plot(df_ERPdata,2,frontal,6)
six_std_frontal = df2plot(df_ERPdata,7,frontal,6)
six_pom_frontal = df2plot(df_ERPdata,5,frontal,6)

two_omi = df_omi[df_omi['age']==2][df_omi['electrode'].isin(somato)]
two_omi = two_omi.groupby('sub').mean()
two_omi = two_omi.drop(['condition','electrode','age'], axis=1)

four_omi = df_omi[df_omi['age']==4][df_omi['electrode'].isin(somato)]
four_omi = four_omi.groupby('sub').mean()
four_omi = four_omi.drop(['condition','electrode','age'], axis=1)

six_omi = df_omi[df_omi['age']==6][df_omi['electrode'].isin(somato)]
six_omi = six_omi.groupby('sub').mean()
six_omi = six_omi.drop(['condition','electrode','age'], axis=1)

def plot_ERPs(condition1,condition2,color1,color2):
  plt.plot(np.mean(condition1), c=(color1))
  plt.plot(np.mean(condition2), c=(color2))
  plt.xticks(range(99,999,199),('0','200','400','600','800'), fontsize=20)
  plt.xlim(0,999)
  plt.ylim(-7,4)
  plt.gca().invert_yaxis()
  plt.xlabel("Time (ms)", fontsize = 20)
  plt.ylabel("Amplitude (µV)", fontsize = 20)
  plt.gca().tick_params(axis='x', labelsize=20, pad=15)
  plt.gca().tick_params(axis='y', labelsize=20, pad=15)
  ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

def plot_omi_ERPs(condition):
  plt.plot(np.mean(condition), c='r')
  plt.xticks(range(99,999,199),('-400','-200','0','200','400'), fontsize=20)
  plt.xlim(0,999)
  plt.ylim(-7,4)
  plt.gca().invert_yaxis()
  plt.xlabel("Time (ms)", fontsize = 20)
  plt.ylabel("Amplitude (µV)", fontsize = 20)
  plt.gca().tick_params(axis='x', labelsize=20, pad=15)
  plt.gca().tick_params(axis='y', labelsize=20, pad=15)
  ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

###ERP 2 years old

plot_ERPs(two_fam_somato,two_con_somato,(0.44, 0.68, 0.28),(0.44, 0.19, 0.63))
plt.title("Somatosensory repetition suppression 2 yo")
plt.legend(['Familiarization', 'Control'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/two_rs_somato.png")
plt.show()

plot_ERPs(two_fam_frontal,two_con_frontal,(0.44, 0.68, 0.28),(0.44, 0.19, 0.63))
plt.title("Frontal repetition suppression 2 yo")
plt.legend(['Familiarization', 'Control'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/two_rs_frontal.png")
plt.show()

plot_ERPs(two_std_somato,two_dev_somato,'k',(0.11,0.1,1))
plt.title("Somatosensory deviance 2 yo")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/two_dev_somato.png")
plt.show()

plot_ERPs(two_std_frontal,two_dev_frontal,'k',(0.11,0.1,1))
plt.title("Frontal deviance 2 yo")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/two_dev_frontal.png")
plt.show()

plot_ERPs(two_std_somato,two_pom_somato,'k',(0.11,0.1,1))
plt.title("Somatosensory postomission 2 yo")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/two_pom_somato.png")
plt.show()

plot_ERPs(two_std_frontal,two_pom_frontal,'k',(0.11,0.1,1))
plt.title("Frontal postomission 2 yo")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/two_pom_frontal.png")
plt.show()

plot_omi_ERPs(two_omi)
plt.title("Somatosensory omission 2 yo")
plt.tight_layout()
#plt.savefig("figures/ERP/two_omi.png")
plt.show()

###ERP 4 years old

plot_ERPs(four_fam_somato,four_con_somato,(0.44, 0.68, 0.28),(0.44, 0.19, 0.63))
plt.title("Somatosensory repetition suppression 4 yo")
plt.legend(['Familiarization', 'Control'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/four_rs_somato.png")
plt.show()

plot_ERPs(four_fam_frontal,four_con_frontal,(0.44, 0.68, 0.28),(0.44, 0.19, 0.63))
plt.title("Frontal repetition suppression 4 yo")
plt.legend(['Familiarization', 'Control'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/four_rs_frontal.png")
plt.show()

plot_ERPs(four_std_somato,four_dev_somato,'k',(0.11,0.1,1))
plt.title("Somatosensory deviance 4 yo")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/four_dev_somato.png")
plt.show()

plot_ERPs(four_std_frontal,four_dev_frontal,'k',(0.11,0.1,1))
plt.title("Frontal deviance 4 yo")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/four_dev_frontal.png")
plt.show()

plot_ERPs(four_std_somato,four_pom_somato,'k',(0.11,0.1,1))
plt.title("Somatosensory postomission 4 yo")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/four_pom_somato.png")
plt.show()

plot_ERPs(four_std_frontal,four_pom_frontal,'k',(0.11,0.1,1))
plt.title("Frontal postomission 4 yo")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/four_pom_frontal.png")
plt.show()

plot_omi_ERPs(four_omi)
plt.title("Somatosensory omission 4 yo")
plt.tight_layout()
#plt.savefig("figures/ERP/four_omi.png")
plt.show()

###ERP 6 years old

plot_ERPs(six_fam_somato,six_con_somato,(0.44, 0.68, 0.28),(0.44, 0.19, 0.63))
plt.title("Somatosensory repetition suppression 6 yo")
plt.legend(['Familiarization', 'Control'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/six_rs_somato.png")
plt.show()

plot_ERPs(six_fam_frontal,six_con_frontal,(0.44, 0.68, 0.28),(0.44, 0.19, 0.63))
plt.title("Frontal repetition suppression 6 yo")
plt.legend(['Familiarization', 'Control'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/six_rs_frontal.png")
plt.show()

plot_ERPs(six_std_somato,six_dev_somato,'k',(0.11,0.1,1))
plt.title("Somatosensory deviance 6 yo")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/six_dev_somato.png")
plt.show()

plot_ERPs(six_std_frontal,six_dev_frontal,'k',(0.11,0.1,1))
plt.title("Frontal deviance 6 yo")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/six_dev_frontal.png")
plt.show()

plot_ERPs(six_std_somato,six_pom_somato,'k',(0.11,0.1,1))
plt.title("Somatosensory postomission 6 yo")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/six_pom_somato.png")
plt.show()

plot_ERPs(six_std_frontal,six_pom_frontal,'k',(0.11,0.1,1))
plt.title("Frontal postomission 6 yo")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
#plt.savefig("figures/ERP/six_pom_frontal.png")
plt.show()

plot_omi_ERPs(six_omi)
plt.title("Somatosensory omission 6 yo")
plt.tight_layout()
#plt.savefig("figures/ERP/six_omi.png")
plt.show()
