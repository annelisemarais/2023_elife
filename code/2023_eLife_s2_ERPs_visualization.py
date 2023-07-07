"""
author = 'Anne-Lise Marais, see annelisemarais.github.io'
publication = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old. eLife'
corresponding author = 'nadege.roche@unicaen.fr'


This code plots ERPs by age and condition. 
"""

print(__doc__)



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#################
##ERPs visualization
#################

#load data
typical_TS = pd.read_csv("data/typical_TS.csv", index_col=0, header=0)
typical_omi = pd.read_csv("data/typical_omi.csv", index_col=0, header=0)

#Choose electrodes for Region Of Interest (ROI)
somato = [28,29,35,36,41,42,47,52]
frontal = [5,6,7,12,13,106,112,129] 


def df2plot(df,condition,electrode,age):
  data = df[df['condition']==condition][df['electrode'].isin(electrode)][df['age']==age]
  data = data.groupby('sub').mean()
  data = data.drop(['condition','electrode','age'], axis=1)
  return data

two_fam_somato = df2plot(typical_TS,3,somato,2)
two_con_somato = df2plot(typical_TS,1,somato,2)
two_dev_somato = df2plot(typical_TS,2,somato,2)
two_std_somato = df2plot(typical_TS,7,somato,2)
two_pom_somato = df2plot(typical_TS,5,somato,2)

two_fam_frontal = df2plot(typical_TS,3,frontal,2)
two_con_frontal = df2plot(typical_TS,1,frontal,2)
two_dev_frontal = df2plot(typical_TS,2,frontal,2)
two_std_frontal = df2plot(typical_TS,7,frontal,2)
two_pom_frontal = df2plot(typical_TS,5,frontal,2)

four_fam_somato = df2plot(typical_TS,3,somato,4)
four_con_somato = df2plot(typical_TS,1,somato,4)
four_dev_somato = df2plot(typical_TS,2,somato,4)
four_std_somato = df2plot(typical_TS,7,somato,4)
four_pom_somato = df2plot(typical_TS,5,somato,4)

four_fam_frontal = df2plot(typical_TS,3,frontal,4)
four_con_frontal = df2plot(typical_TS,1,frontal,4)
four_dev_frontal = df2plot(typical_TS,2,frontal,4)
four_std_frontal = df2plot(typical_TS,7,frontal,4)
four_pom_frontal = df2plot(typical_TS,5,frontal,4)


two_omi = typical_omi[typical_omi['age']==2][typical_omi['electrode'].isin(somato)]
two_omi = two_omi.drop(['condition','electrode','age'], axis=1)
two_omi = two_omi.groupby('sub').mean()


four_omi = typical_omi[typical_omi['age']==4][typical_omi['electrode'].isin(somato)]
four_omi = four_omi.drop(['condition','electrode','age'], axis=1)
four_omi = four_omi.groupby('sub').mean()


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
plt.savefig("figures/ERP/two_rs_somato.png")
plt.show()

plot_ERPs(two_fam_frontal,two_con_frontal,(0.44, 0.68, 0.28),(0.44, 0.19, 0.63))
plt.title("Frontal repetition suppression 2 yo")
plt.legend(['Familiarization', 'Control'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/two_rs_frontal.png")
plt.show()

plot_ERPs(two_std_somato,two_dev_somato,'k',(0.11,0.1,1))
plt.title("Somatosensory deviance 2 yo")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/two_dev_somato.png")
plt.show()

plot_ERPs(two_std_frontal,two_dev_frontal,'k',(0.11,0.1,1))
plt.title("Frontal deviance 2 yo")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/two_dev_frontal.png")
plt.show()

plot_ERPs(two_std_somato,two_pom_somato,'k',(1,0.5,0.1))
plt.title("Somatosensory postomission 2 yo")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/two_pom_somato.png")
plt.show()

plot_ERPs(two_std_frontal,two_pom_frontal,'k',(1,0.5,0.1))
plt.title("Frontal postomission 2 yo")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/two_pom_frontal.png")
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
plt.savefig("figures/ERP/four_rs_somato.png")
plt.show()

plot_ERPs(four_fam_frontal,four_con_frontal,(0.44, 0.68, 0.28),(0.44, 0.19, 0.63))
plt.title("Frontal repetition suppression 4 yo")
plt.legend(['Familiarization', 'Control'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/four_rs_frontal.png")
plt.show()

plot_ERPs(four_std_somato,four_dev_somato,'k',(0.11,0.1,1))
plt.title("Somatosensory deviance 4 yo")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/four_dev_somato.png")
plt.show()

plot_ERPs(four_std_frontal,four_dev_frontal,'k',(0.11,0.1,1))
plt.title("Frontal deviance 4 yo")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/four_dev_frontal.png")
plt.show()

plot_ERPs(four_std_somato,four_pom_somato,'k',(1,0.5,0.1))
plt.title("Somatosensory postomission 4 yo")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/four_pom_somato.png")
plt.show()

plot_ERPs(four_std_frontal,four_pom_frontal,'k',(1,0.5,0.1))
plt.title("Frontal postomission 4 yo")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/four_pom_frontal.png")
plt.show()

plot_omi_ERPs(four_omi)
plt.title("Somatosensory omission 4 yo")
plt.tight_layout()
plt.savefig("figures/ERP/four_omi.png")
plt.show()


################
##ATYPICAL CHILDREN
################

#load data
atypical_TS = pd.read_csv("data/atypical_TS.csv", index_col=0, header=0)
atypical_omi = pd.read_csv("data/atypical_omi.csv", index_col=0, header=0)

def df2plot_aty(df,condition,electrode):
  data = df[df['condition']==condition][df['electrode'].isin(electrode)]
  data = data.groupby('sub').mean()
  data = data.drop(['condition','electrode'], axis=1)
  return data

atyp_fam_somato = df2plot_aty(atypical_TS,3,somato)
atyp_con_somato = df2plot_aty(atypical_TS,1,somato)
atyp_dev_somato = df2plot_aty(atypical_TS,2,somato)
atyp_std_somato = df2plot_aty(atypical_TS,7,somato)
atyp_pom_somato = df2plot_aty(atypical_TS,5,somato)

atyp_fam_frontal = df2plot_aty(atypical_TS,3,frontal)
atyp_con_frontal = df2plot_aty(atypical_TS,1,frontal)
atyp_dev_frontal = df2plot_aty(atypical_TS,2,frontal)
atyp_std_frontal = df2plot_aty(atypical_TS,7,frontal)
atyp_pom_frontal = df2plot_aty(atypical_TS,5,frontal)

###ERP atypical 4yo

plot_ERPs(atyp_fam_somato,atyp_con_somato,(0.44, 0.68, 0.28),(0.44, 0.19, 0.63))
plt.title("Somatosensory repetition suppression atypical")
plt.legend(['Familiarization', 'Control'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/atyp_rs_somato.png")
plt.show()

plot_ERPs(atyp_fam_frontal,atyp_con_frontal,(0.44, 0.68, 0.28),(0.44, 0.19, 0.63))
plt.title("Frontal repetition suppression atypical")
plt.legend(['Familiarization', 'Control'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/atyp_rs_frontal.png")
plt.show()

plot_ERPs(atyp_std_somato,atyp_dev_somato,'k',(0.11,0.1,1))
plt.title("Somatosensory deviance atypical")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/atyp_dev_somato.png")
plt.show()

plot_ERPs(atyp_std_frontal,atyp_dev_frontal,'k',(0.11,0.1,1))
plt.title("Frontal deviance atypical")
plt.legend(['Standard', 'Deviant'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/atyp_dev_frontal.png")
plt.show()

plot_ERPs(atyp_std_somato,atyp_pom_somato,'k',(1,0.5,0.1))
plt.title("Somatosensory postomission atypical")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/atyp_pom_somato.png")
plt.show()

plot_ERPs(atyp_std_frontal,atyp_pom_frontal,'k',(1,0.5,0.1))
plt.title("Frontal postomission atypical")
plt.legend(['Standard', 'Postomission'], fontsize=13)
plt.tight_layout()
plt.savefig("figures/ERP/atyp_pom_frontal.png")
plt.show()


atyp_omi = atypical_omi[atypical_omi['condition']==4][atypical_omi['electrode'].isin(somato)]
atyp_omi = atyp_omi.drop(['condition','electrode'], axis=1)
atyp_omi = atyp_omi.groupby('sub').mean()

plot_omi_ERPs(atyp_omi)
plt.title("Somatosensory omission atypical")
plt.tight_layout()
plt.savefig("figures/ERP/atyp_omi.png")
plt.show()
