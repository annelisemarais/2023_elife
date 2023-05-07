"""
author = 'Anne-Lise Marais, see annelisemarais.github.io'
publication = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old. eLife'
corresponding author = 'nadege.roche@unicaen.fr'

This code plot amplitudes and latencies obtained in step 3. 
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
##VIOLINPLOT
#################

all_age_stats = pd.read_excel("data/all_age_stats.xlsx")

####OMISSION

f, ax = plt.subplots(figsize=(4,6))
f.suptitle('omi somatosensory latency', fontsize = 18)
sns.violinplot(x=all_age_stats['age'], y=all_age_stats['omi_amp'], orient="v", color="white", cut=0)
sns.swarmplot(x=all_age_stats['age'], y=all_age_stats['omi_amp'], orient="v", size=7, palette="Reds")
ax.set_xlabel('')
ax.set_xticklabels(["Two","Four","Six"], fontsize=16)
ax.set_ylim(-15, 8)
ax.set_ylabel("Amplitude (µV)", fontsize = 16)
ax.tick_params(axis='y', labelsize=14)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
#plt.savefig("figures/violin/omi_somato_byage.png")
plt.show()

f, ax = plt.subplots(figsize=(4,6))
f.suptitle('omi somatosensory latency', fontsize = 18)
sns.violinplot(x=all_age_stats['age'], y=all_age_stats['omi_lat'], orient="v", color="white", cut=0)
sns.swarmplot(x=all_age_stats['age'], y=all_age_stats['omi_lat'], orient="v", size=7, palette="Reds")
ax.set_xlabel('')
ax.set_xticklabels(["Two","Four","Six"], fontsize=16)
ax.set_ylim(560, 770)
plt.yticks(range(600,850,50),('100','150','200','250','300'), fontsize=13)
ax.set_ylabel("Latency poststimulation (ms)", fontsize = 16)
ax.tick_params(axis='y', labelsize=14)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
#plt.savefig("figures/violin/rs_somato_lat.png")
plt.show()




####OTHER than omission

####Amplitudes

###2 years old

def plot_amplitude(data, age, condition1, condition2, color1, color2, tick1, tick2):
	mydata = data[data['age']==age]
	f, ax = plt.subplots(figsize=(3,6))
	sns.violinplot(data=mydata[[condition1,condition2]], orient="v", color="white", cut=0)
	sns.swarmplot(data=mydata[[condition1,condition2]], orient="v", size=7, palette={condition1:color1, condition2:color2})
	ax.set_xlabel('')
	ax.set_ylim(-15, 14)
	ax.set_xticklabels([tick1,tick2], fontsize=16)
	ax.set_ylabel("Amplitude (µV)", fontsize = 16)
	ax.tick_params(axis='y', labelsize=14)
	ax.invert_yaxis()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)


plot_amplitude(all_age_stats,2,'fam_somato','con_somato',(0.44,0.68,0.28),(0.44,0.19,0.63),"Fam","Con")
plt.suptitle('Somatosensory RS 2', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/two_rs_somato.png")
plt.show()

plot_amplitude(all_age_stats,2,'fam_frontal','con_frontal',(0.44,0.68,0.28),(0.44,0.19,0.63),"Fam","Con")
plt.suptitle('Frontal RS 2', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/two_rs_frontal.png")
plt.show()

plot_amplitude(all_age_stats,2,'stddev_somato','dev_somato',"grey",(0.11,0.11,1),"Std","Dev")
plt.suptitle('Somatosensory Deviance 2', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/two_dev_somato.png")
plt.show()

plot_amplitude(all_age_stats,2,'stddev_frontal','dev_frontal',"grey",(0.11,0.11,1),"Std","Dev")
plt.suptitle('Frontal Deviance 2', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/two_dev_frontal.png")
plt.show()

plot_amplitude(all_age_stats,2,'stdpom_somato','pom_somato',"grey",(1,0.5,0),"Std","Pom")
plt.suptitle('Somatosensory postomission 2', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/two_pom_somato.png")
plt.show()

plot_amplitude(all_age_stats,2,'stdpom_frontal','pom_frontal',"grey",(1,0.5,0),"Std","Pom")
plt.suptitle('Frontal postomission 2', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/two_pom_frontal.png")
plt.show()

####4 years old

plot_amplitude(all_age_stats,4,'fam_somato','con_somato',(0.44,0.68,0.28),(0.44,0.19,0.63),"Fam","Con")
plt.suptitle('Somatosensory RS 4', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/four_rs_somato.png")
plt.show()

plot_amplitude(all_age_stats,4,'fam_frontal','con_frontal',(0.44,0.68,0.28),(0.44,0.19,0.63),"Fam","Con")
plt.suptitle('Frontal RS 4', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/four_rs_frontal.png")
plt.show()

plot_amplitude(all_age_stats,4,'stddev_somato','dev_somato',"grey",(0.11,0.11,1),"Std","Dev")
plt.suptitle('Somatosensory Deviance 4', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/four_dev_somato.png")
plt.show()

plot_amplitude(all_age_stats,4,'stddev_frontal','dev_frontal',"grey",(0.11,0.11,1),"Std","Dev")
plt.suptitle('Frontal Deviance 4', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/four_dev_frontal.png")
plt.show()

plot_amplitude(all_age_stats,4,'stdpom_somato','pom_somato',"grey",(1,0.5,0),"Std","Pom")
plt.suptitle('Somatosensory postomission 4', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/four_pom_somato.png")
plt.show()

plot_amplitude(all_age_stats,4,'stdpom_frontal','pom_frontal',"grey",(1,0.5,0),"Std","Pom")
plt.suptitle('Frontal postomission 4', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/four_pom_frontal.png")
plt.show()


####6 years old

plot_amplitude(all_age_stats,6,'fam_somato','con_somato',(0.44,0.68,0.28),(0.44,0.19,0.63),"Fam","Con")
plt.suptitle('Somatosensory RS 6', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/six_rs_somato.png")
plt.show()

plot_amplitude(all_age_stats,6,'fam_frontal','con_frontal',(0.44,0.68,0.28),(0.44,0.19,0.63),"Fam","Con")
plt.suptitle('Frontal RS 6', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/six_rs_frontal.png")
plt.show()

plot_amplitude(all_age_stats,6,'stddev_somato','dev_somato',"grey",(0.11,0.11,1),"Std","Dev")
plt.suptitle('Somatosensory Deviance 6', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/six_dev_somato.png")
plt.show()

plot_amplitude(all_age_stats,6,'stddev_frontal','dev_frontal',"grey",(0.11,0.11,1),"Std","Dev")
plt.suptitle('Frontal Deviance 6', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/six_dev_frontal.png")
plt.show()

plot_amplitude(all_age_stats,6,'stdpom_somato','pom_somato',"grey",(1,0.5,0),"Std","Pom")
plt.suptitle('Somatosensory postomission 6', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/six_pom_somato.png")
plt.show()

plot_amplitude(all_age_stats,6,'stdpom_frontal','pom_frontal',"grey",(1,0.5,0),"Std","Pom")
plt.suptitle('Frontal postomission 6', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/six_pom_frontal.png")
plt.show()



####Latencies

def plot_latencies(data, condition, ROI):
  f, ax = plt.subplots(figsize=(4,6))
  sns.violinplot(x=data['age'], y=data[condition], orient="v", color="white", cut=0)
  sns.swarmplot(x=data['age'], y=data[condition], orient="v", size=7)
  ax.set_xlabel('')
  ax.set_xticklabels(["Two","Four","Six"], fontsize=16)
  
  if ROI == 'somato':
    plt.yticks(range(200,500,50),('100','150','200','250','300','350'), fontsize=13)
    ax.set_ylim(180, 470)
  else:
    plt.yticks(range(400,1000,100),('300','400','500','600','700','800'), fontsize=13)
    ax.set_ylim(380, 920)

  ax.set_ylabel("Latency poststimulation (ms)", fontsize = 16)
  ax.tick_params(axis='y', labelsize=14)
  ax.invert_yaxis()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  

plot_latencies(all_age_stats,'rs_latency_somato','somato')
plt.suptitle('RS somato latency', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/lat_rs_somato.png")
plt.show()

plot_latencies(all_age_stats,'rs_latency_frontal','frontal')
plt.suptitle('RS frontal latency', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/lat_rs_frontal.png")
plt.show()

plot_latencies(all_age_stats,'dev_latency_somato','somato')
plt.suptitle('Dev somato latency', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/lat_dev_somato.png")
plt.show()

plot_latencies(all_age_stats,'dev_latency_frontal','frontal')
plt.suptitle('Dev frontal latency', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/lat_dev_frontal.png")
plt.show()

plot_latencies(all_age_stats,'pom_latency_somato','somato')
plt.suptitle('Pom somato latency', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/lat_pom_somato.png")
plt.show()

plot_latencies(all_age_stats,'pom_latency_frontal','frontal')
plt.suptitle('Pom frontal latency', fontsize = 18)
plt.tight_layout()
#plt.savefig("figures/violin/lat_pom_frontal.png")
plt.show()