#Daniel Morton
#Dataset not included- owned by outside
#input is procedure codes, name
import pandas as pd
import numpy as np
import logging
import os
import re
import matplotlib.pyplot as plt
import tsne_scatter_plot
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth

#modify cluster number here
clusterNum=5

#reading in the data into datastructures
print("Loading Data...\n")
dataIn_df = pd.read_json("data.json")
namelist=[]
for col in dataIn_df.columns:
    namelist.append(col)

#gathering/prepping procedure information
ProcedureList=[]
for names in namelist:
    ProcedureList.append(dataIn_df.at['Procedures', names])

#Quick frequency transformation mostly for investigation
newRows=[]
for names in namelist:
    Proced=dataIn_df.at['Procedures', names]
    for procedures in Proced:
        section_char=procedures[0]
        system_char=procedures[1]
        root_char=procedures[2]
        region_char=procedures[3]
        row_dict={
            "Name": names,
            "Procedure": procedures,
            "Section": procedures[0],
            "System":procedures[1],
            "Root Op":procedures[2],
            "Body Region":procedures[3]}
        newRows.append(row_dict)
masterFrame = pd.DataFrame(newRows)
frequency=masterFrame.apply(pd.value_counts)
#saving metastats for future use
print("Checking and Saving Frequency Information...\n")
file_freq=open("proc_frequencies.csv","w")
file_freq.write("Variable Name, Frequency \n")
proc_frequencies = frequency[frequency["Procedure"]>=0]
procfreqList=proc_frequencies["Procedure"].tolist()
procfreqNameArr=proc_frequencies.index.values
vocab_list = list(procfreqNameArr)
procedure_total=proc_frequencies["Procedure"].sum()
file_freq.write("Procedures,"+str(procedure_total)+"\n")
for x in range(0,len(vocab_list)):
	file_freq.write(vocab_list[x]+","+str(procfreqList[x])+"\n")
#I should write a new function or object but didn't
sec_frequencies = frequency[frequency["Section"]>=0]
secfreqList=sec_frequencies["Section"].tolist()
secfreqNameArr=sec_frequencies.index.values
sec_list = list(secfreqNameArr)
file_freq.write("Section,"+str(procedure_total)+"\n")
for x in range(0,len(sec_list)):
	file_freq.write(sec_list[x]+","+str(secfreqList[x])+"\n")

sys_frequencies = frequency[frequency["System"]>=0]
sysfreqList=sys_frequencies["System"].tolist()
sysfreqNameArr=sys_frequencies.index.values
sys_list = list(sysfreqNameArr)
file_freq.write("System,"+str(procedure_total)+"\n")
for x in range(0,len(sys_list)):
	file_freq.write(sys_list[x]+","+str(sysfreqList[x])+"\n")
file_freq.close()
frequency.hist(layout=(2,3), bins=15, figsize=[8,4], xlabelsize=7,ylabelsize=8)
plt.savefig('freqdistributions.pdf')
print("Success\n")

print("Training Neural Nets and Vectorizing Procedures...\n")
#bag of words model to vectorize procedures
model = Word2Vec(min_count=1,window=741, size=150, alpha= .05, min_alpha=.00005, negative=10)
model.build_vocab(ProcedureList)
model.train(ProcedureList, total_examples=2000, epochs=50, report_delay=1)
model.init_sims(replace=True)

#Generate Plots
tsnescatterplot(model, vocab_list, "All Procedures")
print("Success\n")
modelPath='patientProcedure_vectors.wv'
model.wv.save(modelPath)

#Transform Word List
#Note the distance information from individuals is now encoded in the model
bag_dim = len(model.wv.__getitem__(ProcedureList[0][0]))
wrd_array = np.empty((0, bag_dim), dtype='f')

for x in range(0,len(namelist)):
	ProcedureList[x]
	for words in ProcedureList[x]:
		wrd_array.append(wrd_array, wrdmodel.wv.__getitem__([wrd_score]), axis=0)

#reduce dimensionality before lowering to kmeans, otherwise our space would be too large
#we also want to eliminate artificial clustering caused by high dim
reduc = PCA(n_components=15).fit_transform(wrd_array)

#after viewing the data, we believe kmeans or central points may function well
#may need to change the method
print("Starting Wandering Means")
bandwidth=estimate_bandwidth(reduc)
MS_cluster=MeanShift(bandwidth=bandwidth, bin_seeding=True)
MS_cluster.fit(wrd_array)
labels=MS_cluster.labels_
centers=MS_cluster_centers_

unique=np.unique(labels)
cluster_num=len(unique)

#We could add central value extraction, finding closest procedures
#Report what have right now
print("number of estimated clusters : %d" % unique)
file_clusters=open("proc_.csv","w")
for x in range(0,len(namelist)):
	file_clusters.write(namelist[x]+","+MS_cluster.predict()+"\n")

#print("Starting a KMeans Cluster (5)")
#sklearn.cluster.KMeans(5)

#used to view individual components
while True:
    controlString=input("Type name as it appears in file or quit:")
    test=0
    for names in namelist:
        if names==controlString:
            tsnescatterplot(model, ProcedureList[test], names)
            break
        else:
        	test+=1
    print(controlString)
    if controlString=='quit':
    	break
