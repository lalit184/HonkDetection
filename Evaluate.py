import json
from DetectorModel import Detect,PreProcess
from DataProcessing import DataProcessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
with open('./Jsons/Train.json') as f:
    data = json.load(f)
obj=DataProcessing()
yieldData=obj.InputsAndLabels(data)
avgIOU=[]
IOUArray=np.zeros((15,40))
for t in range(20):
	for delta in range(5,20):
		d_count=0
		for wav, label in yieldData:
			prediction=Detect(wav,Delta=delta,PowerFraction=0.6+0.02*t)
			intersection=prediction*label[:,1]
			union=prediction+label[:,1]-intersection
			IOU=np.sum(intersection)/(np.sum(union)+1)
			print(IOU,np.sum(intersection),(np.sum(union)+1),np.sum(label[:,1]))
			avgIOU.append(IOU)
		IOUArray[delta-5,t]=np.mean(np.array(avgIOU))

			
plt.imshow(IOUArray, cmap='hot', interpolation='nearest')
plt.show()
plt.savefig("hyperparameters.png")

