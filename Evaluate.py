import json
from DetectorModel import Detect,PreProcess
from DataProcessing import DataProcessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from time import sleep

with open('dum.json') as f:
    data = json.load(f)

IOUArray=np.zeros((10,40))
obj=DataProcessing()


d_count=0
while True:
	yieldData=obj.InputsAndLabels(data)
	try:
		for wav, label in yieldData:
			#print(wav)
			prediction=Detect(wav,Delta=13,PowerFraction=0.7)
			print("total frames of honks are",np.sum(prediction))
	except:
		print("getting new batch of signal")		