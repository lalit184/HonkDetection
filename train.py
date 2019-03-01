from Parameters import Parameter
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from DataProcessing import DataProcessing
from model import LSTM
import numpy as np
import time
from matplotlib import pyplot as plt




models = LSTM()
loss_function = nn.BCELoss(size_average=True,reduce=True)
gamma=3
models.load_state_dict(torch.load("mymodel.pt"))

models.eval()
optimizer = optim.Adagrad(models.parameters())

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
TrainLossesBCE=[]
TrainLossesFoc=[]
ValidateLossesBCE=[]
ValidateLossesFoc=[]
DataObject=DataProcessing()
np.set_printoptions(threshold=np.nan)

def train():
	for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
		print("Beginning as a batch")
		StepsOfEpoch=0
		DataMethodObject=DataObject.FetchTrainInputsAndLabels()
		LossesAverageBCE=0
		LossesAverageFoc=0
		for wav, label in DataMethodObject:
			then=time.time()
			StepsOfEpoch+=1
			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			models.zero_grad()
			models.init_hidden()
			# Also, we need to clear out the hidden state of the LSTM,
			# detaching it from its history on the last instance.
			output = models(torch.tensor(wav).float())
			#print(output.detach().numpy())
			#print(label)
			#print(output)
			#print(label)
			# Step 4. Compute the loss, gradients, and update the parameters by
			#  calling optimizer.step()
			lossBCE = loss_function(output, torch.tensor(label).float())
			lossFocal = loss_function(output**((1-output)**gamma), torch.tensor(label).float())
			loss=lossBCE+lossFocal

			loss.backward()
			optimizer.step()
			now=time.time()
			LossesAverageBCE+=lossBCE.detach().numpy()/6000
			LossesAverageFoc+=lossFocal.detach().numpy()/6000
			print("Epoch:",epoch,"Step:",StepsOfEpoch," of 6000 steps, LossBCE:",lossBCE.detach().numpy(),"Focal",lossFocal.detach().numpy(),"Time taken is ",now-then)
			
		TrainLossesBCE.append(LossesAverageBCE)	
		TrainBcePlot,=plt.plot(np.array(TrainLossesBCE),color='red')

		TrainLossesFoc.append(LossesAverageFoc)	
		TrainFocPlot,=plt.plot(np.array(TrainLossesFoc),color='blue')

		validate()
		ValidateBcePlot,=plt.plot(np.array(ValidateLossesBCE),color='green')
		ValidateFocPlot,=plt.plot(np.array(ValidateLossesFoc), color='yellow')
		plt.legend(handles=[TrainFocPlot,TrainBcePlot,ValidateFocPlot,ValidateBcePlot])	
		plt.xlabel("Epochs")
		plt.ylabel("loss")
		plt.savefig("Losses.png")
		plt.close()

		torch.save(models.state_dict(),"mymodel.pt")
		
def validate():
	print("Beginning as a batch")
	DataMethodObject=DataObject.FetchValidateInputsAndLabels()
	LossesAverageFoc=0
	LossesAverageBCE=0
	StepsOfEpoch=0
		
	for wav, label in DataMethodObject:
		then=time.time()
		StepsOfEpoch+=1
		# Step 1. Remember that Pytorch accumulates gradients.
		# We need to clear them out before each instance
		models.zero_grad()
		models.init_hidden()
		# Also, we need to clear out the hidden state of the LSTM,
		# detaching it from its history on the last instance.
		output = models(torch.tensor(wav).float())
			
		#print(output.detach().numpy())
		print(label*output.detach().numpy())
		# Step 4. Compute the loss, gradients, and update the parameters by
		#  calling optimizer.step()
		lossBCE = loss_function(output, torch.tensor(label).float())
		lossFocal = loss_function(output**((1-output)**gamma), torch.tensor(label).float())
		loss=lossBCE+lossFocal

		now=time.time()
		LossesAverageBCE+=lossBCE.detach().numpy()/2000
		LossesAverageFoc+=lossFocal.detach().numpy()/2000
		print("Step:",StepsOfEpoch," of 2000 steps of validation, Loss:",loss.detach().numpy(),"Time taken is ",now-then)
	ValidateLossesBCE.append(LossesAverageBCE)
	ValidateLossesFoc.append(LossesAverageFoc)	
#train()
validate()	
			