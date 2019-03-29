import os
import glob
import time
import numpy as np
from scipy.io import wavfile
import csv
from operator import eq
from scipy import stats
from python_speech_features import mfcc
from scipy.io import wavfile
from scipy import signal
import librosa

class DataProcessing():
	def __init__(self):
		self.Names2Label=	{	
								"air_conditioner":0, "car_horn":1,
								"children_playing":0,"dog_bark":0,
								"drilling":0,"engine_idling":0,
								"gun_shot":0,"jackhammer":0,
								"siren":0,"street_music":0
							}
		self.NumClasses=2
		self.TotalEpochs=20
		
		self.SamplingFrequency=44100
		self.SignalLength=441000
		self.SamplesInWindow=1024

		
					
	
	def FetchAnnotation(self,Name):
		"""
		This is to generate time frame level one hot encoding 
		of the class labels.
		"""
		Annotation=np.zeros(self.SignalLength)
		with open(Name) as f:
			reader = csv.reader(f, delimiter="\t")
			AnnotationCSV = list(reader)
		
		for category in AnnotationCSV:
			Annotation[	int(self.SamplingFrequency*float(category[0])):
						int(self.SamplingFrequency*float(category[1]))] =self.Names2Label[category[2]]
		Annotation=Annotation[:self.SamplesInWindow*(self.SignalLength//self.SamplesInWindow)]				
		Annotation = Annotation.reshape((-1,self.SamplesInWindow))

		Time2WindowLabel=np.zeros((Annotation.shape[0],self.NumClasses))
		
		for i in range(Annotation.shape[0]):
			Time2WindowLabel[i,:]=np.eye(self.NumClasses)[int(stats.mode(Annotation[i,:]).mode[0])] 
			
		return Time2WindowLabel

	def FetchSignal(self,Name):
		SamplingFrequency, Data = wavfile.read(Name)
		Data=Data.astype(float)
		stft = librosa.stft(Data, n_fft=4*self.SamplesInWindow, hop_length=self.SamplesInWindow)
		stft_magnitude, stft_phase = librosa.magphase(stft)
		print(Name)
		#print("stft size",stft_magnitude.shape)
		"""
		Data=Data[:self.SignalLength].astype(float)
		MelSpectrum = librosa.feature.melspectrogram(y=Data,sr=SamplingFrequency,n_fft=self.SamplesInWindow,hop_length=self.SamplesInWindow,fmax=22050)
		epsilon=1e-10
		MelLogSpectrum=np.log(MelSpectrum+epsilon).T
		MelLogSpectrum=MelLogSpectrum[:430,:]
		"""
		return stft_magnitude

	def InputsAndLabels(self,JsonDict):
		WaveFileNames=JsonDict.keys()
		for WaveFileName in WaveFileNames:
			MelFeatures=self.FetchSignal(WaveFileName)
			LabelArray=self.FetchAnnotation(JsonDict[WaveFileName])
			
			yield MelFeatures,LabelArray

	
