import os
import glob
import time
import numpy as np
from scipy.io import wavfile
import csv
from Parameters import Parameter
from operator import eq
from scipy import stats
from python_speech_features import mfcc


class DataProcessing(Parameter):
	def __init__(self):
		super(DataProcessing, self).__init__()
		self.Names2Label=	{	
								"air_conditioner":0, "car_horn":1,
								"children_playing":0,"dog_bark":0,
								"drilling":0,"engine_idling":0,
								"gun_shot":0,"jackhammer":0,
								"siren":0,"street_music":0
							}
		self.TrainWavFileDirectory='./URBAN-SED_v2.0.0/audio/train/'
		self.TrainTxtAnnotationDirectory='./URBAN-SED_v2.0.0/annotations/train/'
		
		self.ValidateWavFileDirectory='./URBAN-SED_v2.0.0/audio/validate/'
		self.ValidateTxtAnnotationDirectory='./URBAN-SED_v2.0.0/annotations/validate/'
			
	
	def FetchAnnotation(self,Name):
		"""
		We take the annotation data and generate a one hot label for each 
		element in the sound vector.

		For a sound of type 	 [aaaabbbbaaa]
		we have the annotation  [[00001111000],
								 [11110000111]]
		which is a psuedo one hot vector annotationn
		"""
		Annotation=np.zeros(int(self.SignalLength/self.SubSamplingRate))
		with open(Name) as f:
			reader = csv.reader(f, delimiter="\t")
			AnnotationCSV = list(reader)
		
		for category in AnnotationCSV:
			Annotation[	int(self.SamplingFrequency*float(category[0])/self.SubSamplingRate):
						int(self.SamplingFrequency*float(category[1])/self.SubSamplingRate)] =self.Names2Label[category[2]]
		Annotation = Annotation.reshape((-1,int(self.WindowTime*self.SamplingFrequency/self.SubSamplingRate)))

		Time2WindowLabel=np.zeros((Annotation.shape[0],self.NumClasses))
		
		for i in range(Annotation.shape[0]):
			Time2WindowLabel[i,:]=np.eye(self.NumClasses)[int(stats.mode(Annotation[i,:]).mode[0])] 
		return Time2WindowLabel

	def FetchSignal(self,Name):
		SamplingFrequency, Data = wavfile.read(Name)
		Data=Data[::self.SubSamplingRate]
		Data=Data[:441000]
		Data=mfcc(	signal=Data,samplerate=self.SamplingFrequency/self.SubSamplingRate,nfft=self.WindowSize,nfilt=100,
					winlen=self.WindowTime,winstep=self.WindowStep,winfunc=np.hamming,numcep=self.NumCep,highfreq=2000,lowfreq=10)

		"""
		the size of Data is  SignalLength/WindowTime,NumCep
		"""
		return Data

	def FetchTrainInputsAndLabels(self):
		WaveFilesList=[]
		AnnotationFileList=[]

		for file in glob.glob(self.TrainWavFileDirectory+"*.wav"):
			WaveFilesList.append(file)

		for file in glob.glob(self.TrainTxtAnnotationDirectory+"*.txt"):
			AnnotationFileList.append(file)

		WaveFilesList.sort()	
		AnnotationFileList.sort()
		#WaveFilesList=WaveFilesList[:2000]
		#AnnotationFileList=AnnotationFileList[:2000]

		
		for WavFileName,AnnotationFileName in zip(WaveFilesList,AnnotationFileList):
			"""

			Some reshaping op

			"""
			#print(WavFileName,AnnotationFileName)
			WaveArray=self.FetchSignal(WavFileName)
			LabelArray=self.FetchAnnotation(Name=AnnotationFileName)
			
			yield WaveArray,LabelArray[:,1]
						
	def FetchValidateInputsAndLabels(self):
		WaveFilesList=[]
		AnnotationFileList=[]

		for file in glob.glob(self.ValidateWavFileDirectory+"*.wav"):
			WaveFilesList.append(file)

		for file in glob.glob(self.ValidateTxtAnnotationDirectory+"*.txt"):
			AnnotationFileList.append(file)

		WaveFilesList.sort()	
		AnnotationFileList.sort()
		#WaveFilesList=WaveFilesList[:665]
		#AnnotationFileList=AnnotationFileList[:665]

		
		for WavFileName,AnnotationFileName in zip(WaveFilesList,AnnotationFileList):
			"""

			Some reshaping op

			"""
			#print(WavFileName,AnnotationFileName)
			WaveArray=self.FetchSignal(WavFileName)
			LabelArray=self.FetchAnnotation(Name=AnnotationFileName)
			yield WaveArray,LabelArray[:,1]
						
				


			
				



	
