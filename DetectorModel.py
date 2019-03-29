import numpy as np
import librosa
import sys
np.set_printoptions(threshold=sys.maxsize)
def PreProcess(stft_magnitude,PowerFraction):
	
	SamplesInWindow=1024
	
	Signal=Signal.astype(float)
	print(Signal.shape)
	stft = librosa.stft(Signal, n_fft=4*SamplesInWindow, hop_length=SamplesInWindow)
	stft_magnitude, stft_phase = librosa.magphase(stft)
	stft_magnitude=stft_magnitude[:,50:]
	
	RescaledSTFT=np.zeros_like(stft_magnitude).astype(float)
	for i in range(len(stft_magnitude)):
		Snippet=stft_magnitude[i,:].astype(float)
		if np.max(Snippet)>1e5:
			RescaledSTFT[i,:]=(Snippet-np.min(Snippet))/(np.max(Snippet)-np.min(Snippet))
		else:
			RescaledSTFT[i,:]=np.zeros_like(Snippet)
		boolean=RescaledSTFT[i,:]>PowerFraction*np.max(RescaledSTFT[i,:])
		boolean=boolean.astype(float)
		RescaledSTFT[i,:]=RescaledSTFT[i,:]*boolean

	return RescaledSTFT[:2048,:430]


def Detect(Signal,Delta,PowerFraction):
	Magnitude=PreProcess(Signal,PowerFraction)
	print(Magnitude.shape)
	Magnitude=Magnitude.T
	Prediction=np.zeros(430)
	for j in range(430):
		Snippet=Magnitude[j,:]
		for i in range(2048):
			NonZeroELems=0
			while Snippet[i]>0:
				#print("hey")
				NonZeroELems+=1
				#print(NonZeroELems)
				i+=1
				if i>2048-1:
					break
			#print(NonZeroELems)
			if NonZeroELems < Delta and NonZeroELems >2:
				Prediction[j]=1
				break
	return Prediction			



	

