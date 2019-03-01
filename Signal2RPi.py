from gpiozero import MCP3008
import numpy as np
from Parameters import Parameters
import time

class Signal2Rpi(Parameters):
	def __init__(self):
		super(Signal2Rpi, self).__init__()
		self.GPIOPinNumber=0
		self.SamplePeriod=float(1.0/self.SamplingFrequency)
		self.GPIOPin=MCP3008(self.GPIOPinNumber)

	def GetSignalWindow(self):
		WindowedSignal=np.zeros(self.WindowSize)
		CurrentTimeUs=time.time()
		for i in range(self.WindowSize):
			WindowedSignal[i]=self.GPIOPin.value
			
			"""
			The following method is made with the objective to make 
			sure that the samples are taken at a sampling frequency 
			of 44100 hz The method is not robust yet and i have to 
			come up with some thing  better. 
			"""
			NewTimeUs=time.time()
			if self.SamplePeriod-NewTimeUs+CurrentTimeUs>0:
				time.sleep(self.SamplePeriod-NewTimeUs+CurrentTimeU)
			CurrentTimeUs=NewTimeUs	

		return WindowedSignal
			
