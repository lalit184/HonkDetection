class Parameter(object):
	def __init__(self):
		self.NumClasses=2
		self.TotalEpochs=20
		self.SamplingFrequency=44100
		self.SubSamplingRate=1
		self.SignalLength=441000
		self.WindowTime=0.05
		self.WindowStep=0.05
		self.WindowSize=int(self.WindowTime*self.SamplingFrequency/self.SubSamplingRate)
		self.SubSamplingRate=1
		self.NumCep=100
		self.batch_size=1#int(self.SignalLength/(self.WindowTime*self.SamplingFrequency))
		

