###############Requirements#######################
# - Install Anaconda
# - conda install numba
# - conda install cudatoolkit
# If it still complains about drivers install the
# official release of the cudatoolkit from nvidia
#(r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu)
##################################################

from pydf import *
import nodes as Nodes
from numba import cuda
import numpy as np

#Receives inputs, sends data, calls kernel and returns value
#<function Name> <total number of inputs> <cuda.jit header> <num Blocks> <num threads> <Number of the input you want back> <Number of times the kernel will execute>
class CudaNode(Node):
		def __init__(self,f,inputn,header, numBlocks, numThreads, toHost=[], ntimes=1):
			Node.__init__(self,f,inputn)
			
			self.inputn = inputn
			self.device_array = []
			self.host_array = []
			self.header = header
			self.numBlocks = numBlocks
			self.numThreads = numThreads
			self.toHost = toHost #default is empty		
			self.nTimes = ntimes
				
		def decorate_cudafunc(self):
			cuda_function = cuda.jit(self.header)(self.f)
			#cuda_function = cuda.jit(device=True)(self.f)
			return cuda_function
			
		def send_toDevice(self,args):
			#for i in range(self.inputn):
			for arg in args:
				if(np.isscalar(arg.val)): #scalar numbers are not to be sent(explicity)
					self.device_array += [arg.val]
				else:
					self.device_array += [cuda.to_device(arg.val)]
				
		def get_fromDevice(self):
			if not self.toHost: #if empty, send it all
				for i in range(self.inputn):
					self.host_array += [self.device_array[i].copy_to_host()]	
			elif len(self.toHost)==1: #if its just one return, no need to use an array
				self.host_array = (self.device_array[self.toHost[0]].copy_to_host()).tolist() 
				#print self.host_array
			else:
				for i in range(len(self.toHost)):
					self.host_array += [self.device_array[self.toHost[i]].copy_to_host()]
		
		def select_d(self,workerid):
			num_gpus = len(list(cuda.gpus))
			print "Device selected:", workerid % num_gpus
			cuda.select_device(workerid % num_gpus)
		
		def run(self, args, workerid, operq):
			#print cuda.gpus #Get all devices
			#print cuda.cudadrv.devices._DeviceList.current #returns active device
			
			self.select_d(workerid) # selects device
			
			self.f = self.decorate_cudafunc() #decorate here to have no problems with CudaContext
			
			#sending / doing function / retrieving
			self.send_toDevice(args)
			print "Executing GPU Kernel"
			for i in xrange(self.nTimes):
				self.f[self.numBlocks,self.numThreads]( *self.device_array ) #unpacking			
			self.get_fromDevice()
			cuda.close() #close context to free variables
			
			opers = self.create_oper(self.host_array, workerid, operq)
			self.sendops(opers, operq)

#Receives a value from another node and sends it to the GPU
class cuSend(Node):
	def __init__(self,device=0):
		self.device = device
		self.inport = [[]]
		self.dsts = []
		self.affinity = [0] #The Affinity MUST not be the same for the cuKernel and cuGet
	def f(self,value):
		d_arr = cuda.to_device(value)
		return d_arr
		
	def run(self, args, workerid, operq):
		
		cuda.select_device(self.device)
		
		d_arr = self.f(args[0].val)
		
		result = d_arr.get_ipc_handle()
		
		opers = self.create_oper(result, workerid, operq)
		self.sendops(opers, operq)

class cuKernel(CudaNode):
	def __init__(self,f,inputn,header, numBlocks, numThreads, toHost=[], nTimes=1):
		
		CudaNode.__init__(self,f,inputn,header,numBlocks,numThreads, toHost, nTimes)
		
		self.kernel_inputs = []
		self.affinity = [1] # Default is different from the Default of cuSend()
		
	def get_data(self,args):
		for arg in args:
			value = arg.val
			if(np.isscalar(value)): #scalar numbers are not to be sent(explicity)
				self.kernel_inputs += [value]
			else:
				self.kernel_inputs += [value.open()] #If the Workerid is the same as the one that did the copy you gonna receive a exception
	
	def close(self,args):
		for arg in args:
			value = arg.val
			if np.isscalar(value)==False: 
				value.close()
			
	def run(self, args, workerid, operq):
	
		self.get_data(args)
			
		self.f = self.decorate_cudafunc() #decorate here to have no problems with CudaContext
		
		print " Executing Kernel "
		for i in xrange(self.nTimes):
			self.f[self.numBlocks,self.numThreads]( *self.kernel_inputs )
		
		self.close(args)
		
		if len(self.toHost)==1:
			result = args[self.toHost[0]].val
		else:
			result = [ args[i].val for i in self.toHost ]

		opers = self.create_oper(result, workerid, operq)
		self.sendops(opers, operq)

class cuGet(Node):
	def __init__(self):
		self.inport = [[]]
		self.dsts = []
		self.affinity = [1] #Different affiny from cuSend()
		
	def f(self,arg):

		d_arr = arg.open()

		value = [d_arr.copy_to_host()]
		arg.close()

		return value
		
	def run(self, args, workerid, operq):

		result=[]
		if(isinstance(args[0].val,list)):
			for arg in args[0].val:
				result += [self.f(arg)]
		else:
			result = [self.f(args[0].val)]
		opers = self.create_oper(result, workerid, operq)
		self.sendops(opers, operq)
