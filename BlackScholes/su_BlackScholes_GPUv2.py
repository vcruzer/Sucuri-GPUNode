################################################
#Black Scholes using Sucuri with GPUNode (Numba)
################################################

import numpy as np
from numba import cuda
import math
import sys
sys.path.append("./../Sucuri")
from pyDF import *

#@cuda.jit('void(float32[:],float32[:],float32[:],float32[:],float32[:],float32,float32,int32)')
def BlackScholesGPU(d_CallResult,d_PutResult,d_StockPrice,d_OptionStrike,d_OptionYears,RiskFree,Volatility,optN):
	tid = cuda.grid(1)
	THREAD_N = cuda.gridsize(1)

	def cndGPU(d):
		A1 = 0.31938153
		A2 = -0.356563782
		A3 = 1.781477937
		A4 = -1.821255978
		A5 = 1.330274429
		RSQRT2PI = 0.39894228040143267793994605993438

		K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
		cnd = RSQRT2PI * math.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))

		if(d>0):
			cnd=1.0 - cnd

		return cnd;

	for i in xrange(tid,optN,THREAD_N):
		sqrtT = math.sqrt(d_OptionYears[i])
		d1 = ( math.log(d_StockPrice[i]/d_OptionStrike[i]) + (RiskFree+0.5*Volatility*Volatility)*d_OptionYears[i])/(Volatility * sqrtT)
		d2 = d1 - Volatility * sqrtT

		CNDD1 = cndGPU(d1)
		CNDD2 = cndGPU(d2)

		expRT = math.exp(-RiskFree * d_OptionYears[i])
		d_CallResult[i] = d_StockPrice[i] * CNDD1 - d_OptionStrike[i] * expRT * CNDD2
		d_PutResult[i] = d_OptionStrike[i] * expRT * (1.0 - CNDD2) - d_StockPrice[i] * (1.0 - CNDD1)

def printer(args):
	for arg in args:
		print args

print "Black Scholes with Sucuri and Numba"

NUM_ITERATIONS = int(sys.argv[1])
nworkers = int(sys.argv[2])
#OPT_Z = OPT_N * sizeof(float)
graph = DFGraph()
sched = Scheduler(graph, nworkers, mpi_enabled = False)

#Initializing arrays
np.random.seed(5347)
OPT_N = 4000000

OPT = Feeder(OPT_N)
RISKFREE = Feeder(0.02)
VOLATILITY = Feeder(0.30)

#TODO Fix numpy issue (?)

StockPrice    = Feeder(np.array(np.random.uniform(5.0,30.0,[OPT_N]), dtype=np.float32).tolist())
OptionStrike  = Feeder(np.array(np.random.uniform(1.0,100.0,[OPT_N]), dtype=np.float32).tolist())
OptionYears   = Feeder(np.array(np.random.uniform(0.25,10.0,[OPT_N]), dtype=np.float32).tolist())
CallResultGPU = Feeder(np.zeros(OPT_N,dtype=np.float32).tolist())
PutResultGPU  = Feeder(np.zeros(OPT_N,dtype=np.float32).tolist())

S1 = cuSend()
S2 = cuSend()
S3 = cuSend()
S4 = cuSend()
S5 = cuSend()



#TODO enable multiple runs of the same Kernel
GPU = cuKernel(BlackScholesGPU,8,
			  'void(float32[:],float32[:],float32[:],float32[:],float32[:],float32,float32,int32)', 
			  480,128, [0,1], NUM_ITERATIONS)

G1 = cuGet()

P = Node(printer,1)

#Adding to the graph
graph.add(OPT)
graph.add(RISKFREE)
graph.add(VOLATILITY)
graph.add(CallResultGPU)
graph.add(PutResultGPU)
graph.add(StockPrice)
graph.add(OptionStrike)
graph.add(OptionYears)
graph.add(S1)
graph.add(S2)
graph.add(S3)
graph.add(S4)
graph.add(S5)
graph.add(G1)
graph.add(GPU)
graph.add(P)

#Adding edges
CallResultGPU.add_edge(S1,0) #GPU node returns
PutResultGPU.add_edge(S2,0) #GPU node returns
StockPrice.add_edge(S3,0)
OptionStrike.add_edge(S4,0)
OptionYears.add_edge(S5,0)
S1.add_edge(GPU,0)
S2.add_edge(GPU,1)
S3.add_edge(GPU,2)
S4.add_edge(GPU,3)
S5.add_edge(GPU,4)
RISKFREE.add_edge(GPU,5)
VOLATILITY.add_edge(GPU,6)
OPT.add_edge(GPU,7)

GPU.add_edge(G1,0)

G1.add_edge(P,0)

#BlackScholesGPU[480,128](d_CallResult,d_PutResult,d_StockPrice,d_OptionStrike,d_OptionYears,RISKFREE,VOLATILITY,OPT_N)

sched.start()


        