################################################################################################
#Black Scholes using numba for GPU computing
#>python BlackScholes_Numba.py <interations> <wants to run result check(1 or 0)>
#
#Based on:
#github.com/fernandoc1/Benchmarking-CUDA/
################################################################################################

import numpy as np
from numba import cuda
import math
from sys import argv

@cuda.jit('float32(float32)',device=True,inline=True)
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

@cuda.jit('void(float32[:],float32[:],float32[:],float32[:],float32[:],float32,float32,int32)')
def BlackScholesGPU(d_CallResult,d_PutResult,d_StockPrice,d_OptionStrike,d_OptionYears,RiskFree,Volatility,optN):
	tid = cuda.grid(1)
	THREAD_N = cuda.gridsize(1)

	for i in xrange(tid,optN,THREAD_N):
		sqrtT = math.sqrt(d_OptionYears[i])
		d1 = ( math.log(d_StockPrice[i]/d_OptionStrike[i]) + (RiskFree+0.5*Volatility*Volatility)*d_OptionYears[i])/(Volatility * sqrtT)
		d2 = d1 - Volatility * sqrtT

		CNDD1 = cndGPU(d1)
		CNDD2 = cndGPU(d2)

		expRT = math.exp(-RiskFree * d_OptionYears[i])
		d_CallResult[i] = d_StockPrice[i] * CNDD1 - d_OptionStrike[i] * expRT * CNDD2
		d_PutResult[i] = d_OptionStrike[i] * expRT * (1.0 - CNDD2) - d_StockPrice[i] * (1.0 - CNDD1)

#CPU Functions
def cndCPU(d):
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

def Black_ScholesCPU(h_CallResult,h_PutResult,h_StockPrice,h_OptionStrike,h_OptionYears,RiskFree,Volatility,optN):
	for i in xrange(0,optN):
		sqrtT = math.sqrt(h_OptionYears[i])
		d1 = ( math.log(h_StockPrice[i]/h_OptionStrike[i]) + (RiskFree+0.5*Volatility*Volatility)*h_OptionYears[i])/(Volatility * sqrtT)
		d2 = d1 - Volatility * sqrtT

		CNDD1 = cndCPU(d1)
		CNDD2 = cndCPU(d2)

		expRT = math.exp(-RiskFree * h_OptionYears[i])
		h_CallResult[i] = h_StockPrice[i] * CNDD1 - h_OptionStrike[i] * expRT * CNDD2
		h_PutResult[i] = h_OptionStrike[i] * expRT * (1.0 - CNDD2) - h_StockPrice[i] * (1.0 - CNDD1)


OPT_N = 4000000;
NUM_ITERATIONS = int(argv[1])
#OPT_Z = OPT_N * sizeof(float)
RISKFREE = 0.02
VOLATILITY = 0.30

print "Black Scholes with Numba"

#Initializing arrays
np.random.seed(5347)

h_CallResultCPU = np.zeros(shape=(OPT_N), dtype=np.float32)	#faz vetor
h_PutResultCPU  = np.full((OPT_N),-1.0, dtype=np.float32)
h_StockPrice    = np.array(np.random.uniform(5.0,30.0,[OPT_N]), dtype=np.float32)
h_OptionStrike  = np.array(np.random.uniform(1.0,100.0,[OPT_N]), dtype=np.float32)
h_OptionYears   = np.array(np.random.uniform(0.25,10.0,[OPT_N]), dtype=np.float32)
h_CallResultGPU = np.zeros(OPT_N,dtype=np.float32)
h_PutResultGPU  = np.zeros(OPT_N,dtype=np.float32)

#d_CallResult = np.zeros(OPT_N)
#d_PutResult = np.zeros(OPT_N)

#d_StockPrice = np.zeros(OPT_N)
#d_OptionStrike = np.zeros(OPT_N)
#d_OptionYears = np.zeros(OPT_N)


print "Copying Data to GPU"
d_StockPrice = cuda.to_device(h_StockPrice)
d_OptionStrike = cuda.to_device(h_OptionStrike)
d_OptionYears = cuda.to_device(h_OptionYears)

d_CallResult = cuda.to_device(h_CallResultGPU)
d_PutResult = cuda.to_device(h_PutResultGPU)

print "Executing Black-Scholes GPU Kernel "+str(NUM_ITERATIONS)+" Interations . . ."
for i in xrange(NUM_ITERATIONS):
	BlackScholesGPU[480,128](d_CallResult,d_PutResult,d_StockPrice,d_OptionStrike,d_OptionYears,RISKFREE,VOLATILITY,OPT_N)

h_CallResultGPU = d_CallResult.copy_to_host()
h_PutResultGPU = d_PutResult.copy_to_host()

#Executar CPU se quiser  
if(argv[2]=='1'):
	#Calculating CPU
	print "Calculating CPU"
	Black_ScholesCPU(h_CallResultCPU, h_PutResultCPU, h_StockPrice, h_OptionStrike, h_OptionYears, RISKFREE, VOLATILITY, OPT_N)

	print "Comparing Results"

	sum_delta = 0
	sum_ref = 0
	max_delta = 0

	for i in xrange(OPT_N):
	    ref   = h_CallResultCPU[i]
	    delta = math.fabs(h_CallResultCPU[i] - h_CallResultGPU[i])
	    if(delta > max_delta):
			max_delta = delta

	    sum_delta += delta
	    sum_ref   += math.fabs(ref)
	    
	L1norm = sum_delta / sum_ref
	print "L1 norm: "+ str(L1norm)
	print "Max absolute error: " +str(max_delta)

	print "[BlackScholes] - Test Summary\n"
	print ("PASSED" if (L1norm < 1e-6) else "FAILED")


        