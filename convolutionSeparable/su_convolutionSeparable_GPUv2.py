import numpy as np
from numba import cuda,float32
from sys import argv,path
import math
path.append("./../Sucuri")
from pyDF import *

#@cuda.jit('void(float32[:],float32[:],float32[:],int32,int32,int32)')
def convolutionColumnsGPU(d_Dst, d_Src, c_Kernel, imageW, imageH, pitch):
	COLUMNS_BLOCKDIM_X = 16
	COLUMNS_BLOCKDIM_Y = 8
	COLUMNS_RESULT_STEPS = 8
	COLUMNS_HALO_STEPS = 1
	KERNEL_RADIUS = 8

	#cuda.const.array_like(c_Kernel)
	#s_Data = cuda.shared.array(shape=(COLUMNS_BLOCKDIM_X,(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1), dtype=float32)
	s_Data = cuda.shared.array(shape=(16,81), dtype=float32)
			
	#Offset to the upper halo edge
	baseX = cuda.blockIdx.x * COLUMNS_BLOCKDIM_X + cuda.threadIdx.x 
	baseY = (cuda.blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + cuda.threadIdx.y 
	#d_Src += baseY * pitch + baseX 
	#d_Dst += baseY * pitch + baseX 
	desvio = baseY * pitch + baseX 

	#Main data
	for i in xrange(COLUMNS_HALO_STEPS,COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS):
		s_Data[cuda.threadIdx.x][cuda.threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[desvio + (i * COLUMNS_BLOCKDIM_Y * pitch)] 

	#Upper halo
	for i in xrange(COLUMNS_HALO_STEPS):
		s_Data[cuda.threadIdx.x][cuda.threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =  d_Src[desvio + (i * COLUMNS_BLOCKDIM_Y * pitch)] if (baseY >= -i * COLUMNS_BLOCKDIM_Y) else 0 

	#Lower halo
	for i in xrange(COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS,COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS):
		s_Data[cuda.threadIdx.x][cuda.threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= d_Src[desvio + (i * COLUMNS_BLOCKDIM_Y * pitch)] if (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) else 0 

	#Compute and store results
	cuda.syncthreads() 
	for i in xrange(COLUMNS_HALO_STEPS,COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS):
		sum = 0.0
		for j in xrange(-KERNEL_RADIUS,KERNEL_RADIUS+1):
			sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[cuda.threadIdx.x][cuda.threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j] 

		d_Dst[desvio+(i * COLUMNS_BLOCKDIM_Y * pitch)] = sum 


#@cuda.jit('void(float32[:],float32[:],float32[:],int32,int32,int32)')
def convolutionRowsGPU(d_Dst, d_Src, c_Kernel, imageW, imageH, pitch):
	ROWS_BLOCKDIM_X = 16
	ROWS_BLOCKDIM_Y = 4
	ROWS_RESULT_STEPS = 8
	ROWS_HALO_STEPS = 1
	KERNEL_RADIUS = 8

	#s_Data = cuda.shared.array(shape=(ROWS_BLOCKDIM_Y,(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X),dtype=float32)
	s_Data = cuda.shared.array(shape=(4,160),dtype=float32)
	#Offset to the left halo edge
	baseX = (cuda.blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + cuda.threadIdx.x 
	baseY = cuda.blockIdx.y * ROWS_BLOCKDIM_Y + cuda.threadIdx.y 

	#d_Src += baseY * pitch + baseX 
	#d_Dst += baseY * pitch + baseX
	desvio = baseY * pitch + baseX

	#Load main data
	for i in xrange(ROWS_HALO_STEPS,ROWS_HALO_STEPS + ROWS_RESULT_STEPS):
		s_Data[cuda.threadIdx.y][cuda.threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[desvio + (i * ROWS_BLOCKDIM_X)] 

	#Load left halo
	for i in xrange(ROWS_HALO_STEPS):
		s_Data[cuda.threadIdx.y][cuda.threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[desvio + (i * ROWS_BLOCKDIM_X)] if (baseX >= -i * ROWS_BLOCKDIM_X ) else 0 

	#Load right halo
	for i in xrange(ROWS_HALO_STEPS + ROWS_RESULT_STEPS,ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS):
		s_Data[cuda.threadIdx.y][cuda.threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[desvio  + (i * ROWS_BLOCKDIM_X)] if (imageW - baseX > i * ROWS_BLOCKDIM_X) else 0 

	#Compute and store results
	cuda.syncthreads() 
	for i in xrange(ROWS_HALO_STEPS,ROWS_HALO_STEPS + ROWS_RESULT_STEPS):
		sum = 0.0
		for j in xrange(-KERNEL_RADIUS,KERNEL_RADIUS+1):
			sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[cuda.threadIdx.y][cuda.threadIdx.x + i * ROWS_BLOCKDIM_X + j] 

		d_Dst[desvio + (i * ROWS_BLOCKDIM_X)] = sum 


def printer(args):
	print "Done"#args

print "Convolution Separable with Sucuri and GPU"

NUM_ITERATIONS = int(sys.argv[1])
nworkers = int(sys.argv[2])

imageW = 3072
imageH = 3072
#Defines
KERNEL_RADIUS = 8
KERNEL_LENGTH = (2*KERNEL_RADIUS+1)
COLUMNS_BLOCKDIM_X = 16
COLUMNS_BLOCKDIM_Y = 8
COLUMNS_RESULT_STEPS = 8
COLUMNS_HALO_STEPS = 1
ROWS_BLOCKDIM_X = 16
ROWS_BLOCKDIM_Y = 4
ROWS_RESULT_STEPS = 8
ROWS_HALO_STEPS = 1

row_blocks =(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y)
row_threads = (ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y)

col_blocks = (imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y))
col_threads = (COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y)

np.random.seed(200)

graph = DFGraph()
sched = Scheduler(graph, nworkers, mpi_enabled = False)

#Nodes
imW = Feeder(imageW)
imW2 = Feeder(imageW)
imH = Feeder(imageH)
d_Kernel = Feeder(np.array(np.random.uniform(0.0,16.0,[KERNEL_LENGTH]), dtype=np.float32).tolist())
d_Input = Feeder(np.array(np.random.uniform(0.0,16.0,[imageW*imageH]), dtype=np.float32).tolist())
d_Output = Feeder(np.zeros(imageW*imageH,dtype=np.float32).tolist())


S1=cuSend()
S2=cuSend()
S3=cuSend()

GPU1 = cuKernel(convolutionRowsGPU,6,
			  'void(float32[:],float32[:],float32[:],int32,int32,int32)', 
			  row_blocks,row_threads, [0])#, NUM_ITERATIONS)

GPU2 = cuKernel(convolutionColumnsGPU,6,
			  'void(float32[:],float32[:],float32[:],int32,int32,int32)', 
			  col_blocks,col_threads, [0])#, NUM_ITERATIONS)

G1=cuGet()
P = Node(printer,1)

#Adding to graph
graph.add(imW)
graph.add(imW2)
graph.add(imH)
graph.add(d_Kernel)
graph.add(d_Input)
graph.add(d_Output)
graph.add(S1)
graph.add(S2)
graph.add(S3)
graph.add(GPU1)
graph.add(GPU2)
graph.add(G1)
graph.add(P)

#convolutionRowsGPU[row_blocks,row_threads](d_Buffer,d_Input, d_Kernel, imageW, imageH, imageW)
#convolutionColumnsGPU[col_blocks,col_threads](d_Output, d_Buffer, d_Kernel, imageW, imageH, imageW)
#Edges to GPU1 (Rows Kernel)
d_Output.add_edge(S1,0)
d_Input.add_edge(S2,0)
d_Kernel.add_edge(S3,0)

S1.add_edge(GPU1,0)
S2.add_edge(GPU1,1)
S3.add_edge(GPU1,2)
imW.add_edge(GPU1,3)
imH.add_edge(GPU1,4)
imW2.add_edge(GPU1,5)

#Edges to GPU2 (Columns Kernel)
S1.add_edge(GPU2,0)
GPU1.add_edge(GPU2,1)
S3.add_edge(GPU2,2)
imW.add_edge(GPU2,3)
imH.add_edge(GPU2,4)
imW2.add_edge(GPU2,5)

GPU2.add_edge(G1,0)
G1.add_edge(P,0)

sched.start()

