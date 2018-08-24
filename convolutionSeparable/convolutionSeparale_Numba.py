import numpy as np
from numba import cuda,float32
from sys import argv
import math

@cuda.jit('void(float32[:],float32[:],float32[:],int32,int32,int32)')
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


@cuda.jit('void(float32[:],float32[:],float32[:],int32,int32,int32)')
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

def convolutionColumnCPU(h_Dst, h_Src, h_Kernel, imageW, imageH, kernelR):
	for y in xrange(0,imageH):
		for x in xrange(0, imageW):
			sum = 0.0
			for k in xrange(-kernelR,kernelR+1):
				d = y + k;
				if (d >= 0 and d < imageH):
					sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k]

				h_Dst[y * imageW + x] = sum

def convolutionRowCPU(h_Dst, h_Src, h_Kernel, imageW, imageH, kernelR):
	for y in xrange(0,imageH):
		for x in xrange(0,imageW):
			sum = 0.0
			for k in xrange(-kernelR,kernelR+1):
				d = x + k;
				if(d >= 0 and d < imageW):
					sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k]

			h_Dst[y * imageW + x] = sum
       
 
imageW = 3072 
imageH = 3072 
iterations = 16 
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

print "Convolution Separable with Numba"

np.random.seed(200)

h_Kernel = np.array(np.random.uniform(0,16,[KERNEL_LENGTH]), dtype=np.float32)
h_Input = np.array(np.random.uniform(0.0,16.0,[imageW*imageH]), dtype=np.float32)
#h_Kernel = np.array(np.random.randint(16,size=(KERNEL_LENGTH)), dtype=np.float32)
#h_Input = np.array(np.random.randint(16,size=(imageW*imageH)), dtype=np.float32)

print "Copying data to GPU"
d_Output = cuda.to_device(np.empty_like(h_Input))
d_Buffer = cuda.to_device(np.empty_like(h_Input))
d_Kernel = cuda.to_device(h_Kernel)
d_Input = cuda.to_device(h_Input)


row_blocks =(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y)
row_threads = (ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y)

col_blocks = (imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y))
col_threads = (COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y)

print "Executing Kernel"

for i in xrange(0,iterations):
	convolutionRowsGPU[row_blocks,row_threads](d_Buffer,d_Input, d_Kernel, imageW, imageH, imageW)
	convolutionColumnsGPU[col_blocks,col_threads](d_Output, d_Buffer, d_Kernel, imageW, imageH, imageW)

h_OutputGPU = d_Output.copy_to_host()

if argv[1]=='1':
	h_Buffer=np.empty_like(h_Input)
	h_OutputCPU=np.empty_like(h_Input)
	print "Executing CPU version"
	convolutionRowCPU(h_Buffer, h_Input, h_Kernel, imageW, imageH, KERNEL_RADIUS);
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Kernel,imageW, imageH, KERNEL_RADIUS);

	print " Comparing the results\n"
	sum = 0.0
	delta = 0.0 
	for i in xrange(0,imageW * imageH):
		delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]) 
		sum   += h_OutputCPU[i] * h_OutputCPU[i] 

	L2norm = math.sqrt(delta / sum) 
	print " ...Relative L2 norm: "+str(L2norm)
	print ("PASSED" if (L2norm < 1e-6) else "FAILED") 

