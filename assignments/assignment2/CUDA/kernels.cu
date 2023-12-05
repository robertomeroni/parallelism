#include <math.h>
#include <float.h>
#include <cuda.h>

#define maxBlockDim 1024

__global__ void gpu_Heat (double *h, double *g, int N) {

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	// kernel computation
	if (i>0 && i< N - 1 && j>0 && j< N - 1)
	{
		g[i*N+j]= 0.25 * (h[ i*N + (j-1) ]+  // left
					     	h[ i*N + (j+1) ]+  // right
				         h[ (i-1)*N + j ]+  // top
				            h[ (i+1)*N + j ]); // bottom
	}
}

__global__ void gpu_Residual (double *h, double *g, int N, double *diff)
{	
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
  int index = (N-2) * (i-1) + j-1;
	// kernel computation
	if (i>0 && i< N - 1 && j>0 && j< N - 1)
	{
		g[i*N+j]= 0.25 * (h[ i*N + (j-1) ]+  // left
					     	h[ i*N + (j+1) ]+  // right
				         h[ (i-1)*N + j ]+  // top
				            h[ (i+1)*N + j ]); // bottom
	
	  diff[index] = g[i*N + j] - h[i*N + j];
	  diff[index] = diff[index] * diff[index];
	}
}

__global__ void gpu_Reduction(double *g_idata, double *g_odata, int N) {
  __shared__ double sdata[maxBlockDim];
  unsigned int s;

  // Cada thread realiza la suma parcial de los datos que le
  // corresponden y la deja en la memoria compartida
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int gridSize = blockDim.x*2*gridDim.x;
  sdata[tid] = 0;
  while (i < N) {
    sdata[tid] += g_idata[i] + g_idata[i+blockDim.x];
    i += gridSize;
  }
  __syncthreads();

  // Hacemos la reduccion en la memoria compartida
  for (s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // desenrrollamos el ultimo warp activo
  if (tid < 32) {
    volatile double *smem = sdata;

    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
  }


  // El thread 0 escribe el resultado de este bloque en la memoria global
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

