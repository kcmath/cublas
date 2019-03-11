#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 32

int main()
{
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	
	int j;
	float* x;
	float* y;

	x=(float*)malloc(n*sizeof(*x));
	y=(float*)malloc(n*sizeof(*y));

	for(j=0;j<n;j++)
	{
		x[j]=(float)j;
		y[j]=(float)j;	
	}

	float* d_x;
	float* d_y;
	float* result;

	cudaStat=cudaMalloc((void**)&d_x,n*sizeof(*x));
	cudaStat=cudaMalloc((void**)&d_y,n*sizeof(*y));

	stat = cublasCreate(&handle);
	stat = cublasSetVector(n,sizeof(*x),x,1,d_x,1);
	stat = cublasSetVector(n,sizeof(*y),y,1,d_y,1);

	stat = cublasSdot(handle,n,d_x,1,d_y,1,result);

	printf("dot product x.y\n");
	printf("%f\n",*result);

	cudaFree(d_x);
	cudaFree(d_y);

	free(x);
	free(y);

	return 0;
}

