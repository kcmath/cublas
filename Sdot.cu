#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include "cublas_v2.h"
#define n 9999999

int main()
{
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
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

	//device
	float* d_x;
	float* d_y;
	float* result;
	float* d_result;

	cudaStat=cudaMalloc((void**)&d_x,n*sizeof(*x));
	cudaStat=cudaMalloc((void**)&d_y,n*sizeof(*y));
	cudaStat=cudaMalloc((void**)&d_result,sizeof(*d_result));
	stat = cublasCreate(&handle);
	stat = cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);
	stat = cublasSetVector(n,sizeof(*x),x,1,d_x,1);
	stat = cublasSetVector(n,sizeof(*y),y,1,d_y,1);

	cudaEventRecord(start);
	stat = cublasSdot(handle,n,d_x,1,d_y,1,d_result);
	cudaEventRecord(stop);
	cublasGetVector(1,sizeof(float),d_result,1,result,1);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("dot product x.y\n");
	printf("%f\n",*result);
	printf("GPU time: %f sec", milliseconds);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);

	return 0;
}

