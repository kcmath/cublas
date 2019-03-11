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

    int i;
    float* x;
    float* y;
    
    x = (float*)malloc(n*sizeof(*x));
    y = (float*)malloc(n*sizeof(*y));
    
    for (i=0;i<n;i++)
    {
        x[i] = (float)i;
        y[i] = (float)i;
    }
    
    float* d_x;
    float* d_y;
    float alpha=1.0;

    cudaMalloc((void**)&d_x,n*sizeof(*x));
    cudaMalloc((void**)&d_y,n*sizeof(*y));

    stat = cublasCreate(&handle);
    stat = cublasSetVector(n,sizeof(*x),x,1,d_x,1);
    stat = cublasSetVector(n,sizeof(*y),y,1,d_y,1);
    
    stat = cublasSaxpy(handle,n,&alpha,d_x,1,d_y,1);
    
    stat = cublasGetVector(n,sizeof(float),d_y,1,y,1);
    printf("y = alpha*x + y\n");
    for(i=0;i<n;i++)
    {
        printf("%f\n",y[i]);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
    free(x);
    free(y);
    
	return EXIT_SUCCESS;
}

