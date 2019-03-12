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
    float* result;
    
    x = (float*)malloc(n*sizeof(*x));
    
    for (i=0;i<n;i++)
    {
        x[i] = (float)i;
    }
    
    float* d_x;

    cudaMalloc((void**)&d_x,n*sizeof(*x));

    stat = cublasCreate(&handle);
    stat = cublasSetVector(n,sizeof(*x),x,1,d_x,1);
    
    stat = cublasSnrm2(handle,n ,d_x, 1,result);
    

    printf("Euclidean norm of x:%f",*result);
    printf("\n");

    cudaFree(d_x);
    cublasDestroy(handle);
    free(x);
    
	return EXIT_SUCCESS;
}

