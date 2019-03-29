#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define m 6
#define n 5

int main()
{
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	int i, j;
	float* a;
	float* x;
	float* y;
	a = (float*)malloc(m*n*sizeof(float));
	x = (float*)malloc(n*sizeof(float));
	y = (float*)malloc(m*sizeof(float));

	int ind = 11;
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			a [j*m+i]=( float ) ind ++;
		}
	}

	printf("a:\n");
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%4.0f", a [j*m+i]);
		}
		printf("\n");
	}

	for (i = 0; i < n; i++) x[i] = 1.0f;
	for (i = 0; i < m; i++) y[i] = 0.0f;

	float* d_a;
	float* d_x;
	float* d_y;

	cudaStat = cudaMalloc((void**)&d_a,m*n*sizeof(*a));
	cudaStat = cudaMalloc((void**)&d_x,n*sizeof(*x));
	cudaStat = cudaMalloc((void**)&d_y,m*sizeof(*y));

	stat = cublasCreate(&handle);
	stat = cublasSetMatrix(m,n,sizeof(*a), a, m, d_a, m);
	stat = cublasSetVector(n,sizeof(*x),x,1,d_x,1);
	stat = cublasSetVector(m,sizeof(*y),y,1,d_y,1);

	float al = 1.0f;
	float bet =0.0f;

	stat = cublasSgemv(handle, CUBLAS_OP_N,m,n,&al,d_a,m,d_x,1,&bet,d_y,1);
	stat = cublasGetVector(m,sizeof(*y),d_y,1,y,1);
	printf ( "Sgemv ::\n" );
	for ( j =0; j < m ; j ++){
		printf ( "%5.0f\n",y[j]);
	}

	cudaFree (d_a);
	cudaFree (d_x);
	cudaFree (d_y);
	cublasDestroy(handle);
	free(a);
	free(x);
	free(y);

	return EXIT_SUCCESS;
}
