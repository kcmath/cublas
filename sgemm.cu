#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define m 6
#define n 4
#define k 5
int main ()
{
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	int i,j;
	float* a;
	float* b;
	float* c;

	a = (float*)malloc(m*k*sizeof(float));
	b = (float*)malloc(k*n*sizeof(float));
	c = (float*)malloc(m*n*sizeof(float));

	int ind = 11;
	for (j = 0; j < k; j++) {
		for (i = 0; i < m; i++) {
			a [j*m+i] = (float)ind++;
		}
	}

	printf("a:\n");
	for (i = 0; i < m; i++) {
		for (j = 0; j < k; j++) {
			printf("%4.0f", a[j*m+i]);
		}
		printf("\n");
	}

	ind = 11;
	for (j = 0; j < n; j++) {
		for (i = 0; i < k; i++) {
			b [j*m+i] = (float)ind++;
		}
	}

	printf("b:\n");
	for (i = 0; i < k; i++) {
		for (j = 0; j < n; j++) {
			printf("%4.0f", b[j*m+i]);
		}
		printf("\n");
	}

	ind = 11;
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			c [j*m+i] = (float)ind++;
		}
	}

	printf("c:\n");
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%4.0f", c[j*m+i]);
		}
		printf("\n");
	}

	float* d_a;
	float* d_b;
	float* d_c;


cudaStat = cudaMalloc((void**)&d_a,m*k*sizeof(*a));
cudaStat = cudaMalloc((void**)&d_b,k*n*sizeof(*b));
cudaStat = cudaMalloc((void**)&d_c,m*n*sizeof(*c));

stat = cublasCreate(&handle);
stat = cublasSetMatrix(m,k,sizeof(*a),a,m,d_a,m);
stat = cublasSetMatrix(k,n,sizeof(*b),b,k,d_b,k);
stat = cublasSetMatrix(m,n,sizeof(*c),c,m,d_c,m);

float al=1.0f;
float bet=1.0f;

stat = cublasSgemm (handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,d_a,m,d_b,k,&bet,d_c,m);

stat = cublasGetMatrix(m,n,sizeof(*c),d_c,m,c,m);

printf("c:\n");
for (i = 0; i < m; i++) {
	for (j = 0; j < n; j++) {
		printf("%7.0f", c[j*m+i]);
	}
	printf("\n");
}
	return EXIT_SUCCESS;
}
