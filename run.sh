#!/bin/bash

nvcc sdot.cu -o sdot -lcublas
nvcc saxpy.cu -o saxpy -lcublas
nvcc sgemv.cu -o sgemv -lcublas
