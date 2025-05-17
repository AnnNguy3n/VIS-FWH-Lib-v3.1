#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cfloat>


__global__ void cuda_set_array_value(float* array, int length, float value){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) array[index] = value;
};


__global__ void copy_from_operands(float * __restrict__ dest, float * __restrict__ operands, int * __restrict__ arrCpy, int length, int numCpy){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numCpy*length){
        int i = index / length;
        int j = index % length;
        dest[index] = operands[arrCpy[i]*length + j];
    }
};


__global__ void update_temp_weight(float * __restrict__ temp_weight_new, float * __restrict__ temp_weight_old, float * __restrict__ operands, int * __restrict__ arrOpr, int length, int numOpr, bool isMul){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        int i = index / length;
        int j = index % length;
        if (isMul)
            temp_weight_new[index] = temp_weight_old[j] * operands[arrOpr[i]*length + j];
        else
            temp_weight_new[index] = temp_weight_old[j] / operands[arrOpr[i]*length + j];
    }
};


__device__ float safe_root(float x, int deg) {
    if (x < 0.0) {
        if (deg % 2 == 0) return 0.0;
        else return -powf(-x, 1.0 / deg);
    }
    return powf(x, 1.0 / deg);
}


__global__ void update_last_weight(float * __restrict__ last_weight, float * __restrict__ curr_weight, float * __restrict__ temp_weight, int length, int numOpr, bool isAdd, int fml_deg, int eval_method){
    // eval_method: 0 - classic, 1 - root
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        int j = index % length;
        float val;
        if (eval_method == 0) val = temp_weight[index];
        else val = safe_root(temp_weight[index], fml_deg);
        if (isAdd) last_weight[index] = curr_weight[j] + val;
        else last_weight[index] = curr_weight[j] - val;
    }
};


__global__ void update_last_weight_through_operands(float * __restrict__ last_weight, float * __restrict__ curr_weight, float * __restrict__ operands, int * __restrict__ arrOpr, int length, int numOpr, bool isAdd){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        int i = index / length;
        int j = index % length;
        if (isAdd) last_weight[index] = curr_weight[j] + operands[arrOpr[i]*length + j];
        else last_weight[index] = curr_weight[j] - operands[arrOpr[i]*length + j];
    }
};


__global__ void replace_nan_and_inf(float* array, int length, int numOpr){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        if (isnan(array[index]) || isinf(array[index]))
            array[index] = -FLT_MAX;
    }
};
