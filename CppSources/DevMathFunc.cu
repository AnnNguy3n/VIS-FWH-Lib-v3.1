#pragma once
#include <cuda_runtime.h>


__device__ __forceinline__
int f_arg_min(const float* array, int start, const int end) {
    int min_pos = start;
    while (++start < end) if (array[start] < array[min_pos]) min_pos = start;
    return min_pos;
}


__device__ __forceinline__
void f_fill_value(float* array, int start, const int end, const float value) {
    while (start < end) array[start++] = value;
}


__device__ __forceinline__
void int_fill_value(int* array, int start, const int end, const int value) {
    while (start < end) array[start++] = value;
}


__device__ __forceinline__
void f_copy_array(float* __restrict__ dst, const float* __restrict__ src, int d_x, const int d_y, int s_x) {
    while (d_x < d_y) dst[d_x++] = src[s_x++];
}


__device__ __forceinline__
bool f_is_in(const float* array, int start, const int end, const float value) {
    while (start < end) if (array[start++] == value) return true;
    return false;
}


__device__ __forceinline__
void f_mul_inpl(float* __restrict__ A, const float* __restrict__ B, int arr_len) {
    while (arr_len--) A[arr_len] *= B[arr_len];
}

__device__ __forceinline__
void f_div_inpl(float* __restrict__ A, const float* __restrict__ B, int arr_len) {
    while (arr_len--) A[arr_len] /= B[arr_len];
}

__device__ __forceinline__
void f_add_inpl(float* __restrict__ A, const float* __restrict__ B, int arr_len) {
    while (arr_len--) A[arr_len] += B[arr_len];
}

__device__ __forceinline__
void f_sub_inpl(float* __restrict__ A, const float* __restrict__ B, int arr_len) {
    while (arr_len--) A[arr_len] -= B[arr_len];
}
