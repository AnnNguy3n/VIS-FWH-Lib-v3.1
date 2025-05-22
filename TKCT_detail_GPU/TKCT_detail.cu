#pragma once
#include <cfloat>
#include "../CppSources/Generator/HomogeneousPolynomial/CUDA/HP_Kernel.cu"
#include "../CppSources/DevMathFunc.cu"
#include <iostream>
#include <string>
#include "../CppSources/Utils/WorkWithFile.cpp"
#include <fstream>
using namespace std;


__constant__ int NUM_SYMBOL_UNIQUE;
__constant__ int INDEX_LEN;
__constant__ int ARRAY_LEN;

__constant__ int INDEX[100];

constexpr int CHUNK_SIZE = 128;
constexpr int NUM_COLS = 30;
__constant__ float OPERAND[NUM_COLS][CHUNK_SIZE];


__device__ __forceinline__
void _calculate_formula(
    const int*   __restrict__ formula,
    const int F_len,
    const int eval_method,
          float* __restrict__ result,
    const int offset
) {
    //
    extern __shared__ float smem[];
    float* temp_0 = smem + 2 * CHUNK_SIZE * threadIdx.x;
    float* temp_1 = temp_0 + CHUNK_SIZE;

    //
    int temp_op;
    int deg = 0;
    f_fill_value(temp_0, 0, CHUNK_SIZE, 0.0f);

    int next_oprt = formula[0];
    for (int i = 1; i < F_len; i += 2) {
        int oprt = next_oprt;
        int oprand = formula[i];
        if (i + 1 != F_len) next_oprt = formula[i+1];

        if (oprt < 2) {
            deg = 1;
            temp_op = oprt;
            f_copy_array(temp_1, OPERAND[oprand], 0, CHUNK_SIZE, 0);
        } else {
            if (oprt == 2) {
                ++deg;
                f_mul_inpl(temp_1, OPERAND[oprand], CHUNK_SIZE);
            } else {
                --deg;
                f_div_inpl(temp_1, OPERAND[oprand], CHUNK_SIZE);
            }
        }

        if (i + 1 == F_len || next_oprt < 2) {
            if (eval_method == 1) {
                for (int j = 0; j < CHUNK_SIZE; ++j)
                    temp_1[j] = safe_root(temp_1[j], deg);
            }

            if (temp_op) f_sub_inpl(temp_0, temp_1, CHUNK_SIZE);
            else f_add_inpl(temp_0, temp_1, CHUNK_SIZE);
        }
    }

    for (int j = 0; j < CHUNK_SIZE; ++j) {
        if (isnan(temp_0[j]) || isinf(temp_0[j]))
            temp_0[j] = -FLT_MAX;
    }

    int end_offset = min(offset + CHUNK_SIZE, ARRAY_LEN);
    for (int j = offset; j < end_offset; ++j)
        result[j] = temp_0[j-offset];
}


__global__ void calculate_formula(
    const int*   __restrict__ N_formula,
          float* __restrict__ N_result,
    const int*   __restrict__ cp_f_start,
    const int*   __restrict__ cp_f_len,
    const int num_array,
    const int eval_method,
    const int offset
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_array) return;

    const int start = cp_f_start[tid];
    const int* formula = &N_formula[start];
    const int F_len = cp_f_len[tid];
    float* result = N_result + tid * ARRAY_LEN;

    _calculate_formula(formula, F_len, eval_method, result, offset);
}


int main(int argc, char* argv[]) {
    string folder_data = argv[1];
    int eval_method = stoi(argv[2]);
    int center_method_num = stoi(argv[3]);

    int *_INDEX, *_SYMBOL, *_BOOL_ARG;
    float *_PROFIT, *_OPERAND;
    int index_len, rows, cols;
    read_binary_file_1d(_INDEX, index_len, folder_data + "/InputData/INDEX.bin");
    read_binary_file_1d(_SYMBOL, rows, folder_data + "/InputData/SYMBOL.bin");
    read_binary_file_1d(_BOOL_ARG, rows, folder_data + "/InputData/BOOL_ARG.bin");
    read_binary_file_1d(_PROFIT, rows, folder_data + "/InputData/PROFIT.bin");
    read_binary_file_2d(_OPERAND, cols, rows, folder_data + "/InputData/OPERAND.bin");

    int max_ = 0;
    for (int i = 0; i < rows; ++i) {
        if (_SYMBOL[i] > max_) max_ = _SYMBOL[i];
    }
    ++max_;
    cudaMemcpyToSymbol(INDEX_LEN, &index_len, 4);
    cudaMemcpyToSymbol(INDEX, _INDEX, 4 * index_len);
    cudaMemcpyToSymbol(ARRAY_LEN, &rows, 4);
    cudaMemcpyToSymbol(NUM_SYMBOL_UNIQUE, &max_, 4);

    //
    int *SYMBOL, *BOOL_ARG;
    float *PROFIT;
    cudaMalloc((void**)&SYMBOL, 4*rows);
    cudaMalloc((void**)&BOOL_ARG, 4*rows);
    cudaMalloc((void**)&PROFIT, 4*rows);

    cudaMemcpy(SYMBOL, _SYMBOL, 4*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(BOOL_ARG, _BOOL_ARG, 4*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(PROFIT, _PROFIT, 4*rows, cudaMemcpyHostToDevice);

    //
    int *_N_formula, *_cp_f_start, *_cp_f_len;
    int num_array, NF_arr_len;
    int *N_formula, *cp_f_start, *cp_f_len;

    read_binary_file_1d(_N_formula, NF_arr_len, folder_data + "/InputData/N_formula.bin");
    read_binary_file_1d(_cp_f_start, num_array, folder_data + "/InputData/cp_f_start.bin");
    read_binary_file_1d(_cp_f_len, num_array, folder_data + "/InputData/cp_f_len.bin");

    cudaMalloc((void**)&N_formula, 4*NF_arr_len);
    cudaMalloc((void**)&cp_f_start, 4*num_array);
    cudaMalloc((void**)&cp_f_len, 4*num_array);

    cudaMemcpy(N_formula, _N_formula, 4*NF_arr_len, cudaMemcpyHostToDevice);
    cudaMemcpy(cp_f_start, _cp_f_start, 4*num_array, cudaMemcpyHostToDevice);
    cudaMemcpy(cp_f_len, _cp_f_len, 4*num_array, cudaMemcpyHostToDevice);

    //
    float *host_N_result, *dev_N_result;
    host_N_result = new float[num_array*rows];
    cudaMalloc((void**)&dev_N_result, num_array*rows*4);

    dim3 threads(32);
    int num_block = (num_array + threads.x - 1) / threads.x;

    for (int offset = 0; offset < rows; offset += CHUNK_SIZE) {
        for (int i = 0; i < NUM_COLS; ++i)
            cudaMemcpyToSymbol(OPERAND[i], _OPERAND + i * rows + offset, 4 * min(CHUNK_SIZE, rows - offset), 0);

        calculate_formula<<<num_block, threads, threads.x * 2 * CHUNK_SIZE * 4>>>(
            N_formula, dev_N_result, cp_f_start, cp_f_len, num_array, eval_method, offset
        );
    }

    cudaMemcpy(host_N_result, dev_N_result, num_array*rows*4, cudaMemcpyDeviceToHost);
}