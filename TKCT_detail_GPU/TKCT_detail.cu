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


__device__ __forceinline__
void _StreakInvest(
    const float* __restrict__ weight,
          int*   __restrict__ arr_invest,
    const int*   __restrict__ symbol,
    const int*   __restrict__ sufficient_liquidity,
    const float threshold,
    const int   method_num
) {
    // Trích các mảng tạm từ shared memory
    extern __shared__ char smem[];

    uint8_t* symbol_streak_base = reinterpret_cast<uint8_t*>(smem);
    uint8_t* symbol_streak = symbol_streak_base + threadIdx.x * NUM_SYMBOL_UNIQUE;

    // Khởi tạo
    int market_streak = 0;
    for (int i = 0; i < NUM_SYMBOL_UNIQUE; ++i) symbol_streak[i] = 0;
    int_fill_value(arr_invest, 0, ARRAY_LEN, false);

    // Duyệt ngược qua từng chu kỳ
    for (int cycle_idx = INDEX_LEN - 2; cycle_idx >= 0; --cycle_idx) {
        int start = INDEX[cycle_idx];
        int end   = INDEX[cycle_idx + 1];

        //
        int N = INDEX_LEN - 1 - cycle_idx;
        bool any_pass_threshold = false;

        // Duyệt qua tất cả công ty trong chu kỳ hiện tại
        for (int i = start; i < end; ++i) {
            int sym = symbol[i];
            if (weight[i] <= threshold) {
                symbol_streak[sym] = 0;
                continue;
            }

            any_pass_threshold = true;
            int sym_streak = ++symbol_streak[sym];

            if (!sufficient_liquidity[i]) continue;

            //
            if (N >= method_num) {
                if (sym_streak > min(market_streak, method_num - 1)) {
                    arr_invest[i] = true;
                }
            }
        }

        // Cập nhật market streak
        market_streak = any_pass_threshold ? market_streak + 1 : 0;
    }
}


__global__ void StreakInvest(
    const float* __restrict__ N_weight,
          int*   __restrict__ N_arr_invest,
    const int*   __restrict__ symbol,
    const int*   __restrict__ sufficient_liquidity,
    const float* __restrict__ N_threshold,
    const int num_array,
    const int method_num
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_array) return;

    _StreakInvest(
        N_weight + tid * ARRAY_LEN,
        N_arr_invest + tid * ARRAY_LEN,
        symbol, sufficient_liquidity,
        N_threshold[tid], method_num
    );
}


__device__ __forceinline__
void _calculate_formula(
    const float* __restrict__ operand,
          float* __restrict__ temp_0,
          float* __restrict__ temp_1,
    const int*   __restrict__ formula,
    const int F_len,
    const int eval_method
) {
    int temp_op;
    int deg = 0;
    f_fill_value(temp_0, 0, ARRAY_LEN, 0.0f);

    for (int i = 1; i < F_len; i += 2) {
        int oprt = formula[i-1];
        int oprand = formula[i];
        int next_oprt = formula[i+1];

        if (oprt < 2) {
            deg = 1;
            temp_op = oprt;
            f_copy_array(temp_1, operand, 0, ARRAY_LEN, oprand * ARRAY_LEN);
        } else {
            if (oprt == 2) {
                ++deg;
                f_mul_inpl(temp_1, operand + oprand * ARRAY_LEN, ARRAY_LEN);
            } else {
                --deg;
                f_div_inpl(temp_1, operand + oprand * ARRAY_LEN, ARRAY_LEN);
            }
        }

        if (i + 1 == F_len || next_oprt < 2) {
            if (eval_method == 1) {
                for (int j = 0; j < ARRAY_LEN; ++j)
                    temp_1[j] = safe_root(temp_1[j], deg);
            }

            if (temp_op)
                f_sub_inpl(temp_0, temp_1, ARRAY_LEN);
            else
                f_add_inpl(temp_0, temp_1, ARRAY_LEN);
        }
    }

    for (int j = 0; j < ARRAY_LEN; ++j) {
        if (isnan(temp_0[j]) || isinf(temp_0[j]))
            temp_0[j] = -FLT_MAX;
    }
}


__global__ void calculate_formula(
    const float* __restrict__ operand,
          float* __restrict__ N_temp_0,
          float* __restrict__ N_temp_1,
    const int*   __restrict__ N_formula,
    const int*   __restrict__ cp_f_start,
    const int*   __restrict__ cp_f_len,
    const int num_array,
    const int eval_method
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_array) return;

    float* temp_0 = N_temp_0 + tid * ARRAY_LEN;
    float* temp_1 = N_temp_1 + tid * ARRAY_LEN;
    const int start = cp_f_start[tid];
    const int* formula = &N_formula[start];
    const int F_len = cp_f_len[tid];
    _calculate_formula(
        operand, temp_0, temp_1, formula, F_len, eval_method
    );
}


int main(int argc, char* argv[]) {
    string folder_data = argv[1];
    int eval_method = stoi(argv[2]);
    int center_method_num = stoi(argv[3]);

    int *_INDEX, *_SYMBOL, *_BOOL_ARG;
    float *_PROFIT, *_OPERAND;

    int *SYMBOL, *BOOL_ARG, index_len, rows, cols;
    float *PROFIT, *OPERAND;

    // Load data
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
    cudaMalloc((void**)&SYMBOL, 4*rows);
    cudaMalloc((void**)&BOOL_ARG, 4*rows);
    cudaMalloc((void**)&PROFIT, 4*rows);
    cudaMalloc((void**)&OPERAND, 4*rows*cols);

    cudaMemcpy(SYMBOL, _SYMBOL, 4*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(BOOL_ARG, _BOOL_ARG, 4*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(PROFIT, _PROFIT, 4*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(OPERAND, _OPERAND, 4*rows*cols, cudaMemcpyHostToDevice);

    //
    delete[] _INDEX;
    delete[] _SYMBOL;
    delete[] _BOOL_ARG;
    delete[] _PROFIT;
    delete[] _OPERAND;

    //
    int *_N_formula, *_cp_f_start, *_cp_f_len;
    int num_array;
    int *N_formula, *cp_f_start, *cp_f_len;

    read_binary_file_1d(_N_formula, num_array, folder_data + "/InputData/N_formula.bin");
    read_binary_file_1d(_cp_f_start, num_array, folder_data + "/InputData/cp_f_start.bin");
    read_binary_file_1d(_cp_f_len, num_array, folder_data + "/InputData/cp_f_len.bin");

    cudaMalloc((void**)&N_formula, 4*num_array);
    cudaMalloc((void**)&cp_f_start, 4*num_array);
    cudaMalloc((void**)&cp_f_len, 4*num_array);

    cudaMemcpy(N_formula, _N_formula, 4*num_array, cudaMemcpyHostToDevice);
    cudaMemcpy(cp_f_start, _cp_f_start, 4*num_array, cudaMemcpyHostToDevice);
    cudaMemcpy(cp_f_len, _cp_f_len, 4*num_array, cudaMemcpyHostToDevice);

    //
    delete[] N_formula;
    delete[] cp_f_start;
    delete[] cp_f_len;

    // Tính weight
    float* _N_temp_0;
    _N_temp_0 = new float[num_array*rows];
    float* N_temp_0;
    cudaMalloc((void**)&N_temp_0, num_array*4*rows);
    float* N_temp_1;
    cudaMalloc((void**)&N_temp_1, num_array*4*rows);

    dim3 threads(32);
    int num_block = (num_array + threads.x - 1) / threads.x;
    calculate_formula<<<num_block, threads>>>(
        OPERAND, N_temp_0, N_temp_1, N_formula, cp_f_start, cp_f_len, num_array, eval_method
    ); cudaDeviceSynchronize();

    /* copy weight về host để lưu lại */
    cudaMemcpy(_N_temp_0, N_temp_0, num_array*4*rows, cudaMemcpyDeviceToHost);
    ofstream out(folder_data + "/OutputData/weights.bin", ios::binary);
    if (!out.is_open()) throw runtime_error("Cant open file");
    out.write(reinterpret_cast<const char*>(_N_temp_0), num_array*rows*4);
    out.close();

    // Tính invest L
    float *_N_threshold, *N_threshold;
    read_binary_file_1d(_N_threshold, num_array, folder_data + "/InputData/N_thresholdL.bin");
    cudaMalloc((void**)&N_threshold, 4*num_array);
    cudaMemcpy(N_threshold, _N_threshold, 4*num_array, cudaMemcpyHostToDevice);

    int* _N_arr_invest, *N_arr_invest;
    _N_arr_invest = new int[num_array*rows];
    cudaMalloc((void**)&N_arr_invest, num_array*4*rows);

    StreakInvest<<<num_block, threads.x, NUM_SYMBOL_UNIQUE * threads.x>>>(
        N_temp_0, N_arr_invest, SYMBOL, BOOL_ARG, N_threshold, num_array, center_method_num-1
    );
    cudaMemcpy(_N_arr_invest, N_arr_invest, 4*rows*num_array, cudaMemcpyDeviceToHost);

    /* Lưu N_arr_invest*/
    ofstream outL(folder_data + "/OutputData/N_invest_L.bin", ios::binary);
    if (!outL.is_open()) throw runtime_error("Cant open file");
    outL.write(reinterpret_cast<const char*>(_N_arr_invest), num_array*rows*4);
    outL.close();

    delete[] _N_threshold;

    // Tính invest C
    read_binary_file_1d(_N_threshold, num_array, folder_data + "/InputData/N_thresholdC.bin");
    cudaMemcpy(N_threshold, _N_threshold, 4*num_array, cudaMemcpyHostToDevice);

    StreakInvest<<<num_block, threads.x, NUM_SYMBOL_UNIQUE * threads.x>>>(
        N_temp_0, N_arr_invest, SYMBOL, BOOL_ARG, N_threshold, num_array, center_method_num
    );
    cudaMemcpy(_N_arr_invest, N_arr_invest, 4*rows*num_array, cudaMemcpyDeviceToHost);

    /* Lưu N_arr_invest*/
    ofstream outC(folder_data + "/OutputData/N_invest_C.bin", ios::binary);
    if (!outC.is_open()) throw runtime_error("Cant open file");
    outC.write(reinterpret_cast<const char*>(_N_arr_invest), num_array*rows*4);
    outC.close();

    delete[] _N_threshold;

    // Tính invest R
    read_binary_file_1d(_N_threshold, num_array, folder_data + "/InputData/N_thresholdR.bin");
    cudaMemcpy(N_threshold, _N_threshold, 4*num_array, cudaMemcpyHostToDevice);

    StreakInvest<<<num_block, threads.x, NUM_SYMBOL_UNIQUE * threads.x>>>(
        N_temp_0, N_arr_invest, SYMBOL, BOOL_ARG, N_threshold, num_array, center_method_num + 1
    );
    cudaMemcpy(_N_arr_invest, N_arr_invest, 4*rows*num_array, cudaMemcpyDeviceToHost);

    /* Lưu N_arr_invest*/
    ofstream outR(folder_data + "/OutputData/N_invest_R.bin", ios::binary);
    if (!outR.is_open()) throw runtime_error("Cant open file");
    outR.write(reinterpret_cast<const char*>(_N_arr_invest), num_array*rows*4);
    outR.close();

    delete[] _N_threshold;
}