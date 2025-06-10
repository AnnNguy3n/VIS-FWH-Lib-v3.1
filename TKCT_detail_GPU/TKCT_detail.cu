#pragma once
#include <cfloat>
#include "../CppSources/Generator/HomogeneousPolynomial/CUDA/HP_Kernel.cu"
#include "../CppSources/DevMathFunc.cu"
#include <iostream>
#include <string>
#include "../CppSources/Utils/WorkWithFile.cpp"
#include <fstream>
#include <chrono>
using namespace std;


__constant__ int NUM_SYMBOL_UNIQUE;
__constant__ int INDEX_LEN;
__constant__ int ARRAY_LEN;

__constant__ int INDEX[100];

constexpr int CHUNK_SIZE = 192;


__device__ __forceinline__
void _calculate_formula(
    const int*   __restrict__ formula,
    const int F_len,
    const int eval_method,
          float* __restrict__ result,
    const int offset,
    const float* __restrict__ d_OPERAND
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
            f_copy_array(temp_1, d_OPERAND + oprand * ARRAY_LEN + offset, 0, CHUNK_SIZE, 0);
        } else {
            if (oprt == 2) {
                ++deg;
                f_mul_inpl(temp_1, d_OPERAND + oprand * ARRAY_LEN + offset, CHUNK_SIZE);
            } else {
                --deg;
                f_div_inpl(temp_1, d_OPERAND + oprand * ARRAY_LEN + offset, CHUNK_SIZE);
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

    int end_offset = min(offset + CHUNK_SIZE, ARRAY_LEN);
    while (end_offset-- > offset) {
        int idx = end_offset - offset;
        if (isnan(temp_0[idx]) || isinf(temp_0[idx]))
            temp_0[idx] = -FLT_MAX;

        result[end_offset] = temp_0[idx];
    }
}


__global__ void calculate_formula(
    const int*   __restrict__ N_formula,
          float* __restrict__ N_result,
    const int*   __restrict__ cp_f_start,
    const int*   __restrict__ cp_f_len,
    const int num_array,
    const int eval_method,
    const int offset,
    const float* __restrict__ d_OPERAND
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_array) return;

    const int start = cp_f_start[tid];
    const int F_len = cp_f_len[tid];
    const int* formula = &N_formula[start];
    float* result = N_result + tid * ARRAY_LEN;

    _calculate_formula(formula, F_len, eval_method, result, offset, d_OPERAND);
}


__device__ __forceinline__
void _StreakInvest(
    const float* __restrict__ weight,
          bool*  __restrict__ arr_invest,
    const int*   __restrict__ symbol,
    const int*   __restrict__ sufficient_liquidity,
    const float threshold,
    const int   method_num
) {
    // Trích các mảng tạm từ shared memory
    extern __shared__ char smem1[];

    uint8_t* symbol_streak_base = reinterpret_cast<uint8_t*>(smem1);
    uint8_t* symbol_streak = symbol_streak_base + threadIdx.x * NUM_SYMBOL_UNIQUE;

    // Khởi tạo
    int market_streak = 0;
    for (int i = 0; i < NUM_SYMBOL_UNIQUE; ++i) symbol_streak[i] = 0;
    for (int i = 0; i < ARRAY_LEN; ++i) arr_invest[i] = false;

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
          bool*  __restrict__ N_arr_invest,
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


int main(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();
    string folder_data = argv[1];
    int eval_method = stoi(argv[2]);
    int center_method_num = stoi(argv[3]);

    // Load data
    int *_INDEX, *_SYMBOL, *_BOOL_ARG;
    float *_PROFIT, *_OPERAND;
    int index_len, rows, cols;
    read_binary_file_1d(_INDEX, index_len, folder_data + "/InputData/INDEX.bin");
    read_binary_file_1d(_SYMBOL, rows, folder_data + "/InputData/SYMBOL.bin");
    read_binary_file_1d(_BOOL_ARG, rows, folder_data + "/InputData/BOOL_ARG.bin");
    read_binary_file_1d(_PROFIT, rows, folder_data + "/InputData/PROFIT.bin");
    read_binary_file_2d(_OPERAND, cols, rows, folder_data + "/InputData/OPERAND.bin");

    int *SYMBOL, *BOOL_ARG;
    float *PROFIT, *d_OPERAND;
    cudaMalloc((void**)&d_OPERAND, rows * cols * sizeof(float));
    cudaMemcpy(d_OPERAND, _OPERAND, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&SYMBOL, rows * sizeof(float));
    cudaMalloc((void**)&BOOL_ARG, rows * sizeof(float));
    cudaMalloc((void**)&PROFIT, rows * sizeof(float));
    cudaMemcpy(SYMBOL, _SYMBOL, rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(BOOL_ARG, _BOOL_ARG, rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(PROFIT, _PROFIT, rows * sizeof(float), cudaMemcpyHostToDevice);

    // Ghi vào constants
    int max_ = 0;
    for (int i = 0; i < rows; ++i) {
        if (_SYMBOL[i] > max_) max_ = _SYMBOL[i];
    }
    ++max_;
    cudaMemcpyToSymbol(INDEX_LEN, &index_len, 4);
    cudaMemcpyToSymbol(INDEX, _INDEX, 4 * index_len);
    cudaMemcpyToSymbol(ARRAY_LEN, &rows, 4);
    cudaMemcpyToSymbol(NUM_SYMBOL_UNIQUE, &max_, 4);

    // Đọc formula
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

    // Tính toán weights
    float *host_N_result, *dev_N_result;
    host_N_result = new float[num_array*rows];
    cudaMalloc((void**)&dev_N_result, num_array*rows*4);

    dim3 threads(32);
    int num_block = (num_array + threads.x - 1) / threads.x;

    for (int offset = 0; offset < rows; offset += CHUNK_SIZE) {
        calculate_formula<<<num_block, threads, threads.x * 2 * CHUNK_SIZE * 4>>>(
            N_formula, dev_N_result, cp_f_start, cp_f_len, num_array, eval_method, offset, d_OPERAND
        ); cudaDeviceSynchronize();
    }
    cudaMemcpy(host_N_result, dev_N_result, num_array*rows*4, cudaMemcpyDeviceToHost);

    // Lưu kết quả công thức
    ofstream outRes(folder_data + "/OutputData/N_result.bin", ios::binary);
    if (!outRes.is_open()) throw runtime_error("Cant open N_result.bin");
    outRes.write(reinterpret_cast<const char*>(host_N_result), num_array * rows * sizeof(float));
    outRes.close();

    // Tính invest L, C, R
    float *_N_threshold, *N_threshold;
    cudaMalloc((void**)&N_threshold, 4*num_array);

    bool* _N_arr_invest, *N_arr_invest;
    _N_arr_invest = new bool[num_array*rows];
    cudaMalloc((void**)&N_arr_invest, num_array*rows);

    --center_method_num;
    for (char c : string("LCR")) {
        read_binary_file_1d(_N_threshold, num_array, folder_data + "/InputData/N_threshold" + string(1, c) +  ".bin");
        cudaMemcpy(N_threshold, _N_threshold, 4*num_array, cudaMemcpyHostToDevice);

        StreakInvest<<<num_block, threads, max_ * threads.x>>>(
            dev_N_result, N_arr_invest, SYMBOL, BOOL_ARG, N_threshold, num_array, center_method_num
        ); cudaDeviceSynchronize();
        ++center_method_num;
        cudaMemcpy(_N_arr_invest, N_arr_invest, rows*num_array, cudaMemcpyDeviceToHost);

        /* Lưu N_arr_invest*/
        ofstream outL(folder_data + "/OutputData/N_invest_" + string(1, c) + ".bin", ios::binary);
        if (!outL.is_open()) throw runtime_error("Cant open file");
        outL.write(reinterpret_cast<const char*>(_N_arr_invest), num_array*rows);
        outL.close();

        delete[] _N_threshold;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    delete[] _INDEX;
    delete[] _SYMBOL;
    delete[] _PROFIT;
    delete[] _BOOL_ARG;
    delete[] _OPERAND;
    delete[] _N_arr_invest;
    delete[] host_N_result;
    delete[] _N_formula;
    delete[] _cp_f_start;
    delete[] _cp_f_len;

    cudaFree(SYMBOL);
    cudaFree(PROFIT);
    cudaFree(BOOL_ARG);
    cudaFree(d_OPERAND);
    cudaFree(dev_N_result);
    cudaFree(N_formula);
    cudaFree(cp_f_start);
    cudaFree(cp_f_len);
    cudaFree(N_arr_invest);
    cudaFree(N_threshold);
}