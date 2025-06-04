#pragma once
#include <cfloat>
#include "../CppSources/Generator/HomogeneousPolynomial/CUDA/HP_Kernel.cu"
#include "../CppSources/DevMathFunc.cu"
#include <iostream>
#include <string>
#include "../CppSources/Utils/WorkWithFile.cpp"
#include <fstream>
#include <chrono>
#include <vector>
using namespace std;


__constant__ int NUM_SYMBOL_UNIQUE;
__constant__ int INDEX_LEN;
__constant__ int ARRAY_LEN;

__constant__ int INDEX[100];

constexpr int CHUNK_SIZE = 192;
constexpr int TEMP_SIZE = 5000;
constexpr int TARGET_SAVED = 10000;


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


bool arrays_are_equal(const bool* a, const bool* b, size_t size) {
    bool all_false = true;
    for (size_t i = 0; i < size; ++i) {
        if (a[i] ^ b[i]) return false;
        if (a[i] | b[i]) all_false = false;
    }
    return !all_false;
}


bool check_similar(
    bool* AL, bool* AC, bool* AR,
    bool* BL, bool* BC, bool* BR,
    int center_method_num, float tk_new_rate,
    int* INDEX, int index_len
) {
    int count_similar = 0;
    int num_cycle_cal = index_len - 1 - center_method_num;

    for (int i = num_cycle_cal; i > 0; --i) {
        int start = INDEX[i];
        int end   = INDEX[i + 1];
        count_similar += arrays_are_equal(AC + start, BC + start, end - start);
    }
    for (int i = num_cycle_cal - 1; i > 0; --i) {
        int start = INDEX[i];
        int end   = INDEX[i + 1];
        count_similar += arrays_are_equal(AR + start, BR + start, end - start);
    }
    for (int i = num_cycle_cal + 1; i > 0; --i) {
        int start = INDEX[i];
        int end   = INDEX[i + 1];
        count_similar += arrays_are_equal(AL + start, BL + start, end - start);
    }

    float rate = (float)count_similar / (3.0f * num_cycle_cal);
    return rate >= tk_new_rate;
}


int main(int argc, char* argv[]) {
    auto start_timing = std::chrono::high_resolution_clock::now();
    string folder_data = argv[1];
    int eval_method = stoi(argv[2]);
    int center_method_num = stoi(argv[3]);
    float tk_new_rate = stof(argv[4]);

    // Load Data
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
    int _num_symbol_unique = 0;
    for (int i = 0; i < rows; ++i) {
        if (_SYMBOL[i] > _num_symbol_unique) _num_symbol_unique = _SYMBOL[i];
    }
    ++_num_symbol_unique;
    cudaMemcpyToSymbol(INDEX_LEN, &index_len, 4);
    cudaMemcpyToSymbol(INDEX, _INDEX, 4 * index_len);
    cudaMemcpyToSymbol(ARRAY_LEN, &rows, 4);
    cudaMemcpyToSymbol(NUM_SYMBOL_UNIQUE, &_num_symbol_unique, 4);

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

    // Đọc threshold LCR
    float *_N_threshold_L, *N_threshold_L;
    cudaMalloc((void**)&N_threshold_L, 4*num_array);
    read_binary_file_1d(_N_threshold_L, num_array, folder_data + "/InputData/N_thresholdL.bin");
    cudaMemcpy(N_threshold_L, _N_threshold_L, 4*num_array, cudaMemcpyHostToDevice);

    float *_N_threshold_C, *N_threshold_C;
    cudaMalloc((void**)&N_threshold_C, 4*num_array);
    read_binary_file_1d(_N_threshold_C, num_array, folder_data + "/InputData/N_thresholdC.bin");
    cudaMemcpy(N_threshold_C, _N_threshold_C, 4*num_array, cudaMemcpyHostToDevice);

    float *_N_threshold_R, *N_threshold_R;
    cudaMalloc((void**)&N_threshold_R, 4*num_array);
    read_binary_file_1d(_N_threshold_R, num_array, folder_data + "/InputData/N_thresholdR.bin");
    cudaMemcpy(N_threshold_R, _N_threshold_R, 4*num_array, cudaMemcpyHostToDevice);

    // Chuẩn bị các mảng tạm
    /* Mảng weight */
    float *host_N_result, *dev_N_result;
    host_N_result = new float[TEMP_SIZE*rows];
    cudaMalloc((void**)&dev_N_result, TEMP_SIZE*rows*4);

    /* Các mảng invest LCR */
    bool* host_N_arr_invest_L, *N_arr_invest_L;
    host_N_arr_invest_L = new bool[TEMP_SIZE*rows];
    cudaMalloc((void**)&N_arr_invest_L, TEMP_SIZE*rows);

    bool* host_N_arr_invest_C, *N_arr_invest_C;
    host_N_arr_invest_C = new bool[TEMP_SIZE*rows];
    cudaMalloc((void**)&N_arr_invest_C, TEMP_SIZE*rows);

    bool* host_N_arr_invest_R, *N_arr_invest_R;
    host_N_arr_invest_R = new bool[TEMP_SIZE*rows];
    cudaMalloc((void**)&N_arr_invest_R, TEMP_SIZE*rows);

    // Các mảng lưu kết quả
    bool *is_saved = new bool[num_array];
    vector<float*> saved_arr_weight;
    vector<bool*> saved_invest_L;
    vector<bool*> saved_invest_C;
    vector<bool*> saved_invest_R;
    int count_saved = 0;

    dim3 threads(32);
    int num_block = (TEMP_SIZE + threads.x - 1) / threads.x;

    // Gán mảng is_saved = false
    memset(is_saved, 0, num_array * sizeof(bool));

    // Tính toán từng đoạn nhỏ, mỗi lần tối đa TEMP_SIZE công thức
    for (int start_index = 0; start_index < num_array; start_index += TEMP_SIZE) {
        int end_index = min(start_index + TEMP_SIZE, num_array);

        { // Tính toán weights và các array invest LCR
            // Tính weight
            for (int offset = 0; offset < rows; offset += CHUNK_SIZE) {
                calculate_formula<<<num_block, threads, threads.x * 2 * CHUNK_SIZE * 4>>>(
                    N_formula + cp_f_start[start_index],
                    dev_N_result,
                    cp_f_start + start_index,
                    cp_f_len + start_index,
                    end_index - start_index,
                    eval_method,
                    offset,
                    d_OPERAND
                ); cudaDeviceSynchronize();
            }
            cudaMemcpy(host_N_result, dev_N_result, TEMP_SIZE*rows*4, cudaMemcpyDeviceToHost);

            // Tính invest LCR
            StreakInvest<<<num_block, threads, _num_symbol_unique * threads.x>>>(
                dev_N_result,
                N_arr_invest_L,
                SYMBOL,
                BOOL_ARG,
                N_threshold_L + start_index,
                end_index - start_index,
                center_method_num - 1
            ); cudaDeviceSynchronize();
            cudaMemcpy(host_N_arr_invest_L, N_arr_invest_L, rows*TEMP_SIZE, cudaMemcpyDeviceToHost);

            StreakInvest<<<num_block, threads, _num_symbol_unique * threads.x>>>(
                dev_N_result,
                N_arr_invest_C,
                SYMBOL,
                BOOL_ARG,
                N_threshold_C + start_index,
                end_index - start_index,
                center_method_num
            ); cudaDeviceSynchronize();
            cudaMemcpy(host_N_arr_invest_C, N_arr_invest_C, rows*TEMP_SIZE, cudaMemcpyDeviceToHost);

            StreakInvest<<<num_block, threads, _num_symbol_unique * threads.x>>>(
                dev_N_result,
                N_arr_invest_R,
                SYMBOL,
                BOOL_ARG,
                N_threshold_R + start_index,
                end_index - start_index,
                center_method_num + 1
            ); cudaDeviceSynchronize();
            cudaMemcpy(host_N_arr_invest_R, N_arr_invest_R, rows*TEMP_SIZE, cudaMemcpyDeviceToHost);
        }

        // Lọc
        for (int fml_id = start_index; fml_id < end_index; ++fml_id) {
            int local_id = fml_id - start_index;
            int fml_offset = rows * local_id;
            float* fml_weight = host_N_result + fml_offset;
            bool* fml_invest_L = host_N_arr_invest_L + fml_offset;
            bool* fml_invest_C = host_N_arr_invest_C + fml_offset;
            bool* fml_invest_R = host_N_arr_invest_R + fml_offset;

            bool is_similar = false;
            for (int saved_id = saved_arr_weight.size() - 1; saved_id >= 0; --saved_id) {
                if (check_similar(
                    fml_invest_L, fml_invest_C, fml_invest_R,
                    saved_invest_L[saved_id], saved_invest_C[saved_id], saved_invest_R[saved_id],
                    center_method_num, tk_new_rate, _INDEX, index_len
                )) {is_similar = true; break;}
            }
            if (!is_similar) {
                float* saved_fml_weight = new float[rows];
                bool* saved_fml_invest_L = new bool[rows];
                bool* saved_fml_invest_C = new bool[rows];
                bool* saved_fml_invest_R = new bool[rows];
                memcpy(saved_fml_weight, fml_weight, 4*rows);
                memcpy(saved_fml_invest_L, fml_invest_L, rows);
                memcpy(saved_fml_invest_C, fml_invest_C, rows);
                memcpy(saved_fml_invest_R, fml_invest_R, rows);
                saved_arr_weight.push_back(saved_fml_weight);
                saved_invest_L.push_back(saved_fml_invest_L);
                saved_invest_C.push_back(saved_fml_invest_C);
                saved_invest_R.push_back(saved_fml_invest_R);

                ++count_saved;
                is_saved[fml_id] = true;
            }
        }
    }

    int saved_start_index;
    if (count_saved > TARGET_SAVED)
        saved_start_index = count_saved - TARGET_SAVED;
    else
        saved_start_index = 0;

    // Lưu kết quả công thức
    ofstream outRes(folder_data + "/OutputData/N_result.bin", ios::binary);
    if (!outRes.is_open()) throw runtime_error("Cant open N_result.bin");
    for (int i = saved_start_index; i < saved_arr_weight.size(); ++i)
        outRes.write(reinterpret_cast<const char*>(saved_arr_weight[i]), rows * sizeof(float));
    outRes.close();

    // Lưu invest LCR
    ofstream out_L(folder_data + "/OutputData/N_invest_L.bin", ios::binary);
    if (!out_L.is_open()) throw runtime_error("Cant open file N_invest_L");
    for (int i = saved_start_index; i < saved_arr_weight.size(); ++i)
        out_L.write(reinterpret_cast<const char*>(saved_invest_L[i]), rows * sizeof(bool));
    out_L.close();

    ofstream out_C(folder_data + "/OutputData/N_invest_C.bin", ios::binary);
    if (!out_C.is_open()) throw runtime_error("Cant open file N_invest_C");
    for (int i = saved_start_index; i < saved_arr_weight.size(); ++i)
        out_C.write(reinterpret_cast<const char*>(saved_invest_C[i]), rows * sizeof(bool));
    out_C.close();

    ofstream out_R(folder_data + "/OutputData/N_invest_R.bin", ios::binary);
    if (!out_R.is_open()) throw runtime_error("Cant open file N_invest_R");
    for (int i = saved_start_index; i < saved_arr_weight.size(); ++i)
        out_R.write(reinterpret_cast<const char*>(saved_invest_R[i]), rows * sizeof(bool));
    out_R.close();

    // Giải phóng các con trỏ
    cudaFree(SYMBOL);
    cudaFree(PROFIT);
    cudaFree(BOOL_ARG);
    cudaFree(d_OPERAND);
    cudaFree(dev_N_result);
    cudaFree(N_formula);
    cudaFree(cp_f_start);
    cudaFree(cp_f_len);
    cudaFree(N_arr_invest_L);
    cudaFree(N_threshold_L);
    cudaFree(N_arr_invest_C);
    cudaFree(N_threshold_C);
    cudaFree(N_arr_invest_R);
    cudaFree(N_threshold_R);

    delete[] _INDEX;
    delete[] _SYMBOL;
    delete[] _PROFIT;
    delete[] _BOOL_ARG;
    delete[] _OPERAND;
    delete[] host_N_arr_invest_L;
    delete[] host_N_arr_invest_C;
    delete[] host_N_arr_invest_R;
    delete[] host_N_result;
    delete[] _N_formula;
    delete[] _cp_f_start;
    delete[] _cp_f_len;
    delete[] _N_threshold_L;
    delete[] _N_threshold_C;
    delete[] _N_threshold_R;
    delete[] is_saved;
    for (int i = 0; i < saved_arr_weight.size(); ++i) {
        delete[] saved_arr_weight[i];
        delete[] saved_invest_L[i];
        delete[] saved_invest_C[i];
        delete[] saved_invest_R[i];
    }

    auto end_timing = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_timing - start_timing;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;
}
