#pragma once
#include "../../Generator/HomogeneousPolynomial/CUDA/HP_Generator.cu"
#include "../../DevMathFunc.cu"
#include <cfloat>


__device__ __forceinline__
void _M_investMethod(
    const float* __restrict__ weight,
    const float* __restrict__ profit,
    const int*   __restrict__ symbol,
    const int*   __restrict__ sufficient_liquidity,

          float* __restrict__ result,

    const float threshold,
    const int   threshold_cycle_idx,
    const int data_window_length
) {
    // Trích các mảng tạm từ shared memory
    extern __shared__ char smem[];

    uint8_t* symbol_streak_base = reinterpret_cast<uint8_t*>(smem);
    float*   tmp_profit_base    = reinterpret_cast<float*>  (symbol_streak_base + blockDim.x * NUM_SYMBOL_UNIQUE);
    float*   tmp_harmean_base   =                            tmp_profit_base    + blockDim.x * NUM_STRATEGY;
    int*     tmp_icount_base    = reinterpret_cast<int*>    (tmp_harmean_base   + blockDim.x * NUM_STRATEGY * data_window_length);

    // Điểm truy cập riêng của thread hiện tại:
    uint8_t* symbol_streak = symbol_streak_base + threadIdx.x * NUM_SYMBOL_UNIQUE;
    float*   tmp_profit    = tmp_profit_base    + threadIdx.x * NUM_STRATEGY;
    float*   tmp_harmean   = tmp_harmean_base   + threadIdx.x * NUM_STRATEGY * data_window_length;
    int*     tmp_icount    = tmp_icount_base    + threadIdx.x * NUM_STRATEGY;

    // Khởi tạo
    int market_streak = 0;

    for (int i = 0; i < NUM_SYMBOL_UNIQUE; ++i) symbol_streak[i] = 0;
    if (data_window_length == 1){for (int i = 0; i < NUM_STRATEGY; ++i) tmp_harmean[i] = 0.0f;}
    else {for (int i = 0; i < NUM_STRATEGY * data_window_length; ++i) tmp_harmean[i] = FLT_MAX;}

    // Duyệt ngược qua từng chu kỳ
    for (int cycle_idx = INDEX_LEN - 2; cycle_idx >= 1; --cycle_idx) {
        int start = INDEX[cycle_idx];
        int end   = INDEX[cycle_idx + 1];

        // Reset bộ đếm đầu tư cho chu kỳ hiện tại
        int N = INDEX_LEN - 1 - cycle_idx;
        int num_applicable_strategy = min(N, NUM_STRATEGY);
        bool any_pass_threshold = false;

        for (int k = 0; k < num_applicable_strategy; ++k) {
            tmp_profit[k] = 0.0f;
            tmp_icount[k] = 0;
        }

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

            // Update cho từng strategy k (tương ứng độ dài streak yêu cầu)
            float p = profit[i];
            for (int k = 0; k < num_applicable_strategy; ++k) {
                if (sym_streak > min(market_streak, k)) {
                    tmp_profit[k] += p;
                    ++tmp_icount[k];
                }
            }
        }

        // Cập nhật harmean tích luỹ
        for (int k = 0; k < num_applicable_strategy; ++k) {
            if (data_window_length == 1)
                tmp_harmean[k] += 1.0f / (tmp_icount[k] ? tmp_profit[k] / static_cast<float>(tmp_icount[k]) : INTEREST);
            else
                tmp_harmean[k * data_window_length + cycle_idx % data_window_length] = 1.0f / (tmp_icount[k] ? tmp_profit[k] / static_cast<float>(tmp_icount[k]) : INTEREST);
        }

        // Lưu kết quả nếu cycle nằm trong vùng cần ghi
        if (cycle_idx <= NUM_CYCLE_RESULT && threshold_cycle_idx + 1 >= cycle_idx) {
            int offset = (NUM_CYCLE_RESULT - cycle_idx) * NUM_STRATEGY;
            for (int k = 0; k < num_applicable_strategy; ++k) {
                if (data_window_length == 1) {
                    result[offset + k] = static_cast<float>(N - k) / tmp_harmean[k];
                }
                else {
                    float tmp_sum = 0.0;
                    for (int wi = 0; wi < data_window_length; ++wi)
                        tmp_sum += tmp_harmean[k * data_window_length + wi];
                    result[offset + k] = static_cast<float>(data_window_length) / tmp_sum;
                }
            }
        }

        // Cập nhật market streak
        market_streak = any_pass_threshold ? market_streak + 1 : 0;
    }
}


__global__ void M_investMethod(
    const float* __restrict__ N_weight,
    const float* __restrict__ N_threshold,
    const float* __restrict__ profit,
    const int*   __restrict__ symbol,
    const int*   __restrict__ sufficient_liquidity,

          float* __restrict__ N_result,

    const int num_array,
    const int data_window_length
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_array * THRESHOLDS_PER_ARRAY) return;

    int thres_idx = tid % THRESHOLDS_PER_ARRAY;
    int array_idx = tid / THRESHOLDS_PER_ARRAY;

    _M_investMethod(
        N_weight + array_idx * ARRAY_LEN,
        profit, symbol, sufficient_liquidity,
        N_result + tid * NUM_CYCLE_RESULT * NUM_STRATEGY,
        N_threshold[tid], thres_idx / THRESHOLDS_PER_CYCLE,
        data_window_length
    );
}


__device__ __forceinline__
void top_N_unique(
    const float* __restrict__ array,
    const int left, const int right,
          float* __restrict__ result,
    const int start
) {
    extern __shared__ float smem1[];
    float* tmp_top = smem1 + threadIdx.x * THRESHOLDS_PER_CYCLE;

    int size = 0;
    int current_min_pos = -1;
    for (int i = left; i < right; ++i) {
        float val = array[i];
        if (f_is_in(tmp_top, 0, size, val)) continue;

        if (size < THRESHOLDS_PER_CYCLE) {
            tmp_top[size] = val;
            if (current_min_pos == -1 || val < tmp_top[current_min_pos])
                current_min_pos = size;
            ++size;
        } else {
            if (val > tmp_top[current_min_pos]) {
                tmp_top[current_min_pos] = val;
                current_min_pos = f_arg_min(tmp_top, 0, size);
            }
        }
    }

    f_fill_value(tmp_top, size, THRESHOLDS_PER_CYCLE, -FLT_MAX);
    f_copy_array(result, tmp_top, start, start + THRESHOLDS_PER_CYCLE, 0);
}


__global__ void fill_thresholds(
    const float* __restrict__ weights,
          float* __restrict__ thresholds,
    const int num_array
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_array * NUM_CYCLE_DATA) return;

    int cycle_idx  =  tid % NUM_CYCLE_DATA;
    int array_idx  =  tid / NUM_CYCLE_DATA;

    top_N_unique(
        weights + array_idx * ARRAY_LEN,
        INDEX[cycle_idx + 1],
        INDEX[cycle_idx + 2],
        thresholds + tid * THRESHOLDS_PER_CYCLE,
        0
    );
}


__global__ void find_best_results(
    const float* __restrict__ results,
    const float* __restrict__ thresholds,
          float* __restrict__ finals,
    const int num_array
) {
    int tid            = blockIdx.x * blockDim.x + threadIdx.x;
    int total_triplets = num_array * NUM_CYCLE_RESULT * NUM_STRATEGY;
    if (tid >= total_triplets) return;

    int strategy =  tid %  NUM_STRATEGY;
    int cycle    = (tid /  NUM_STRATEGY) % NUM_CYCLE_RESULT;
    int array    =  tid / (NUM_STRATEGY * NUM_CYCLE_RESULT);
    float best_har_val = -FLT_MAX, best_har_thr = 0.0f;
    int offset_local = cycle * NUM_STRATEGY + strategy;
    int th_off_glob = array * THRESHOLDS_PER_ARRAY;

    for (int t = 0; t < THRESHOLDS_PER_ARRAY; ++t) {
        int th_off_local = th_off_glob + t;
        const float* result_t = results + th_off_local * NUM_CYCLE_RESULT * NUM_STRATEGY;
        float har_val = result_t[offset_local];

        if (har_val > best_har_val) {
            best_har_val = har_val;
            best_har_thr = thresholds[th_off_local];
        }
    }

    // Ghi kết quả
    float* final_slot = finals + tid * 2;
    final_slot[0] = best_har_thr;
    final_slot[1] = best_har_val;
}


__global__ void mark_check_save_from_finals(
    const float* __restrict__ finals,
          int*    __restrict__ check_save,
    const int num_array,
    const float eval_threshold
) {
    int array_idx = blockIdx.x;
    int cycle_idx = threadIdx.x;
    if (array_idx >= num_array || cycle_idx >= NUM_CYCLE_RESULT) return;

    bool should_save = false;

    for (int s = 0; s < NUM_STRATEGY; ++s) {
        int offset = (array_idx * NUM_CYCLE_RESULT + cycle_idx) * NUM_STRATEGY + s;
        float har_val = finals[offset * 2 + 1];
        if (har_val > eval_threshold) {
            should_save = true;
            break;
        }
    }

    check_save[array_idx * NUM_CYCLE_RESULT + cycle_idx] = should_save;
}
