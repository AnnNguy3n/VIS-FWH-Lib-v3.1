#pragma once
#include "SIMM_Kernel.cu"


class Multi_investMethod: public Generator {
public:
    float *d_threshold;
    float *d_result;
    float *d_final;
    float *h_final;
    int *d_check_save;
    int *h_check_save;

    Multi_investMethod(string config_path);
    ~Multi_investMethod();

    bool compute_result(bool force_save);
};


Multi_investMethod::Multi_investMethod(string config_path)
: Generator(config_path) {
    int num_array     = config.storage_size + cols;

    // Tổng số phần tử
    size_t total_thresholds = static_cast<size_t>(num_array) * THRESHOLDS_PER_ARRAY;
    size_t result_size      = total_thresholds * NUM_CYCLE_RESULT * NUM_STRATEGY;
    size_t final_size       = static_cast<size_t>(num_array) * NUM_CYCLE_RESULT * NUM_STRATEGY * 2;
    size_t check_size       = static_cast<size_t>(num_array) * NUM_CYCLE_RESULT;

    // Cấp phát
    cudaMalloc((void**)&d_threshold, total_thresholds * sizeof(float));
    cudaMalloc((void**)&d_result, result_size * sizeof(float));
    cudaMalloc((void**)&d_final, final_size * sizeof(float));
    cudaMalloc((void**)&d_check_save, check_size * sizeof(int));

    // Khởi tạo giá trị 0 cho kết quả
    int threads = 32;
    int blocks_result = (result_size + threads - 1) / threads;
    cuda_set_array_value<<<blocks_result, threads>>>(d_result, result_size, 0.0);
    cudaDeviceSynchronize();

    // Cấp phát host
    h_final = new float[final_size];
    h_check_save = new int[check_size];
}


Multi_investMethod::~Multi_investMethod(){
    cudaFree(d_threshold);
    cudaFree(d_result);
    cudaFree(d_final);
    cudaFree(d_check_save);
    delete[] h_final;
    delete[] h_check_save;
}


bool Multi_investMethod::compute_result(bool force_save) {
    int num_array     = count_temp_storage;
    dim3 threads(32);

    // 1. Sinh threshold cho từng (array, cycle) → d_threshold
    int num_fill = num_array * (INDEX_LEN - 2);  // = số cycle áp dụng
    int blocks_fill = (num_fill + threads.x - 1) / threads.x;
    fill_thresholds<<<blocks_fill, threads, THRESHOLDS_PER_CYCLE * 4 * threads.x>>>(
        temp_weight_storage, d_threshold,
        num_array
    );cudaDeviceSynchronize();

    // 2. Chạy kernel đầu tư song song trên (array × threshold)
    int total_threads = num_array * THRESHOLDS_PER_ARRAY;
    int blocks_invest = (total_threads + threads.x - 1) / threads.x;
    // Tính dung lượng shared memory
    int shared_mem_bytes = threads.x * (NUM_SYMBOL_UNIQUE + 12*NUM_STRATEGY);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    M_investMethod<<<blocks_invest, threads, shared_mem_bytes>>>(
        temp_weight_storage,
        d_threshold,
        PROFIT,
        SYMBOL,
        BOOL_ARG,
        d_result,
        num_array
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[INFO] M_investMethod executed in %.3f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err)); raise_error("", "");
    }

    // 3. Tìm threshold tối ưu (geo + har) → d_final
    int final_threads = num_array * NUM_CYCLE_RESULT * NUM_STRATEGY;
    int blocks_final = (final_threads + threads.x - 1) / threads.x;

    find_best_results<<<blocks_final, threads>>>(
        d_result,
        d_threshold,
        d_final,
        num_array
    );cudaDeviceSynchronize();

    // 4. Copy kết quả từ device về host
    size_t final_size = static_cast<size_t>(num_array) * NUM_CYCLE_RESULT * NUM_STRATEGY * 2;
    cudaMemcpy(h_final, d_final, final_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Tính toán d_check_save
    dim3 blocks_check(count_temp_storage);       // mỗi block xử lý 1 array
    dim3 threads_check(NUM_CYCLE_RESULT);        // mỗi thread xử lý 1 cycle

    mark_check_save_from_finals<<<blocks_check, threads_check>>>(
        d_final,
        d_check_save,
        count_temp_storage,
        config.eval_threshold
    );cudaDeviceSynchronize();

    // 6. Copy
    cudaMemcpy(h_check_save, d_check_save, sizeof(int) * count_temp_storage * config.num_cycle, cudaMemcpyDeviceToHost);

    //
    return save_result(force_save, h_final, h_check_save);
}