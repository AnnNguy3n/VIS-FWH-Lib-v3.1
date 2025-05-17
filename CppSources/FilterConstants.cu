#pragma once
#include <cuda_runtime.h>

constexpr int THRESHOLDS_PER_CYCLE = 5;
constexpr int INDEX_LEN = 16;
constexpr int ARRAY_LEN = 4018;
constexpr int NUM_CYCLE_RESULT = 11;
constexpr int NUM_SYMBOL_UNIQUE = 463;
constexpr int NUM_STRATEGY = 3;
constexpr float INTEREST = 1.15f;

constexpr int NUM_CYCLE_DATA = INDEX_LEN - 2;
constexpr int THRESHOLDS_PER_ARRAY = THRESHOLDS_PER_CYCLE * NUM_CYCLE_DATA;
__constant__ int INDEX[INDEX_LEN];
