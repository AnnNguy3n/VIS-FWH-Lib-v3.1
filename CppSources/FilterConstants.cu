#pragma once
#include <cuda_runtime.h>

constexpr int THRESHOLDS_PER_CYCLE = 10;
constexpr int INDEX_LEN = 20;
constexpr int ARRAY_LEN = 5000;
constexpr int NUM_CYCLE_RESULT = 10;
constexpr int NUM_SYMBOL_UNIQUE = 400;
constexpr int NUM_STRATEGY = 5;
constexpr float INTEREST = 1.06f;

constexpr int NUM_CYCLE_DATA = INDEX_LEN - 2;
constexpr int THRESHOLDS_PER_ARRAY = THRESHOLDS_PER_CYCLE * NUM_CYCLE_DATA;
__constant__ int INDEX[INDEX_LEN];
