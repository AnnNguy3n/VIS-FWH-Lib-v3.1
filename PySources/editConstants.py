from PySources.base import Base, np


def edit_constants(vis: Base, config: dict):
    constants_path = __file__.replace("PySources", "CppSources").replace("editConstants.py", "FilterConstants.cu")

    text = f"""#pragma once
#include <cuda_runtime.h>

constexpr int THRESHOLDS_PER_CYCLE = {config.get("thresholds_per_cycle", 10)};
constexpr int INDEX_LEN = {vis.INDEX.size};
constexpr int ARRAY_LEN = {vis.OPERAND.shape[1]};
constexpr int NUM_CYCLE_RESULT = {config["num_cycle_result"]};
constexpr int NUM_SYMBOL_UNIQUE = {max(vis.SYMBOL) + 1};
constexpr int NUM_STRATEGY = {config["num_strategy"]};
constexpr float INTEREST = {vis.INTEREST}f;

constexpr int NUM_CYCLE_DATA = INDEX_LEN - 2;
constexpr int THRESHOLDS_PER_ARRAY = THRESHOLDS_PER_CYCLE * NUM_CYCLE_DATA;
__constant__ int INDEX[INDEX_LEN];
"""
    with open(constants_path, "w") as f:
        f.write(text)