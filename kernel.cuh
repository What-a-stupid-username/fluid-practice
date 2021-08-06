#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define GRID_COUNT 512

void Run(int LoopNum, void* validList, int validCount, void* A, void* b, void* res);