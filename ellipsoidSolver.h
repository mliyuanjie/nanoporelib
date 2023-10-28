#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>
#include <random>

extern "C" __declspec(dllexport) void randomAngleWalk(float* data_array, const float* angle0_array, const float* dipolefield_array, const float* dangle_array, int n, int m, int skips);
