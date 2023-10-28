#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>
#include <random>

extern "C" __declspec(dllexport) void diffusionRandom(float* data, const float* dx, int number, float region, float maxregion, int* maxstep);