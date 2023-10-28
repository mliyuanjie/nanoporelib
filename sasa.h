#pragma once
#include <iostream>
#include <thread>
#include <cmath>
#include <time.h>
#include <random>
#include <Eigen/Core> 
#include <Eigen/SVD>
#include <Eigen/LU>

extern "C" __declspec(dllexport) void volumeCalculator(const double* xyzr, int length, double probe, double precision, double* volume);
extern "C" __declspec(dllexport) void ellipsoidFit(const double* xyz, int n, double error, double* ellipsoid, double* center);
