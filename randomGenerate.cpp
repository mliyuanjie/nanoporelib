#include "pch.h"
#include "randomGenerate.h"
#include <random>
#include <thread>


void fillArray(double* array, int start, int end) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d{ 0.0, 1.0 };
    
    for (int i = start; i < end; ++i) {
        array[i] = d(gen);
    }
}

void generateRandomNumbers(double* array, int n) {
    int max_thread = std::thread::hardware_concurrency();
    max_thread = (max_thread < 1) ? 1 : max_thread;
    max_thread = (max_thread > n) ? n : max_thread;
    std::vector<std::thread> threads;
    int chunk_size = n / max_thread;
    chunk_size = (chunk_size < 1) ? 1 : chunk_size;
    for (int thread_id = 0; thread_id < max_thread; thread_id++) {
        int start = thread_id * chunk_size;
        int end = (thread_id == max_thread - 1) ? n : (thread_id + 1) * chunk_size;
        threads.emplace_back(fillArray, array, start, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return;
}