#include "pch.h"
#include "ellipsoidSolver.h" 

void randomangle_thread(float* data, const float* angle0, const float* dipolefield, const float* dangle, int m, int i, int end, int skips) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<> dist6(1, 100);
    std::normal_distribution<> dist2{ 0.0, 1.0 };
    skips = (skips < 1) ? 1 : skips;
    for (; i < end; i++) {
        data[i * m] = angle0[i];
        double prob = 0.5;
        double sign = 1;
        float data_pre = angle0[i];
        if (dipolefield[i] == 0) {
            for (int j = 1; j < m; j++) {
                for (int skip = 0; skip < skips; skip++) {
                    if (dist6(rng) <= 50)
                        sign = 1;
                    else
                        sign = -1;
                    data_pre = data_pre + dangle[i] * sign;
                }
                data[i * m + j] = data_pre;
            }
            continue;
        }
        double esign = 1;
        if (dipolefield[i] < 0)
            esign = -1;
        for (int j = 1; j < m; j++) {
            for (int skip = 0; skip < skips; skip++) {
                prob = 100.0 / (1.0 + std::exp(dipolefield[i] * 4.05115441e-10 * (std::cos(data_pre - dangle[i]) - std::cos(data_pre + dangle[i]))));
                if ((int)dist6(rng) > (int)prob)
                    sign = -1 * esign;
                else
                    sign = esign;
                data_pre = data_pre + dangle[i] * sign * abs(dist2(rng));
            }
            data[i * m + j] = data_pre;
        }
    }
    return;
}

void randomAngleWalk(float* data, const float* angle0, 
	const float* dipolefield, const float* dangle, int n, int m, int skips) {
    int max_thread = std::thread::hardware_concurrency();
    max_thread = (max_thread < 1) ? 1 : max_thread;
    max_thread = (max_thread > n) ? n : max_thread;
    std::cout << max_thread << " thread!" << std::endl;
    std::vector<std::thread> threads;
    int chunk_size = n / max_thread;
    chunk_size = (chunk_size < 1) ? 1 : chunk_size;
    for (int thread_id = 0; thread_id < max_thread; thread_id++) {
        int start = thread_id * chunk_size;
        int end = (thread_id == max_thread - 1) ? n : (thread_id + 1) * chunk_size;
        threads.emplace_back(randomangle_thread, data, angle0, dipolefield, dangle, m, start, end, skips);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return;
}
