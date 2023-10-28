#include "pch.h"
#include "sasa.h" 


namespace  e = Eigen;
int dfs(std::vector<std::vector<std::vector<int>>>& grids, int x, int y, int z, std::vector<int>& border, int xy, int max_x, int* pointer) {
	std::vector<int> stack;
	int point_index = x + z * xy + max_x * y;
	stack.push_back(point_index);
	int number = 0;
	while (!stack.empty()) {
		point_index = stack.back();
		stack.pop_back();
		x = (point_index % xy) % max_x;
		y = (point_index % xy) / max_x;
		z = point_index / xy;
		if (grids[x][y][z] != 0)
			continue;
		number++;
		grids[x][y][z] = 1;
		if (x > border[0]) {
			point_index = x - 1 + z * xy + max_x * y;
			stack.push_back(point_index);
		}
		if (x < border[1] - 1) {
			point_index = x + 1 + z * xy + max_x * y;
			stack.push_back(point_index);
		}
		if (y > border[2]) {
			point_index = x + z * xy + max_x * (y - 1);
			stack.push_back(point_index);
		}
		if (y < border[3] - 1) {
			point_index = x + z * xy + max_x * (y + 1);
			stack.push_back(point_index);
		}
		if (z > border[4]) {
			point_index = x + (z - 1) * xy + max_x * y;
			stack.push_back(point_index);
		}
		if (z < border[5] - 1) {
			point_index = x + (z + 1) * xy + max_x * y;
			stack.push_back(point_index);
		}

	}
	//std::cout << border[0] << " " << border[2] << " " << border[4] << " " << std::endl;
	pointer[0] = number;
	return number;
}

void disablegrids(const double* data, int start, int end, std::vector<std::vector<std::vector<int>>>& grids) {
	for (int it = start * 4; it < end * 4; it += 4) {
		int x = data[it];
		int y = data[it + 1];
		int z = data[it + 2];
		int r = data[it + 3];
		for (int i = x - r; i < x + r; i++) {
			for (int j = y - r; j < y + r; j++)
				for (int k = z - r; k < z + r; k++)
					if (((i - x) * (i - x) + (j - y) * (j - y) + (k - z) * (k - z)) < r * r && grids[i][j][k] == 0) {
						grids[i][j][k] = 1;
					}
		}
	}
}

void getsubvolume(std::vector<std::vector<std::vector<int>>>& grids, std::vector<int>& volumes, int start, int end) {
	int volume = 0;
	for (int i = start; i < end; i++) {
		volume = 0;
		for (int j = 0; j < grids[start].size(); j++) {
			for (int k = 0; k < grids[start][0].size(); k++) {
				if (grids[i][j][k] == 1) {
					volume += 1;
				}
			}
		}
		volumes[i] = volume;
	}
}

void _volumeCalculator(const e::MatrixXd& xyzr_in, double probe, double interval, double* data) {
	int rows = xyzr_in.rows();
	e::Matrix<double, e::Dynamic, e::Dynamic, e::RowMajor> xyzr;
	xyzr = xyzr_in;
	xyzr.col(3) += e::MatrixXd::Constant(rows, 1, probe);
	xyzr = xyzr / interval;
	xyzr = xyzr.array().round();
	e::MatrixXd sub = e::MatrixXd::Constant(rows, 1, 1);
	e::MatrixXd max_xyzr = xyzr.colwise().maxCoeff();
	e::MatrixXd min_xyzr = xyzr.colwise().minCoeff();
	xyzr.col(0) += (-1 * min_xyzr(0) + max_xyzr(3)) * sub;
	max_xyzr(0) += -1 * min_xyzr(0) + 2 * max_xyzr(3);
	min_xyzr(0) += -1 * min_xyzr(0);
	xyzr.col(1) += (-1 * min_xyzr(1) + max_xyzr(3)) * sub;
	max_xyzr(1) += -1 * min_xyzr(1) + 2 * max_xyzr(3);
	min_xyzr(1) += -1 * min_xyzr(1);
	xyzr.col(2) += (-1 * min_xyzr(2) + max_xyzr(3)) * sub;
	max_xyzr(2) += -1 * min_xyzr(2) + 2 * max_xyzr(3);
	min_xyzr(2) += -1 * min_xyzr(2);
	std::vector<std::vector<std::vector<int>>> grids(max_xyzr(0),
		std::vector<std::vector<int>>(max_xyzr(1),
			std::vector<int>(max_xyzr(2), 0)));
	int max_thread = std::thread::hardware_concurrency();
	max_thread -= 1;
	max_thread = (max_thread < 1) ? 1 : max_thread;
	max_thread = (max_thread > rows) ? rows : max_thread;
	std::vector<std::thread> threads;
	int chunk_size = rows / max_thread;
	chunk_size = (chunk_size < 1) ? 1 : chunk_size;
	for (int i = 0; i < max_thread; i++) {
		int start = i * chunk_size;
		int end = (i == max_thread - 1) ? rows : (i + 1) * chunk_size;
		threads.emplace_back(disablegrids, xyzr.data(), start, end, std::ref(grids));
	}
	for (auto& thread : threads) {
		thread.join();
	}
	threads.clear();
	rows = grids.size();
	max_thread = std::thread::hardware_concurrency();
	max_thread -= 1;
	max_thread = (max_thread < 1) ? 1 : max_thread;
	max_thread = (max_thread > rows) ? rows : max_thread;
	std::vector<int> volumes(rows, 0);
	chunk_size = rows / max_thread;
	chunk_size = (chunk_size < 1) ? 1 : chunk_size;
	for (int i = 0; i < max_thread; i++) {
		int start = i * chunk_size;
		int end = (i == max_thread - 1) ? rows : (i + 1) * chunk_size;
		threads.emplace_back(getsubvolume, std::ref(grids), std::ref(volumes), start, end);
	}
	for (auto& thread : threads) {
		thread.join();
	}
	int volume = 0;
	for (int i = 0; i < volumes.size(); i++) {
		volume += volumes[i];
	}
	double volumes_real = double(volume) * interval * interval * interval;
	data[0] = volumes_real;
	threads.clear();

	// cut space

	int xyz_area = max_xyzr(0) * max_xyzr(1);
	std::vector<int> border0 = { 0, int(max_xyzr(0) / 2), 0, int(max_xyzr(1) / 2), 0, int(max_xyzr(2) / 2) };
	std::vector<int> border1 = { int(max_xyzr(0) / 2), int(max_xyzr(0)), 0, int(max_xyzr(1) / 2), 0, int(max_xyzr(2) / 2) };
	std::vector<int> border2 = { 0, int(max_xyzr(0) / 2), 0, int(max_xyzr(1) / 2), int(max_xyzr(2) / 2), int(max_xyzr(2)) };
	std::vector<int> border3 = { int(max_xyzr(0) / 2), int(max_xyzr(0)), 0, int(max_xyzr(1) / 2), int(max_xyzr(2) / 2), int(max_xyzr(2)) };
	std::vector<int> border4 = { 0, int(max_xyzr(0) / 2), int(max_xyzr(1) / 2), int(max_xyzr(1)), 0, int(max_xyzr(2) / 2) };
	std::vector<int> border5 = { int(max_xyzr(0) / 2), int(max_xyzr(0)), int(max_xyzr(1) / 2), int(max_xyzr(1)), 0, int(max_xyzr(2) / 2) };
	std::vector<int> border6 = { 0, int(max_xyzr(0) / 2), int(max_xyzr(1) / 2), int(max_xyzr(1)), int(max_xyzr(2) / 2), int(max_xyzr(2)) };
	std::vector<int> border7 = { int(max_xyzr(0) / 2), int(max_xyzr(0)), int(max_xyzr(1) / 2), int(max_xyzr(1)), int(max_xyzr(2) / 2), int(max_xyzr(2)) };
	int results[8] = { 0 };

	threads.emplace_back(dfs, std::ref(grids), 0, 0, 0, std::ref(border0), xyz_area, max_xyzr(0), results);
	threads.emplace_back(dfs, std::ref(grids), int(max_xyzr(0)) - 1, 0, 0, std::ref(border1), xyz_area, max_xyzr(0), results + 1);
	threads.emplace_back(dfs, std::ref(grids), 0, 0, int(max_xyzr(2)) - 1, std::ref(border2), xyz_area, max_xyzr(0), results + 2);
	threads.emplace_back(dfs, std::ref(grids), int(max_xyzr(0)) - 1, 0, int(max_xyzr(2)) - 1, std::ref(border3), xyz_area, max_xyzr(0), results + 3);
	threads.emplace_back(dfs, std::ref(grids), 0, int(max_xyzr(1)) - 1, 0, std::ref(border4), xyz_area, max_xyzr(0), results + 4);
	threads.emplace_back(dfs, std::ref(grids), int(max_xyzr(0)) - 1, int(max_xyzr(1)) - 1, 0, std::ref(border5), xyz_area, max_xyzr(0), results + 5);
	threads.emplace_back(dfs, std::ref(grids), 0, int(max_xyzr(1)) - 1, int(max_xyzr(2)) - 1, std::ref(border6), xyz_area, max_xyzr(0), results + 6);
	threads.emplace_back(dfs, std::ref(grids), int(max_xyzr(0)) - 1, int(max_xyzr(1)) - 1, int(max_xyzr(2)) - 1, std::ref(border7), xyz_area, max_xyzr(0), results + 7);
	for (auto& thread : threads)
		thread.join();
	int volume_voids = 0;
	for (int i = 0; i < 8; i++) {
		volume_voids += results[i];
	}
	if (grids[int(max_xyzr(0) / 2)][int(max_xyzr(1) / 2)][int(max_xyzr(1) / 2)] == 1) {
		std::vector<int> border = { 0, int(max_xyzr(0)) - 1, 0, int(max_xyzr(1)) - 1, 0, int(max_xyzr(2)) - 1 };
		volume_voids += dfs(grids, int(max_xyzr(0) / 2), int(max_xyzr(1) / 2), int(max_xyzr(2) / 2), border, xyz_area, max_xyzr(0), results);
	}
	double volumes_voids_real = double(xyz_area * max_xyzr(2) - volume_voids) * interval * interval * interval;
	data[1] = volumes_voids_real;
}

void volumeCalculator(const double* xyzr, int length, double probe, double precision, double* volume) {
	e::Map<const e::Matrix<double, e::Dynamic, 4, e::ColMajor>> xyzr_mat(xyzr, length, 4);
	_volumeCalculator(xyzr_mat, probe, precision, volume);
}

void ellipsoidFit(const double* xyz, int n, double tolerance, double* ellipsoid, double* center) {
	/*fit the ellipsoid with khachiyan algorithm*/ 
	e::MatrixXd Q(4, n);
	e::MatrixXd X(4, 4);
	e::VectorXd M(n);
	Q.setZero();
	e::Map<const e::MatrixXd> P(xyz, 3, n);
	Q.block(0, 0, 3, n) = P;
	Q.block(3, 0, 1, n) = e::MatrixXd::Constant(1, n, 1); 
	e::VectorXd u = e::VectorXd::Constant(n, 1.0 / double(n));
	e::VectorXd new_u(n);
	e::MatrixXd diagonal_u(n, n);
	double err = 1;
	double maximum = 0;
	int j = 0;
	double step_size = 0;
	int count = 0;
	while (err > tolerance) {
		diagonal_u = u.asDiagonal();
		X = Q * diagonal_u * Q.transpose();
		M = (Q.transpose() * X.inverse() * Q).diagonal();
		maximum = M.maxCoeff();
		for (int i = 0; i < n; i++) 
			if (M(i) == maximum) {
				j = i;
				break;
			}
		step_size = (maximum - 4.0) / ((4.0 * maximum - 4.0));
		new_u = (1.0 - step_size) * u;
		new_u(j) = new_u(j) + step_size;
		err = (new_u - u).norm();
		u = new_u;
		count++;
	}
	e::MatrixXd U(n, n); 
	U = u.asDiagonal();
	e::Map<e::Matrix<double, 3, 3, e::RowMajor>> A(ellipsoid);
	e::Map<e::Vector<double, 3>> C(center);
	C = P * u;
	A = 1.0 / 3.0 * (P * U * P.transpose() - (P * u)*(P * u).transpose()).inverse();
	return;
}