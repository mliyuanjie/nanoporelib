#include "pch.h"
#include "beadsClusterSolver.h" 

void CurrentSolver(const e::Matrix3d& RM0_array,
    const e::Matrix3d& EFM_in, const e::Matrix3d& Dipole_in,
    const e::Matrix<double, 3, 6>& Mob_in, const e::Matrix<double, 3, 6>& SQM_in,
    const e::Matrix<double, 3, 1>& E, const e::Matrix3d& dE, const e::Matrix3d& Res_in, double diffusion_reduce, double electrophoretic_reduce,
    int skip, int n, double* const data_ptr) {
    std::random_device device;
    std::mt19937 urng{ device() };
    std::normal_distribution<double> dis{ 0, 1 };
    e::Matrix<double, 3, 6> Mob;
    e::Matrix<double, 3, 3> EFM;
    e::Matrix<double, 3, 3> Dipole;
    e::Vector3d Dipole_torque;
    e::Matrix<double, 3, 6> SQM;
    e::Matrix<double, 3, 3> RM = RM0_array;
    e::Matrix<double, 3, 3> RM_tmp = RM0_array;
    e::Matrix<double, 3, 1> Res;
    e::Matrix<double, 3, 1> increase;
    e::Matrix<double, 6, 1> Ft;
    e::Matrix<double, 6, 1> rand_number;

    int kk = 0;
    skip = (skip < 1) ? 1 : skip;
    for (int i = 0; i < n * skip; i++) {
        Mob.block<3, 3>(0, 0).noalias() = RM * Mob_in.block<3, 3>(0, 0) * RM.transpose();
        Mob.block<3, 3>(0, 3).noalias() = RM * Mob_in.block<3, 3>(0, 3) * RM.transpose();
        SQM.block<3, 3>(0, 0).noalias() = RM * SQM_in.block<3, 3>(0, 0) * RM.transpose();
        SQM.block<3, 3>(0, 3).noalias() = RM * SQM_in.block<3, 3>(0, 3) * RM.transpose();
        EFM.noalias() = RM * EFM_in * RM.transpose();
        Dipole.noalias() = RM * Dipole_in * RM.transpose();
        Dipole_torque = Dipole * E;
        Ft.block<3, 1>(0, 0).noalias() = (Dipole_torque.transpose() * dE).transpose();
        Ft.block<3, 1>(3, 0).noalias() = Dipole_torque.cross(E);
        for (int ri = 0; ri < 6; ri++) {
            rand_number(ri, 0) = dis(urng);
        }
        increase.noalias() = diffusion_reduce * (SQM * rand_number);
        increase.noalias() += electrophoretic_reduce * (EFM * E);
        increase.noalias() += Mob * Ft;
        for (int j = 0; j < 3; j++) {
            RM_tmp.col(j) = RM.col(j) + increase.cross(RM.col(j))
                - 1 / 2 * (RM.col(j) * (increase.transpose() * increase))
                - (RM.col(j).transpose() * increase) * increase;
        }
        RM = RM_tmp;
        e::JacobiSVD<e::MatrixXd> svd;
        svd.compute(RM, e::ComputeFullU | e::ComputeFullV);
        RM = svd.matrixU() * svd.matrixV().transpose();
        if (i % skip == 0) {
            Res = (RM * Res_in * RM.transpose()) * E;
            data_ptr[kk] = Res(2);
            kk++;
        }
    }
    return;
}

void Threadcall(const double* RM0_array,
    const double* EFM_in, const double* Dipole_in,
    const double* Mob_in, const double* SQM_in,
    const double* E_in, const double* dE_in, const double* Res_in, const double* diffusion_reduce, const double* electrophoretic_reduce,
    const int* skip, int col, int row, double* const data_ptr) {
    for (int i = 0; i < row; i++) {
        e::Map<const e::Matrix<double, 3, 3, e::RowMajor>> RM0(RM0_array + i * 9);
        e::Map<const e::Matrix<double, 3, 3, e::RowMajor>> EFM(EFM_in + i * 9);
        e::Map<const e::Matrix<double, 3, 3, e::RowMajor>> Dipole(Dipole_in + i * 9);
        e::Map<const e::Matrix<double, 3, 6, e::RowMajor>> Mob(Mob_in + i * 18);
        e::Map<const e::Matrix<double, 3, 6, e::RowMajor>> SQM(SQM_in + i * 18);
        e::Map<const e::Matrix<double, 3, 1>, e::RowMajor> E(E_in + i * 3);
        e::Map<const e::Matrix<double, 3, 3, e::RowMajor>> dE(dE_in + i * 9);
        e::Map<const e::Matrix<double, 3, 3, e::RowMajor>> Res(Res_in + i * 9);
        CurrentSolver(RM0, EFM, Dipole, Mob, SQM, E, dE, Res,
            diffusion_reduce[i], electrophoretic_reduce[i],
            skip[i], col, data_ptr + i * col);
    }
}

void blockCurrent(const double* RM0_array,
    const double* EFM_in, const double* Dipole_in,
    const double* Mob_in, const double* SQM_in,
    const double* E_in, const double* dE_in, const double* Res_in, const double* diffusion_reduce, const double* electrophoretic_reduce,
    const int* skip, int col, int row, double* const data_ptr) {
    /* input
    m, n = row, col, rwo means the batch, col means the sequence length.
    RM0_array:  m * 3 * 3
    EFM_in:  m * 3 * 3
    Dipole_in: m * 3 * 3
    Mob_in: m * 3 * 6
    SQM_in: m * 3 * 6
    E: m * 3 * 1
    dt: m * 3 * 3
    data_ptr = m * n * 9
    parallel with core number
    */
    int max_thread = std::thread::hardware_concurrency();
    max_thread = (max_thread < 1) ? 1 : max_thread;
    max_thread = (max_thread > row) ? row : max_thread;
    std::cout << max_thread << " thread!" << std::endl;
    std::vector<std::thread> threads;
    int chunk_size = row / max_thread;
    chunk_size = (chunk_size < 1) ? 1 : chunk_size;
    for (int thread_id = 0; thread_id < max_thread; thread_id++) {
        int start = thread_id * chunk_size;
        int end = (thread_id == max_thread - 1) ? row : (thread_id + 1) * chunk_size;
        threads.emplace_back(Threadcall, RM0_array + start * 9,
            EFM_in + start * 9, Dipole_in + start * 9,
            Mob_in + start * 18, SQM_in + start * 18,
            E_in + start * 3, dE_in + start * 9, Res_in + start * 9, diffusion_reduce + start, electrophoretic_reduce + start,
            skip + start, col, end - start, data_ptr + start * col);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return;
}


void RotationMatrixSolver(const e::Matrix3d& RM0_array,
    const e::Matrix3d& EFM_in, const e::Matrix3d& Dipole_in,
    const e::Matrix<double, 3, 6>& Mob_in, const e::Matrix<double, 3, 6>& SQM_in,
    const e::Matrix<double, 3, 1>& E, const e::Matrix3d& dE, double diffusion_reduce, double electrophoretic_reduce,
    int skip, int n, double* const data_ptr) {
    /*compute the rotational matrix sequence (3 * 3) * n
    RM0_array:        rotational matrix initial state (3x3)
    EFM_in:           Electrophoretic_maxtrix_rigid_body.dat (6x3)->(3x3)
    Dipole_in：       Dipole_moment_matrix.dat * diel * rp**3 / viscosity * dt (3x3)
    Mob_in:           Mobility_matrix_rigid_body.dat-> mob; mob = mob ** 2 .* mob.sign
                      mob[1:3, 1:3] /= rp, mob[1:3, 4:6]/=rp**2, mob[4:6, 1:3]/=rp**2, mob[4:6,4:6]/=rp**3
                      (6x6)->(3x6)
    SQM_in:           real(sqrtm(mob))
    E and dE:         electric field and gradient, calculated separately. (3x1) (3x3)
    diffusion_reduce: sqrt(2*kb*T/(6*pi*viscosity) * dt)
    electrophoretic_reduce: water_permittivity*reference_zeta/viscosity/rp*dt
    n = 1024 WILL TAKE 74 Kb, equals 1.02 ms in 500 kHz sampling, skip will upsampling
    data_ptr: results ouput and flatten matrix, data_ptr points array should always there
    */
    std::random_device device;
    std::mt19937 urng{ device() };
    std::normal_distribution<double> dis{ 0, 1 };
    e::Matrix<double, 3, 6> Mob;
    e::Matrix<double, 3, 3> EFM;
    e::Matrix<double, 3, 3> Dipole;
    e::Vector3d Dipole_torque;
    e::Matrix<double, 3, 6> SQM;
    e::Matrix<double, 3, 3> RM = RM0_array;
    e::Matrix<double, 3, 3> RM_tmp = RM0_array;
    e::Matrix<double, 3, 1> increase;
    e::Matrix<double, 6, 1> Ft;
    e::Matrix<double, 6, 1> rand_number;
    // debug 
    /*std::cout << RM0_array << std::endl;
    std::cout << EFM_in << std::endl;
    std::cout << Mob_in << std::endl;
    std::cout << Dipole_in << std::endl;
    std::cout << SQM_in << std::endl;
    std::cout << E << std::endl;
    std::cout << diffusion_reduce << std::endl;*/
    int kk = 0;
    skip = (skip < 1) ? 1 : skip;
    for (int i = 0; i < n * skip; i++) {
        Mob.block<3, 3>(0, 0).noalias() = RM * Mob_in.block<3, 3>(0, 0) * RM.transpose();
        Mob.block<3, 3>(0, 3).noalias() = RM * Mob_in.block<3, 3>(0, 3) * RM.transpose();
        SQM.block<3, 3>(0, 0).noalias() = RM * SQM_in.block<3, 3>(0, 0) * RM.transpose();
        SQM.block<3, 3>(0, 3).noalias() = RM * SQM_in.block<3, 3>(0, 3) * RM.transpose();
        EFM.noalias() = RM * EFM_in * RM.transpose();
        Dipole.noalias() = RM * Dipole_in * RM.transpose();
        Dipole_torque = Dipole * E;
        Ft.block<3, 1>(0, 0).noalias() = (Dipole_torque.transpose() * dE).transpose();
        Ft.block<3, 1>(3, 0).noalias() = Dipole_torque.cross(E);
        for (int ri = 0; ri < 6; ri++) {
            rand_number(ri, 0) = dis(urng);
        }
        increase.noalias() = diffusion_reduce * (SQM * rand_number);
        increase.noalias() += electrophoretic_reduce * (EFM * E);
        increase.noalias() += Mob * Ft;
        for (int j = 0; j < 3; j++) {
            RM_tmp.col(j) = RM.col(j) + increase.cross(RM.col(j)).eval()
                - 1 / 2 * (RM.col(j) * (increase.transpose() * increase)).eval()
                - (RM.col(j).transpose() * increase) * increase;
        }
        RM = RM_tmp;
        e::JacobiSVD<e::MatrixXd> svd;
        svd.compute(RM, e::ComputeFullV | e::ComputeFullU);
        RM = svd.matrixU() * svd.matrixV().transpose();
        if (i % skip == 0) {
            for (int ii = 0; ii < 3; ii++) {
                for (int jj = 0; jj < 3; jj++) {
                    data_ptr[kk] = RM(ii, jj);
                    kk++;
                }
            }
        }
    }
    return;
}

void Threadcall_rotation(const double* RM0_array,
    const double* EFM_in, const double* Dipole_in,
    const double* Mob_in, const double* SQM_in,
    const double* E_in, const double* dE_in, const double* diffusion_reduce, const double* electrophoretic_reduce,
    const int* skip, int col, int row, double* const data_ptr) {
    for (int i = 0; i < row; i++) {
        e::Map<const e::Matrix<double, 3, 3, e::RowMajor>> RM0(RM0_array + i * 9);
        e::Map<const e::Matrix<double, 3, 3, e::RowMajor>> EFM(EFM_in + i * 9);
        e::Map<const e::Matrix<double, 3, 3, e::RowMajor>> Dipole(Dipole_in + i * 9);
        e::Map<const e::Matrix<double, 3, 6, e::RowMajor>> Mob(Mob_in + i * 18);
        e::Map<const e::Matrix<double, 3, 6, e::RowMajor>> SQM(SQM_in + i * 18);
        e::Map<const e::Matrix<double, 3, 1>, e::RowMajor> E(E_in + i * 3);
        e::Map<const e::Matrix<double, 3, 3, e::RowMajor>> dE(dE_in + i * 9);
        RotationMatrixSolver(RM0, EFM, Dipole, Mob, SQM, E, dE,
            diffusion_reduce[i], electrophoretic_reduce[i],
            skip[i], col, data_ptr + i * col * 9);
    }
}

void rotationMatrix(const double* RM0_array,
    const double* EFM_in, const double* Dipole_in,
    const double* Mob_in, const double* SQM_in,
    const double* E_in, const double* dE_in, const double* diffusion_reduce, const double* electrophoretic_reduce,
    const int* skip, int col, int row, double* const data_ptr) {
    /* input
    m, n = row, col, rwo means the batch, col means the sequence length.
    RM0_array:  m * 3 * 3
    EFM_in:  m * 3 * 3
    Dipole_in: m * 3 * 3
    Mob_in: m * 3 * 6
    SQM_in: m * 3 * 6
    E: m * 3 * 1
    dt: m * 3 * 3
    data_ptr = m * n * 9
    parallel with core number
    */
    int max_thread = std::thread::hardware_concurrency();
    max_thread = (max_thread < 1) ? 1 : max_thread;
    max_thread = (max_thread > row) ? row : max_thread;
    std::cout << max_thread << " thread!" << std::endl;
    std::vector<std::thread> threads;
    int chunk_size = row / max_thread;
    chunk_size = (chunk_size < 1) ? 1 : chunk_size;
    for (int thread_id = 0; thread_id < max_thread; thread_id++) {
        int start = thread_id * chunk_size;
        int end = (thread_id == max_thread - 1) ? row : (thread_id + 1) * chunk_size;
        threads.emplace_back(Threadcall_rotation, RM0_array + start * 9,
            EFM_in + start * 9, Dipole_in + start * 9,
            Mob_in + start * 18, SQM_in + start * 18,
            E_in + start * 3, dE_in + start * 9, diffusion_reduce + start, electrophoretic_reduce + start,
            skip + start, col, end - start, data_ptr + start * col * 9);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return;
}






