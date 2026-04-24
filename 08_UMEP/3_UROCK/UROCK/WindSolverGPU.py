#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated (PyCUDA) Jacobi / weighted-Jacobi lambda solver.

Key points:
- Jacobi update runs on GPU (one thread per cell in cells4Solver).
- Main iteration loop runs entirely on GPU; no per-iteration prints.
- Convergence (eps) checked every 100 iterations.
- eps reduction is done on GPU (multi-stage); only two scalars are copied to CPU.
- Iterations executed in batches of 1000 kernel launches (still launches, but no
  full-field CPU transfers between them).
"""

import numpy as np
import time
import pandas as pd
import subprocess
import os

from .GlobalVariables import MAX_ITERATIONS, THRESHOLD_ITERATIONS, DESCENDING_Y, OUTPUT_DIRECTORY

try:
    import pycuda.autoinit  # initializes CUDA context
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError as e:
    raise ImportError(
        "PyCUDA is required for GPU acceleration. Install pycuda and ensure "
        "CUDA toolkit + driver are available."
    ) from e


CUDA_SRC = r"""
extern "C" {

__global__ void jacobi_update_cells(
    const int* __restrict__ cells,
    int nCells,
    const float* __restrict__ lambdaOld,
    float* __restrict__ lambdaNew,

    const float* __restrict__ u0,
    const float* __restrict__ v0,
    const float* __restrict__ w0,

    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float alpha1,
    float A, float B,
    float omegaWJ,
    int descendingY,

    const float* __restrict__ e,
    const float* __restrict__ f,
    const float* __restrict__ g,
    const float* __restrict__ h,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ o,
    const float* __restrict__ p,
    const float* __restrict__ q
){
    int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= nCells) return;

    int base = 3 * t;
    int i = cells[base + 0];
    int j = cells[base + 1];
    int k = cells[base + 2];

    int idx  = (i * ny + j) * nz + k;

    int idx_im1 = ((i - 1) * ny + j) * nz + k;
    int idx_ip1 = ((i + 1) * ny + j) * nz + k;
    int idx_jm1 = (i * ny + (j - 1)) * nz + k;
    int idx_jp1 = (i * ny + (j + 1)) * nz + k;
    int idx_km1 = (i * ny + j) * nz + (k - 1);
    int idx_kp1 = (i * ny + j) * nz + (k + 1);

    float dudx, dvdy, dwdz;

    if (descendingY) {
        dudx = (u0[idx] - u0[idx_ip1]) / dx;
        dvdy = (v0[idx] - v0[idx_jp1]) / dy;
        dwdz = (w0[idx] - w0[idx_kp1]) / dz;

        float neigh =
            e[idx] * lambdaOld[idx_im1] +
            f[idx] * lambdaOld[idx_ip1] +
            A * (g[idx] * lambdaOld[idx_jm1] + h[idx] * lambdaOld[idx_jp1]) +
            B * (m[idx] * lambdaOld[idx_km1] + n[idx] * lambdaOld[idx_kp1]);

        float denom = 2.0f * (o[idx] + A * p[idx] + B * q[idx]);
        float rhs = (-1.0f) * (dx * dx * (-2.0f * alpha1 * alpha1) * (dudx + dvdy + dwdz));

        float lambdaJac = (rhs + neigh) / denom;
        lambdaNew[idx] = (1.0f - omegaWJ) * lambdaOld[idx] + omegaWJ * lambdaJac;
    } else {
        dudx = (u0[idx_ip1] - u0[idx]) / dx;
        dvdy = (v0[idx_jp1] - v0[idx]) / dy;
        dwdz = (w0[idx_kp1] - w0[idx]) / dz;

        float neigh =
            e[idx] * lambdaOld[idx_ip1] +
            f[idx] * lambdaOld[idx_im1] +
            A * (g[idx] * lambdaOld[idx_jp1] + h[idx] * lambdaOld[idx_jm1]) +
            B * (m[idx] * lambdaOld[idx_kp1] + n[idx] * lambdaOld[idx_km1]);

        float denom = 2.0f * (o[idx] + A * p[idx] + B * q[idx]);
        float rhs = (-1.0f) * (dx * dx * (-2.0f * alpha1 * alpha1) * (dudx + dvdy + dwdz));

        float lambdaJac = (rhs + neigh) / denom;
        lambdaNew[idx] = (1.0f - omegaWJ) * lambdaOld[idx] + omegaWJ * lambdaJac;
    }
}


// per-block partial sums over cells:
// num = sum(|curr - prev|), den = sum(|curr|)
__global__ void eps_partials_cells(
    const int* __restrict__ cells,
    int nCells,
    const float* __restrict__ lambdaPrev,
    const float* __restrict__ lambdaCurr,
    int nx, int ny, int nz,
    float* __restrict__ block_sums_num,
    float* __restrict__ block_sums_den
){
    extern __shared__ float sh[];
    float* sh_num = sh;
    float* sh_den = sh + blockDim.x;

    int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    float local_num = 0.0f;
    float local_den = 0.0f;

    if (t < nCells) {
        int base = 3 * t;
        int i = cells[base + 0];
        int j = cells[base + 1];
        int k = cells[base + 2];
        int idx = (i * ny + j) * nz + k;

        float curr = lambdaCurr[idx];
        float prev = lambdaPrev[idx];

        float diff = curr - prev;
        local_num = diff < 0.0f ? -diff : diff;
        local_den = curr < 0.0f ? -curr : curr;
    }

    sh_num[threadIdx.x] = local_num;
    sh_den[threadIdx.x] = local_den;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh_num[threadIdx.x] += sh_num[threadIdx.x + stride];
            sh_den[threadIdx.x] += sh_den[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_sums_num[blockIdx.x] = sh_num[0];
        block_sums_den[blockIdx.x] = sh_den[0];
    }
}


// generic reduce sum: out[block] = sum(in block)
__global__ void reduce_sum_f32(
    const float* __restrict__ in_arr,
    float* __restrict__ out_arr,
    int n
){
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int i = (int)(blockIdx.x * blockDim.x + tid);

    float x = 0.0f;
    if (i < n) x = in_arr[i];
    sh[tid] = x;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out_arr[blockIdx.x] = sh[0];
}

} // extern "C"
"""


def _to_f32_c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


def _to_i32_c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.int32)


def _gpu_reduce_sum(mod, d_in, n, threads=256):
    """
    Reduce device float array d_in (length n) to a single float value.

    IMPORTANT FIX:
    - pycuda.driver.DeviceAllocation has no `.size`.
    - We track buffer sizes ourselves via `nbytes` variables.
    """
    reduce_kernel = mod.get_function("reduce_sum_f32")

    if n <= 0:
        return 0.0

    # stage 1
    blocks = (n + threads - 1) // threads
    d_tmp1_nbytes = blocks * np.dtype(np.float32).itemsize
    d_tmp1 = cuda.mem_alloc(d_tmp1_nbytes)

    reduce_kernel(
        d_in,
        d_tmp1,
        np.int32(n),
        block=(threads, 1, 1),
        grid=(blocks, 1, 1),
        shared=threads * np.dtype(np.float32).itemsize
    )

    curr_n = blocks
    d_curr = d_tmp1

    # second buffer (ping-pong)
    next_blocks = (curr_n + threads - 1) // threads
    d_tmp2_nbytes = max(1, next_blocks) * np.dtype(np.float32).itemsize
    d_tmp2 = cuda.mem_alloc(d_tmp2_nbytes)

    d_next = d_tmp2
    d_next_nbytes = d_tmp2_nbytes

    # reduce until 1 element
    while curr_n > 1:
        blocks = (curr_n + threads - 1) // threads
        needed_bytes = blocks * np.dtype(np.float32).itemsize

        if d_next_nbytes < needed_bytes:
            d_next.free()
            d_next = cuda.mem_alloc(needed_bytes)
            d_next_nbytes = needed_bytes

        reduce_kernel(
            d_curr,
            d_next,
            np.int32(curr_n),
            block=(threads, 1, 1),
            grid=(blocks, 1, 1),
            shared=threads * np.dtype(np.float32).itemsize
        )

        curr_n = blocks
        d_curr, d_next = d_next, d_curr
        # d_next_nbytes remains the capacity of the allocation referenced by d_next,
        # but after swapping pointers we should swap capacities too.
        # easiest: recompute from current allocations we control:
        # (track both capacities explicitly)
        # We'll keep explicit tracking:
        # If swap happened, swap nbytes trackers too:
        d_tmp1_nbytes, d_next_nbytes = d_next_nbytes, d_tmp1_nbytes

    host_val = np.empty((1,), dtype=np.float32)
    cuda.memcpy_dtoh(host_val, d_curr)
    return float(host_val[0])

def _start_gpu_logger(log_path=os.path.join(OUTPUT_DIRECTORY, "gpu_log.csv"), interval_s=1):
    log_file = open(log_path, "w")
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv",
        "-l", str(interval_s)
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.DEVNULL
    )
    return proc, log_file


def _stop_gpu_logger(proc, log_file):
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if log_file is not None:
        log_file.close()


def solver(
    x, y, z, dx, dy, dz,
    u0, v0, w0,
    buildingCoordinates, cells4Solver, cursor,
    maxIterations=MAX_ITERATIONS,
    thresholdIterations=THRESHOLD_ITERATIONS,
    feedback=None
):
    timeStartCalculation = time.time()

    nx = x.size
    ny = y.size
    nz = z.size

    # output wind vectors
    u = u0.copy()
    v = v0.copy()
    w = w0.copy()

    # lambda init (CPU)
    lambdaN = np.ones((nx, ny, nz), dtype=np.float32)
    lambdaN1 = np.ones((nx, ny, nz), dtype=np.float32)
    for arr in (lambdaN, lambdaN1):
        arr[0, :, :] = 0.0
        arr[:, 0, :] = 0.0
        arr[:, :, 0] = 0.0
        arr[-1, :, :] = 0.0
        arr[:, -1, :] = 0.0
        arr[:, :, -1] = 0.0

    # omega from original
    Xi = ((np.cos(np.pi / nx) + (dx / dy) ** 2 * np.cos(np.pi / ny)) / (1 + (dx / dy) ** 2)) ** 2
    omega = 2.0 * ((1.0 - np.sqrt(1.0 - Xi)) / Xi)
    if (omega < 1.0) or (omega > 2.0):
        omega = 1.78

    # Weighted Jacobi relaxation: <= 1.0
    omegaWJ = 1.0

    alpha1 = 1.0
    alpha2 = 1.0
    eta = alpha1 / alpha2
    A = (dx ** 2) / (dy ** 2)
    B = (eta ** 2) * (dx ** 2) / (dz ** 2)

    # obstacle coefficients
    e = np.ones((nx, ny, nz), dtype=np.float32)
    f = np.ones((nx, ny, nz), dtype=np.float32)
    g = np.ones((nx, ny, nz), dtype=np.float32)
    h = np.ones((nx, ny, nz), dtype=np.float32)
    m = np.ones((nx, ny, nz), dtype=np.float32)
    n = np.ones((nx, ny, nz), dtype=np.float32)
    o = np.ones((nx, ny, nz), dtype=np.float32)
    p = np.ones((nx, ny, nz), dtype=np.float32)
    q = np.ones((nx, ny, nz), dtype=np.float32)

    indBelow = pd.MultiIndex.from_tuples(list(zip(*[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2] + 1])))
    indAbove = pd.MultiIndex.from_tuples(list(zip(*[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2] - 1])))
    indFront = pd.MultiIndex.from_tuples(list(zip(*[buildingCoordinates[0], buildingCoordinates[1] - 1, buildingCoordinates[2]])))
    indBehind = pd.MultiIndex.from_tuples(list(zip(*[buildingCoordinates[0], buildingCoordinates[1] + 1, buildingCoordinates[2]])))
    indLeft = pd.MultiIndex.from_tuples(list(zip(*[buildingCoordinates[0] + 1, buildingCoordinates[1], buildingCoordinates[2]])))
    indRight = pd.MultiIndex.from_tuples(list(zip(*[buildingCoordinates[0] - 1, buildingCoordinates[1], buildingCoordinates[2]])))

    indBelowRight = indBelow.intersection(indRight)
    indBelowLeft = indBelow.intersection(indLeft)
    indBelowFront = indBelow.intersection(indFront)
    indBelowBehind = indBelow.intersection(indBehind)

    indE = indBelowRight.union(indRight)
    indF = indBelowLeft.union(indLeft)
    indG = indBelowFront.union(indFront)
    indH = indBelowBehind.union(indBehind)
    indM = indAbove
    indN = indBelow.union(indBelowFront).union(indBelowLeft).union(indBelowRight).union(indBelowBehind)
    indO = indRight.union(indLeft).union(indBelowLeft).union(indBelowRight)
    indP = indBehind.union(indFront).union(indBelowBehind).union(indBelowFront)
    indQ = indBelow.union(indAbove).union(indBelowFront).union(indBelowLeft).union(indBelowRight).union(indBelowBehind)

    if DESCENDING_Y:
        e[indF.get_level_values(0), indF.get_level_values(1), indF.get_level_values(2)] = 0.0
        f[indE.get_level_values(0), indE.get_level_values(1), indE.get_level_values(2)] = 0.0
        g[indH.get_level_values(0), indH.get_level_values(1), indH.get_level_values(2)] = 0.0
        h[indG.get_level_values(0), indG.get_level_values(1), indG.get_level_values(2)] = 0.0
        m[indM.get_level_values(0), indM.get_level_values(1), indM.get_level_values(2)] = 0.0
        n[indN.get_level_values(0), indN.get_level_values(1), indN.get_level_values(2)] = 0.0
    else:
        e[indE.get_level_values(0), indE.get_level_values(1), indE.get_level_values(2)] = 0.0
        f[indF.get_level_values(0), indF.get_level_values(1), indF.get_level_values(2)] = 0.0
        g[indG.get_level_values(0), indG.get_level_values(1), indG.get_level_values(2)] = 0.0
        h[indH.get_level_values(0), indH.get_level_values(1), indH.get_level_values(2)] = 0.0
        m[indM.get_level_values(0), indM.get_level_values(1), indM.get_level_values(2)] = 0.0
        n[indN.get_level_values(0), indN.get_level_values(1), indN.get_level_values(2)] = 0.0

    o[indO.get_level_values(0), indO.get_level_values(1), indO.get_level_values(2)] = 0.5
    p[indP.get_level_values(0), indP.get_level_values(1), indP.get_level_values(2)] = 0.5
    q[indQ.get_level_values(0), indQ.get_level_values(1), indQ.get_level_values(2)] = 0.5

    # cells list
    cells4Solver = np.asarray(cells4Solver)
    if cells4Solver.ndim != 2 or cells4Solver.shape[1] != 3:
        raise ValueError("cells4Solver must be shape (nCells, 3) containing (i, j, k).")
    cells_i32 = _to_i32_c(cells4Solver)
    nCells = int(cells_i32.shape[0])

    # Start GPU usage log
    try:
        gpu_log_proc, gpu_log_file = _start_gpu_logger("gpu_log.csv", interval_s=1)

        # compile kernels
        mod = SourceModule(CUDA_SRC, options=["--use_fast_math"])
        jacobi_kernel = mod.get_function("jacobi_update_cells")
        eps_partials_kernel = mod.get_function("eps_partials_cells")

        # upload once (flattened)
        u0_f = _to_f32_c(u0).ravel()
        v0_f = _to_f32_c(v0).ravel()
        w0_f = _to_f32_c(w0).ravel()

        e_f = _to_f32_c(e).ravel()
        f_f = _to_f32_c(f).ravel()
        g_f = _to_f32_c(g).ravel()
        h_f = _to_f32_c(h).ravel()
        m_f = _to_f32_c(m).ravel()
        n_f = _to_f32_c(n).ravel()
        o_f = _to_f32_c(o).ravel()
        p_f = _to_f32_c(p).ravel()
        q_f = _to_f32_c(q).ravel()

        lambdaOld_f = _to_f32_c(lambdaN).ravel()
        lambdaNew_f = _to_f32_c(lambdaN1).ravel()

        d_cells = cuda.mem_alloc(cells_i32.nbytes); cuda.memcpy_htod(d_cells, cells_i32)

        d_u0 = cuda.mem_alloc(u0_f.nbytes); cuda.memcpy_htod(d_u0, u0_f)
        d_v0 = cuda.mem_alloc(v0_f.nbytes); cuda.memcpy_htod(d_v0, v0_f)
        d_w0 = cuda.mem_alloc(w0_f.nbytes); cuda.memcpy_htod(d_w0, w0_f)

        d_e = cuda.mem_alloc(e_f.nbytes); cuda.memcpy_htod(d_e, e_f)
        d_f = cuda.mem_alloc(f_f.nbytes); cuda.memcpy_htod(d_f, f_f)
        d_g = cuda.mem_alloc(g_f.nbytes); cuda.memcpy_htod(d_g, g_f)
        d_h = cuda.mem_alloc(h_f.nbytes); cuda.memcpy_htod(d_h, h_f)
        d_m = cuda.mem_alloc(m_f.nbytes); cuda.memcpy_htod(d_m, m_f)
        d_n = cuda.mem_alloc(n_f.nbytes); cuda.memcpy_htod(d_n, n_f)
        d_o = cuda.mem_alloc(o_f.nbytes); cuda.memcpy_htod(d_o, o_f)
        d_p = cuda.mem_alloc(p_f.nbytes); cuda.memcpy_htod(d_p, p_f)
        d_q = cuda.mem_alloc(q_f.nbytes); cuda.memcpy_htod(d_q, q_f)

        d_lambdaOld = cuda.mem_alloc(lambdaOld_f.nbytes); cuda.memcpy_htod(d_lambdaOld, lambdaOld_f)
        d_lambdaNew = cuda.mem_alloc(lambdaNew_f.nbytes); cuda.memcpy_htod(d_lambdaNew, lambdaNew_f)

        THREADS = 256
        blocks_cells = (nCells + THREADS - 1) // THREADS
        eps_shared_bytes = 2 * THREADS * np.dtype(np.float32).itemsize

        d_partials_num = cuda.mem_alloc(blocks_cells * np.dtype(np.float32).itemsize)
        d_partials_den = cuda.mem_alloc(blocks_cells * np.dtype(np.float32).itemsize)

        # batching
        BATCH = 1000
        CHECK_EVERY = 1000

        it = 0
        stopped = False
        eps = np.inf
        
        start_time = time.perf_counter() # Time main calculation loop
        while it < maxIterations and not stopped:
            batch_end = min(it + BATCH, maxIterations)

            while it < batch_end:
                jacobi_kernel(
                    d_cells,
                    np.int32(nCells),
                    d_lambdaOld,
                    d_lambdaNew,
                    d_u0, d_v0, d_w0,
                    np.int32(nx), np.int32(ny), np.int32(nz),
                    np.float32(dx), np.float32(dy), np.float32(dz),
                    np.float32(alpha1),
                    np.float32(A), np.float32(B),
                    np.float32(omegaWJ),
                    np.int32(1 if DESCENDING_Y else 0),
                    d_e, d_f, d_g, d_h, d_m, d_n, d_o, d_p, d_q,
                    block=(THREADS, 1, 1),
                    grid=(blocks_cells, 1, 1)
                )

                # swap: d_lambdaOld is always "current"
                d_lambdaOld, d_lambdaNew = d_lambdaNew, d_lambdaOld
                it += 1

                if (it % CHECK_EVERY) == 0:
                    eps_partials_kernel(
                        d_cells,
                        np.int32(nCells),
                        d_lambdaNew,  # prev
                        d_lambdaOld,  # curr
                        np.int32(nx), np.int32(ny), np.int32(nz),
                        d_partials_num,
                        d_partials_den,
                        block=(THREADS, 1, 1),
                        grid=(blocks_cells, 1, 1),
                        shared=eps_shared_bytes
                    )

                    num = _gpu_reduce_sum(mod, d_partials_num, blocks_cells, threads=THREADS)
                    den = _gpu_reduce_sum(mod, d_partials_den, blocks_cells, threads=THREADS)
                    eps = (num / den) if den != 0.0 else np.inf

                    if eps < thresholdIterations:
                        stopped = True
                        break


        # copy final lambda once
        lambda_final_flat = np.empty((nx * ny * nz,), dtype=np.float32)
        cuda.memcpy_dtoh(lambda_final_flat, d_lambdaOld)
        lambdaN1 = lambda_final_flat.reshape((nx, ny, nz))
        end_time = time.perf_counter() # Time main calculation loop
        print(f"Wind solver main calculation total iterations: {it}")
        print(f"Wind solver main calculation time: {end_time-start_time}")
        
        # Stop GPU usage log
    finally:
        _stop_gpu_logger(gpu_log_proc, gpu_log_file)


    # final wind speed (CPU) as original
    if DESCENDING_Y:
        u[1:nx, :, :] = u0[1:nx, :, :] + 0.5 * (
            1.0 / (alpha1 ** 2)) * (lambdaN1[0:nx - 1, :, :] - lambdaN1[1:nx, :, :]) / dx
        v[:, 1:ny, :] = v0[:, 1:ny, :] + 0.5 * (
            1.0 / (alpha1 ** 2)) * (lambdaN1[:, 0:ny - 1, :] - lambdaN1[:, 1:ny, :]) / dy
        w[:, :, 1:nz] = w0[:, :, 1:nz] + 0.5 * (
            1.0 / (alpha2 ** 2)) * (lambdaN1[:, :, 0:nz - 1] - lambdaN1[:, :, 1:nz]) / dz
    else:
        u[1:nx, :, :] = u0[1:nx, :, :] + 0.5 * (
            1.0 / (alpha1 ** 2)) * (lambdaN1[1:nx, :, :] - lambdaN1[0:nx - 1, :, :]) / dx
        v[:, 1:ny, :] = v0[:, 1:ny, :] + 0.5 * (
            1.0 / (alpha1 ** 2)) * (lambdaN1[:, 1:ny, :] - lambdaN1[:, 0:ny - 1, :]) / dy
        w[:, :, 1:nz] = w0[:, :, 1:nz] + 0.5 * (
            1.0 / (alpha2 ** 2)) * (lambdaN1[:, :, 1:nz] - lambdaN1[:, :, 0:nz - 1]) / dz

    # zero wind in buildings
    u[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    u[buildingCoordinates[0] + 1, buildingCoordinates[1], buildingCoordinates[2]] = 0
    v[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    v[buildingCoordinates[0], buildingCoordinates[1] + 1, buildingCoordinates[2]] = 0
    w[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    w[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2] + 1] = 0

    print("Final wind speeds calculated")

    return u, v, w