#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#ifndef BLOCK
#define BLOCK 32
#endif

#define CUDA_OK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while(0)

static void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i=0;i<M;++i)
    for (int j=0;j<N;++j) {
      float s = 0.f;
      for (int k=0;k<K;++k) s += A[i*K + k] * B[k*N + j];
      C[i*N + j] = s;
    }
}

__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float s = 0.f;
  for (int k=0;k<K;++k) s += A[row*K + k] * B[k*N + col];
  C[row*N + col] = s;
}

__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
  __shared__ float As[BLOCK][BLOCK];
  __shared__ float Bs[BLOCK][BLOCK];

  int row = blockIdx.y * BLOCK + threadIdx.y;
  int col = blockIdx.x * BLOCK + threadIdx.x;

  float acc = 0.f;
  for (int t = 0; t < (K + BLOCK - 1)/BLOCK; ++t) {
    int a_col = t*BLOCK + threadIdx.x;
    int b_row = t*BLOCK + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row*K + a_col] : 0.f;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row*N + col] : 0.f;
    __syncthreads();

    #pragma unroll 16
    for (int k=0;k<BLOCK;++k)
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row*N + col] = acc;
}

static float randf() { return (float)rand() / (float)RAND_MAX - 0.5f; }

int main(int argc, char** argv) {
  int M = (argc > 1) ? atoi(argv[1]) : 1024;
  int K = (argc > 2) ? atoi(argv[2]) : M;
  int N = (argc > 3) ? atoi(argv[3]) : M;
  bool skip_cpu = (argc > 4) ? atoi(argv[4]) : 0; // pass 1 to skip CPU ref on big sizes

  printf("GEMM: %d x %d  *  %d x %d  ->  %d x %d  (BLOCK=%d)\n", M, K, K, N, M, N, BLOCK);

  size_t szA = (size_t)M*K, szB = (size_t)K*N, szC = (size_t)M*N;
  float *A = (float*)malloc(szA*sizeof(float));
  float *B = (float*)malloc(szB*sizeof(float));
  float *C = (float*)malloc(szC*sizeof(float));
  float *Cref = skip_cpu ? nullptr : (float*)malloc(szC*sizeof(float));

  srand(0);
  for (size_t i=0;i<szA;++i) A[i]=randf();
  for (size_t i=0;i<szB;++i) B[i]=randf();

  if (!skip_cpu) {
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_gemm(A,B,Cref,M,N,K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    double flops = 2.0*(double)M*(double)N*(double)K;
    printf("CPU time: %.3f ms, %.2f GFLOP/s\n", ms, flops/(ms*1e6));
  }

  float *dA,*dB,*dC;
  CUDA_OK(cudaMalloc(&dA, szA*sizeof(float)));
  CUDA_OK(cudaMalloc(&dB, szB*sizeof(float)));
  CUDA_OK(cudaMalloc(&dC, szC*sizeof(float)));
  CUDA_OK(cudaMemcpy(dA, A, szA*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dB, B, szB*sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(BLOCK,BLOCK);
  dim3 grid((N+BLOCK-1)/BLOCK, (M+BLOCK-1)/BLOCK);

  // Warmup
  gemm_tiled<<<grid,block>>>(dA,dB,dC,M,N,K);
  CUDA_OK(cudaDeviceSynchronize());

  // Measure naive
  cudaEvent_t s1,e1; CUDA_OK(cudaEventCreate(&s1)); CUDA_OK(cudaEventCreate(&e1));
  CUDA_OK(cudaEventRecord(s1));
  gemm_naive<<<grid,block>>>(dA,dB,dC,M,N,K);
  CUDA_OK(cudaEventRecord(e1));
  CUDA_OK(cudaEventSynchronize(e1));
  printf("Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

  float naive_ms=0; CUDA_OK(cudaEventElapsedTime(&naive_ms,s1,e1));

  // Measure tiled
  cudaEvent_t s2,e2; CUDA_OK(cudaEventCreate(&s2)); CUDA_OK(cudaEventCreate(&e2));
  CUDA_OK(cudaEventRecord(s2));
  gemm_tiled<<<grid,block>>>(dA,dB,dC,M,N,K);
  CUDA_OK(cudaEventRecord(e2));
  CUDA_OK(cudaEventSynchronize(e2));
  float tiled_ms=0; CUDA_OK(cudaEventElapsedTime(&tiled_ms,s2,e2));

  CUDA_OK(cudaMemcpy(C, dC, szC*sizeof(float), cudaMemcpyDeviceToHost));

  if (!skip_cpu) {
    double max_err = 0.0, rms = 0.0;
    for (size_t i=0;i<szC;++i) {
      double diff = (double)C[i] - (double)Cref[i];
      max_err = fmax(max_err, fabs(diff));
      rms += diff*diff;
    }
    rms = sqrt(rms / (double)szC);
    printf("Verify: max|Δ|=%.3e  RMS=%.3e\n", max_err, rms);
  }

  double flops = 2.0*(double)M*(double)N*(double)K;
  printf("Naive GPU:  %.3f ms, %.2f GFLOP/s\n", naive_ms, flops/(naive_ms*1e6));
  printf("Tiled GPU:  %.3f ms, %.2f GFLOP/s\n", tiled_ms, flops/(tiled_ms*1e6));

  CUDA_OK(cudaFree(dA)); CUDA_OK(cudaFree(dB)); CUDA_OK(cudaFree(dC));
  free(A); free(B); free(C); if (Cref) free(Cref);
  return 0;
}
