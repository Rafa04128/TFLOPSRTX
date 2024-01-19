#include <hip/hip_runtime.h>

__global__ void gemm_kernel(float16 *A, float16 *B, float16 *C, int N) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(row < N && col < N) {
        float16 sum = 0.0;
        for(int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
