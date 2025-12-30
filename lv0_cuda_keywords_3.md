# CUDA Keywords and Concepts - Systematic Learning Guide

## Level 1: Basic Function Qualifiers

### 1. `__global__`
**Syntax:** `__global__ void kernelName(parameters)`

**Description:** Declares a kernel function that runs on the device (GPU) and is called from the host (CPU). Must return void.

**Example:**
```cuda
__global__ void addKernel(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Called from host
addKernel<<<1, 256>>>(d_a, d_b, d_c);
```

### 2. `__device__`
**Syntax:** `__device__ returnType functionName(parameters)`

**Description:** Declares a function that runs on the device and can only be called from device code (kernel or other device functions).

**Example:**
```cuda
__device__ int square(int x) {
    return x * x;
}

__global__ void computeSquares(int *input, int *output) {
    int i = threadIdx.x;
    output[i] = square(input[i]);  // Call device function
}
```

### 3. `__host__`
**Syntax:** `__host__ returnType functionName(parameters)`

**Description:** Declares a function that runs on the host (CPU). This is the default, so it's optional but can be combined with `__device__`.

**Example:**
```cuda
__host__ __device__ int multiply(int a, int b) {
    return a * b;  // Can be called from both host and device
}
```

## Level 2: Thread Identification

### 4. `threadIdx`
**Syntax:** `threadIdx.x`, `threadIdx.y`, `threadIdx.z`

**Description:** Built-in variable providing the thread index within a block (3D coordinate).

**Example:**
```cuda
__global__ void printThreadIdx() {
    printf("Thread (%d, %d, %d)\n", 
           threadIdx.x, threadIdx.y, threadIdx.z);
}
```

### 5. `blockIdx`
**Syntax:** `blockIdx.x`, `blockIdx.y`, `blockIdx.z`

**Description:** Built-in variable providing the block index within the grid (3D coordinate).

**Example:**
```cuda
__global__ void printBlockIdx() {
    printf("Block (%d, %d, %d)\n", 
           blockIdx.x, blockIdx.y, blockIdx.z);
}
```

### 6. `blockDim`
**Syntax:** `blockDim.x`, `blockDim.y`, `blockDim.z`

**Description:** Built-in variable containing the dimensions of the block (number of threads per block).

**Example:**
```cuda
__global__ void calculateGlobalIndex() {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Global index: %d\n", globalIdx);
}
```

### 7. `gridDim`
**Syntax:** `gridDim.x`, `gridDim.y`, `gridDim.z`

**Description:** Built-in variable containing the dimensions of the grid (number of blocks).

**Example:**
```cuda
__global__ void printGridInfo() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Grid dimensions: (%d, %d, %d)\n", 
               gridDim.x, gridDim.y, gridDim.z);
    }
}
```

## Level 3: Memory Space Qualifiers

### 8. `__shared__`
**Syntax:** `__shared__ type variableName[size]`

**Description:** Declares a variable in shared memory, accessible by all threads in a block. Much faster than global memory.

**Example:**
```cuda
__global__ void sharedMemoryExample(int *input, int *output) {
    __shared__ int sharedData[256];
    
    int tid = threadIdx.x;
    sharedData[tid] = input[tid];
    __syncthreads();  // Wait for all threads
    
    output[tid] = sharedData[tid] * 2;
}
```

### 9. `__constant__`
**Syntax:** `__constant__ type variableName[size]`

**Description:** Declares a variable in constant memory (read-only, cached). Must be declared at file scope.

**Example:**
```cuda
__constant__ float coefficients[5];

__global__ void useConstants(float *output) {
    int i = threadIdx.x;
    output[i] = coefficients[0] * i + coefficients[1];
}

// Copy from host
cudaMemcpyToSymbol(coefficients, h_coeff, 5 * sizeof(float));
```

### 10. `__managed__`
**Syntax:** `__managed__ type variableName`

**Description:** Declares unified memory accessible from both host and device, automatically migrated.

**Example:**
```cuda
__managed__ int managedData[1024];

__global__ void processManaged() {
    int i = threadIdx.x;
    managedData[i] *= 2;
}

int main() {
    managedData[0] = 5;  // Access from host
    processManaged<<<1, 1024>>>();
    cudaDeviceSynchronize();
    printf("%d\n", managedData[0]);  // Access from host again
}
```

## Level 4: Synchronization

### 11. `__syncthreads()`
**Syntax:** `__syncthreads()`

**Description:** Synchronizes all threads within a block. All threads must reach this point before any can proceed.

**Example:**
```cuda
__global__ void reductionSum(int *input, int *output) {
    __shared__ int temp[256];
    int tid = threadIdx.x;
    
    temp[tid] = input[tid];
    __syncthreads();  // Ensure all data is loaded
    
    // Reduction logic here
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();  // Synchronize each iteration
    }
    
    if (tid == 0) output[blockIdx.x] = temp[0];
}
```

### 12. `__syncwarp()`
**Syntax:** `__syncwarp(unsigned mask = 0xffffffff)`

**Description:** Synchronizes threads within a warp. Mask specifies which threads participate.

**Example:**
```cuda
__global__ void warpSyncExample() {
    int laneId = threadIdx.x % 32;
    int value = laneId;
    
    __syncwarp();  // Synchronize all threads in warp
    
    // Safe to use warp-level operations
    value += __shfl_xor_sync(0xffffffff, value, 1);
}
```

## Level 5: Atomic Operations

### 13. `atomicAdd()`
**Syntax:** `atomicAdd(address, val)`

**Description:** Atomically adds val to the value at address and returns the old value.

**Example:**
```cuda
__global__ void atomicAddExample(int *counter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(counter, 1);  // Safely increment shared counter
}
```

### 14. `atomicSub()`, `atomicExch()`, `atomicMin()`, `atomicMax()`
**Syntax:** `atomicOp(address, val)`

**Description:** Atomic operations for subtraction, exchange, minimum, and maximum.

**Example:**
```cuda
__global__ void atomicOperations(int *data) {
    atomicSub(&data[0], 1);      // Decrement
    atomicExch(&data[1], 42);    // Exchange with 42
    atomicMin(&data[2], 10);     // Keep minimum
    atomicMax(&data[3], 100);    // Keep maximum
}
```

### 15. `atomicCAS()`
**Syntax:** `atomicCAS(address, compare, val)`

**Description:** Compare-and-swap: if *address == compare, set *address = val. Returns old value.

**Example:**
```cuda
__global__ void lockExample(int *lock, int *sharedResource) {
    // Spin until we acquire the lock
    while (atomicCAS(lock, 0, 1) != 0);
    
    // Critical section
    *sharedResource += 1;
    
    // Release lock
    atomicExch(lock, 0);
}
```

## Level 6: Warp-Level Primitives

### 16. `__shfl_sync()`
**Syntax:** `__shfl_sync(mask, var, srcLane, width=32)`

**Description:** Shuffle data across threads in a warp. Each thread gets value from specified source lane.

**Example:**
```cuda
__global__ void shuffleExample() {
    int laneId = threadIdx.x % 32;
    int value = laneId * 10;
    
    // Each thread gets value from lane 0
    int result = __shfl_sync(0xffffffff, value, 0);
    printf("Lane %d: got %d from lane 0\n", laneId, result);
}
```

### 17. `__shfl_up_sync()`, `__shfl_down_sync()`, `__shfl_xor_sync()`
**Syntax:** `__shfl_up_sync(mask, var, delta)`, etc.

**Description:** Shuffle variants for different communication patterns within a warp.

**Example:**
```cuda
__global__ void warpReduction() {
    int laneId = threadIdx.x % 32;
    int value = laneId + 1;
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    
    if (laneId == 0) printf("Sum: %d\n", value);
}
```

### 18. `__ballot_sync()`
**Syntax:** `__ballot_sync(mask, predicate)`

**Description:** Returns a bitmask where each bit represents whether the predicate is true for that thread.

**Example:**
```cuda
__global__ void ballotExample(int *data) {
    int tid = threadIdx.x;
    int value = data[tid];
    
    // Get mask of threads where value > 50
    unsigned mask = __ballot_sync(0xffffffff, value > 50);
    
    if (tid == 0) printf("Threads with value > 50: %x\n", mask);
}
```

### 19. `__any_sync()`, `__all_sync()`
**Syntax:** `__any_sync(mask, predicate)`, `__all_sync(mask, predicate)`

**Description:** Returns true if any/all threads in the mask have predicate true.

**Example:**
```cuda
__global__ void convergenceCheck(int *data) {
    int tid = threadIdx.x;
    bool converged = (data[tid] < 0.001);
    
    if (__all_sync(0xffffffff, converged)) {
        if (tid == 0) printf("All threads converged!\n");
    }
}
```

## Level 7: Memory Fence Operations

### 20. `__threadfence()`
**Syntax:** `__threadfence()`

**Description:** Ensures all writes to global/shared memory are visible to all threads in the device before proceeding.

**Example:**
```cuda
__global__ void producerConsumer(int *data, int *flag) {
    if (threadIdx.x == 0) {
        data[0] = 42;           // Produce data
        __threadfence();        // Ensure write is visible
        atomicExch(flag, 1);    // Signal data is ready
    } else {
        while (atomicAdd(flag, 0) == 0);  // Wait for signal
        __threadfence();        // Ensure we see the write
        printf("Consumed: %d\n", data[0]);
    }
}
```

### 21. `__threadfence_block()`
**Syntax:** `__threadfence_block()`

**Description:** Memory fence for threads within the same block only.

**Example:**
```cuda
__global__ void blockFenceExample() {
    __shared__ int sharedData[256];
    int tid = threadIdx.x;
    
    sharedData[tid] = tid;
    __threadfence_block();  // Ensure writes are visible in block
    __syncthreads();
}
```

### 22. `__threadfence_system()`
**Syntax:** `__threadfence_system()`

**Description:** Memory fence for all threads in the system (including host for unified memory).

**Example:**
```cuda
__global__ void systemFence(__managed__ int *data) {
    data[threadIdx.x] = threadIdx.x;
    __threadfence_system();  // Ensure visible to host
}
```

## Level 8: Cooperative Groups

### 23. `cooperative_groups::this_thread_block()`
**Syntax:** `auto block = cooperative_groups::this_thread_block();`

**Description:** Creates a thread block group for more flexible synchronization patterns.

**Example:**
```cuda
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void cooperativeExample() {
    auto block = this_thread_block();
    
    block.sync();  // Equivalent to __syncthreads()
    
    if (block.thread_rank() == 0) {
        printf("Block size: %d\n", block.size());
    }
}
```

### 24. `cooperative_groups::tiled_partition()`
**Syntax:** `auto tile = tiled_partition<Size>(parent_group);`

**Description:** Partitions a group into tiles of specified size for sub-group operations.

**Example:**
```cuda
__global__ void tiledReduction(int *data) {
    auto block = cooperative_groups::this_thread_block();
    auto tile = tiled_partition<32>(block);  // Warp-sized tiles
    
    int value = data[threadIdx.x];
    
    // Reduction within tile
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        value += tile.shfl_down(value, offset);
    }
    
    if (tile.thread_rank() == 0) {
        data[blockIdx.x * blockDim.x / 32 + tile.meta_group_rank()] = value;
    }
}
```

## Level 9: Dynamic Parallelism

### 25. `__global__` (nested kernel launch)
**Syntax:** `kernelName<<<grid, block>>>(params);` from device code

**Description:** Launch kernels from within other kernels (requires compute capability 3.5+).

**Example:**
```cuda
__global__ void childKernel(int depth) {
    printf("Child at depth %d, thread %d\n", depth, threadIdx.x);
}

__global__ void parentKernel(int depth) {
    printf("Parent at depth %d, thread %d\n", depth, threadIdx.x);
    
    if (depth < 3) {
        childKernel<<<1, 4>>>(depth + 1);
        cudaDeviceSynchronize();  // Wait for child
    }
}
```

## Level 10: Texture and Surface Memory

### 26. `texture<>` and `cudaTextureObject_t`
**Syntax:** `texture<Type, Dim> texRef;` (legacy) or `cudaTextureObject_t`

**Description:** Texture memory provides cached, filtered access to global memory with spatial locality benefits.

**Example:**
```cuda
__global__ void textureKernel(cudaTextureObject_t tex, float *output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Read from texture with hardware interpolation
    float value = tex2D<float>(tex, x, y);
    output[y * gridDim.x * blockDim.x + x] = value;
}
```

### 27. `surface<>`
**Syntax:** `surface<void, cudaSurfaceType> surfRef;`

**Description:** Surface memory allows reading and writing to 2D/3D arrays with special hardware support.

**Example:**
```cuda
__global__ void surfaceWrite(cudaSurfaceObject_t surf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float data = x + y;
    surf2Dwrite(data, surf, x * sizeof(float), y);
}
```

## Level 11: Advanced Function Modifiers

### 28. `__noinline__`
**Syntax:** `__noinline__ __device__ void func()`

**Description:** Prevents the compiler from inlining the function.

**Example:**
```cuda
__noinline__ __device__ int complexCalculation(int x) {
    // Force this to be a separate function call
    return x * x * x + 2 * x * x + x + 1;
}
```

### 29. `__forceinline__`
**Syntax:** `__forceinline__ __device__ void func()`

**Description:** Forces the compiler to inline the function.

**Example:**
```cuda
__forceinline__ __device__ int add(int a, int b) {
    return a + b;  // Always inlined for performance
}
```

### 30. `__launch_bounds__()`
**Syntax:** `__global__ void __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM) kernel()`

**Description:** Provides hints to compiler about kernel launch configuration for optimization.

**Example:**
```cuda
__global__ 
__launch_bounds__(256, 4)  // Max 256 threads, min 4 blocks per SM
void optimizedKernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}
```

## Summary

This guide covers CUDA keywords from basic kernel launches to advanced optimization techniques. The progression follows:

1. **Basics**: Function qualifiers and kernel launches
2. **Thread Management**: Understanding thread/block indexing
3. **Memory**: Different memory spaces and their characteristics
4. **Synchronization**: Coordinating threads safely
5. **Atomics**: Thread-safe operations on shared data
6. **Warp-Level**: Efficient intra-warp communication
7. **Memory Ordering**: Ensuring visibility of memory operations
8. **Cooperative Groups**: Modern synchronization abstractions
9. **Dynamic Parallelism**: Kernels launching kernels
10. **Texture/Surface**: Specialized memory for spatial data
11. **Optimization**: Fine-tuning compiler behavior

Each level builds on previous concepts, providing a structured learning path for CUDA programming.
