# CUDA Thread Synchronization: From Basics to Advanced

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding CUDA's Execution Model](#understanding-cudas-execution-model)
3. [Thread Synchronization Within a Block](#thread-synchronization-within-a-block)
4. [Cross-Block Synchronization](#cross-block-synchronization)
5. [Warp-Level Synchronization](#warp-level-synchronization)
6. [Advanced Topics](#advanced-topics)
7. [Best Practices](#best-practices)

---

## Introduction

Thread synchronization in CUDA is crucial for coordinating parallel work and ensuring data consistency. However, CUDA's hierarchical execution model means synchronization works differently at different levels.

**Key Principle:** CUDA provides strong synchronization guarantees within blocks, but weak guarantees across blocks.

---

## Understanding CUDA's Execution Model

Before diving into synchronization, let's understand the hierarchy:

```
Grid (Kernel Launch)
├── Block 0
│   ├── Warp 0 (Threads 0-31)
│   ├── Warp 1 (Threads 32-63)
│   └── ...
├── Block 1
│   ├── Warp 0
│   └── ...
└── ...
```

**Important Facts:**
- Threads are organized into blocks
- Blocks are scheduled independently on Streaming Multiprocessors (SMs)
- Within a block, threads execute in groups of 32 called warps
- No guarantee about block execution order or concurrency

---

## Thread Synchronization Within a Block

### 1. `__syncthreads()` - The Basic Barrier

The most common synchronization primitive. Creates a barrier where all threads in a block must arrive before any can proceed.

```cuda
__global__ void sharedMemoryExample(float *input, float *output, int N) {
    __shared__ float shared[256];  // Shared memory
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Load data into shared memory
    if (gid < N) {
        shared[tid] = input[gid];
    }
    
    __syncthreads();  // BARRIER: Wait for all threads to finish loading
    
    // Phase 2: Process data (all threads can now safely read shared memory)
    if (gid < N) {
        output[gid] = shared[tid] * 2.0f;
    }
}
```

**When to use:**
- After writing to shared memory, before reading
- When coordinating multi-phase algorithms
- Before reusing shared memory for different purposes

**Critical Rules:**
- Must be called by ALL threads in a block or NONE
- Cannot be inside conditional statements where some threads might not reach it
- Cannot synchronize across blocks

### 2. Common Pitfall: Conditional Synchronization

```cuda
// WRONG - Deadlock!
__global__ void badSync(float *data) {
    if (threadIdx.x < 64) {
        __syncthreads();  // Only half the threads reach this
    }
    // Threads >= 64 never reach the barrier, causing deadlock
}

// CORRECT
__global__ void goodSync(float *data) {
    if (threadIdx.x < 64) {
        // Do conditional work
    }
    __syncthreads();  // All threads reach this
}
```

### 3. Practical Example: Parallel Reduction

```cuda
__global__ void parallelReduction(float *input, float *output, int N) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory (tree-based)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();  // Critical: ensure all adds complete before next iteration
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

**Why synchronization is needed:** Each iteration depends on the results of the previous iteration. Without `__syncthreads()`, threads might read stale data.

---

## Cross-Block Synchronization

### The Fundamental Problem

**CUDA does NOT provide built-in cross-block synchronization** for good reasons:

1. **No execution order guarantee:** Block 5 might execute before Block 2
2. **Limited concurrency:** Not all blocks may run simultaneously
3. **Deadlock risk:** A block waiting for another block that hasn't started yet

### Solution 1: Multiple Kernel Launches (Recommended)

Kernel launches provide implicit global synchronization. The GPU guarantees all work from one kernel completes before the next starts.

```cuda
// Step 1: Each block computes partial result
__global__ void partialSum(float *input, float *partials, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        partials[blockIdx.x] = sdata[0];
    }
}

// Step 2: Final reduction
__global__ void finalSum(float *partials, float *result, int numBlocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    
    sdata[tid] = (tid < numBlocks) ? partials[tid] : 0.0f;
    __syncthreads();
    
    // Reduce to single value
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) {
        *result = sdata[0];
    }
}

// Host code
int numBlocks = (N + 255) / 256;
partialSum<<<numBlocks, 256>>>(d_input, d_partials, N);
// Implicit synchronization here
finalSum<<<1, 256>>>(d_partials, d_result, numBlocks);
```

**Advantages:**
- Guaranteed to work on all GPUs
- Clean and understandable
- Kernel launch overhead is typically < 5-10 microseconds (negligible)

### Solution 2: Cooperative Groups (CUDA 9+)

Allows explicit grid-wide synchronization, but with strict limitations.

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cooperativeKernel(float *data, int N) {
    // Get grid group handle
    cg::grid_group grid = cg::this_grid();
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Process data
    if (gid < N) {
        data[gid] = data[gid] * 2.0f;
    }
    
    // Synchronize ALL blocks in the grid
    grid.sync();
    
    // Phase 2: Can safely use results from phase 1
    if (gid < N) {
        data[gid] = data[gid] + 1.0f;
    }
}

// Must use special launch API
void launchCooperativeKernel(float *d_data, int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Check if device supports cooperative launch
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (!prop.cooperativeLaunch) {
        printf("Device doesn't support cooperative groups\n");
        return;
    }
    
    // Check if grid size is feasible
    int maxBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, cooperativeKernel, blockSize, 0);
    
    int maxGridSize = maxBlocks * prop.multiProcessorCount;
    
    if (numBlocks > maxGridSize) {
        printf("Grid too large for cooperative launch\n");
        return;
    }
    
    void *args[] = {&d_data, &N};
    cudaLaunchCooperativeKernel(
        (void*)cooperativeKernel, 
        numBlocks, 
        blockSize, 
        args
    );
}
```

**Requirements:**
- Compute Capability 6.0+ (Pascal and newer)
- ALL blocks must fit on GPU simultaneously
- Must use `cudaLaunchCooperativeKernel()`

**When to use:** Rarely. Most algorithms can be restructured to use multiple kernel launches instead.

### Solution 3: Atomic Operations (Limited Use)

Atomic operations provide thread-safe updates but don't provide full synchronization semantics.

```cuda
__global__ void atomicCounter(int *counter, int *results) {
    // Each block increments counter atomically
    if (threadIdx.x == 0) {
        atomicAdd(counter, 1);
    }
    __syncthreads();
    
    // Store block ID
    results[blockIdx.x] = blockIdx.x;
}
```

**Use cases:**
- Building histograms
- Scatter operations
- Reference counting
- Progress tracking (with care)

**NOT suitable for:**
- General barrier synchronization
- Ordering guarantees between blocks

---

## Warp-Level Synchronization

Threads within a warp (32 threads) execute in lockstep on modern GPUs, enabling specialized optimizations.

### 1. Implicit Warp Synchronization (Pre-Volta)

On pre-Volta GPUs, threads in a warp were implicitly synchronized. **This is no longer guaranteed!**

```cuda
// WRONG on Volta+ (may work on older GPUs)
__global__ void implicitWarpSync(float *data) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    
    if (lane == 0) {
        data[tid] = 42.0f;
    }
    // No sync - assuming warp synchronization
    float value = data[tid - lane];  // May read stale data!
}
```

### 2. `__syncwarp()` - Explicit Warp Synchronization

Introduced in CUDA 9 for Volta+ GPUs.

```cuda
__global__ void explicitWarpSync(float *data) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    
    if (lane == 0) {
        data[tid] = 42.0f;
    }
    __syncwarp();  // Explicit warp-level barrier
    
    float value = data[tid - lane];  // Now safe
}
```

### 3. Warp Shuffle Operations

Communicate directly between threads in a warp without shared memory.

```cuda
__global__ void warpReduction(float *input, float *output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads one value
    float value = (tid < N) ? input[tid] : 0.0f;
    
    // Reduce within warp using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    
    // First thread of each warp writes result
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        atomicAdd(output, value);
    }
}
```

**Shuffle functions:**
- `__shfl_sync()` - Exchange data between arbitrary lanes
- `__shfl_up_sync()` - Get value from lower lane
- `__shfl_down_sync()` - Get value from higher lane
- `__shfl_xor_sync()` - Exchange with lane XOR'd with offset

**Advantages:**
- Faster than shared memory (no memory access)
- Lower latency
- Saves shared memory for other uses

### 4. Warp Vote Functions

Query collective state across a warp.

```cuda
__global__ void warpVoteExample(float *data, int *allPositive, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float value = (tid < N) ? data[tid] : 0.0f;
    
    // Check if all threads in warp have positive values
    int all_pos = __all_sync(0xffffffff, value > 0.0f);
    
    // Check if any thread in warp has negative value
    int any_neg = __any_sync(0xffffffff, value < 0.0f);
    
    // Count how many threads have value > 10
    int count = __popc(__ballot_sync(0xffffffff, value > 10.0f));
}
```

---

## Advanced Topics

### 1. Memory Fence Operations

Control visibility of memory operations without full synchronization.

```cuda
// Ensure shared memory writes are visible to block
__threadfence_block();

// Ensure global memory writes are visible to grid
__threadfence();

// Ensure writes are visible to entire system (multi-GPU)
__threadfence_system();
```

**Example: Producer-Consumer Pattern**

```cuda
__global__ void producerConsumer(int *data, volatile int *flag) {
    if (threadIdx.x == 0) {
        // Producer: write data
        data[blockIdx.x] = 42;
        __threadfence();  // Ensure data is visible
        flag[blockIdx.x] = 1;  // Signal completion
    } else {
        // Consumer: wait for data
        while (flag[blockIdx.x] == 0) {}
        __threadfence();  // Ensure we see the data
        int value = data[blockIdx.x];
    }
}
```

### 2. Lock-Free Algorithms

Advanced patterns using atomics for coordination.

```cuda
__device__ void spinLock(int *mutex) {
    while (atomicCAS(mutex, 0, 1) != 0) {
        // Spin until we acquire lock
    }
}

__device__ void unlock(int *mutex) {
    atomicExch(mutex, 0);
}

__global__ void criticalSection(int *mutex, float *sharedData) {
    spinLock(mutex);
    
    // Critical section - only one thread at a time
    *sharedData += 1.0f;
    
    unlock(mutex);
}
```

**Warning:** Spin locks can waste GPU cycles and cause deadlocks if not used carefully.

### 3. Multi-GPU Synchronization

Synchronizing work across multiple GPUs.

```cuda
// Using CUDA streams and events
cudaEvent_t events[2];
cudaStream_t streams[2];

for (int gpu = 0; gpu < 2; gpu++) {
    cudaSetDevice(gpu);
    cudaStreamCreate(&streams[gpu]);
    cudaEventCreate(&events[gpu]);
}

// Launch work on both GPUs
for (int gpu = 0; gpu < 2; gpu++) {
    cudaSetDevice(gpu);
    myKernel<<<grid, block, 0, streams[gpu]>>>(data[gpu]);
    cudaEventRecord(events[gpu], streams[gpu]);
}

// Wait for both GPUs to complete
for (int gpu = 0; gpu < 2; gpu++) {
    cudaSetDevice(gpu);
    cudaEventSynchronize(events[gpu]);
}
```

---

## Best Practices

### 1. Minimize Synchronization

Every `__syncthreads()` has a cost. Reduce usage when possible.

```cuda
// Unnecessary sync
for (int i = 0; i < 100; i++) {
    data[tid] += 1.0f;
    __syncthreads();  // Not needed if threads don't share data
}

// Better
for (int i = 0; i < 100; i++) {
    data[tid] += 1.0f;
}
__syncthreads();  // Sync once at end if needed
```

### 2. Use Appropriate Synchronization Level

Choose the narrowest synchronization scope needed.

```
Warp-level (__syncwarp, shuffle) 
    ↓ (faster, less overhead)
Block-level (__syncthreads)
    ↓
Grid-level (multiple kernels, cooperative groups)
    ↓ (slower, more overhead)
```

### 3. Avoid Deadlocks

```cuda
// DEADLOCK: Divergent control flow
__global__ void deadlock(float *data) {
    if (threadIdx.x % 2 == 0) {
        __syncthreads();  // Even threads wait
    } else {
        data[threadIdx.x] = 0;
        __syncthreads();  // Odd threads wait
    }
}

// CORRECT: All threads reach same sync point
__global__ void noDeadlock(float *data) {
    if (threadIdx.x % 2 == 0) {
        data[threadIdx.x] = 1;
    } else {
        data[threadIdx.x] = 0;
    }
    __syncthreads();  // All threads sync together
}
```

### 4. Profile Your Code

Use Nsight Compute to identify synchronization bottlenecks.

```bash
ncu --set full --section SyncAnalysis ./myProgram
```

### 5. Consider Alternatives to Synchronization

- **Independent work:** Restructure algorithms to minimize dependencies
- **Double buffering:** Use separate memory regions for read/write
- **Stream parallelism:** Overlap computation and communication

---

## Summary Cheat Sheet

| Synchronization Type | Function | Scope | Speed | Use Case |
|---------------------|----------|-------|-------|----------|
| Warp shuffle | `__shfl_*_sync()` | 32 threads | Fastest | Warp reductions |
| Warp barrier | `__syncwarp()` | 32 threads | Very fast | Warp coordination |
| Block barrier | `__syncthreads()` | Block | Fast | Shared memory coordination |
| Memory fence (block) | `__threadfence_block()` | Block | Fast | Memory visibility |
| Memory fence (grid) | `__threadfence()` | Grid | Medium | Inter-block communication |
| Kernel launch | `kernel<<<>>>()` | Grid | Slow | Cross-block sync (recommended) |
| Cooperative groups | `grid.sync()` | Grid | Slow | Grid-wide barriers (limited) |
| Atomic operations | `atomicAdd()`, etc. | Global | Medium | Scatter, histograms |

**Golden Rule:** Use the minimum synchronization scope required for your algorithm. When in doubt, use multiple kernel launches for cross-block coordination.
