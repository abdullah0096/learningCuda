# CUDA Roadmap

## 🎯 Philosophy

This roadmap follows a **hardware-first, depth-oriented** approach:
1. Understand the machine first
2. Learn the execution model
3. Build foundational algorithms
4. Master tools for visibility
5. Optimize with knowledge
6. Learn advanced patterns
7. Internalize best practices

---

## Phase 1: Hardware Architecture
*Foundation: Know your machine before programming it*

### 1.1 GPU Architecture Fundamentals
- **GPU vs CPU paradigm**
  - Latency-oriented vs throughput-oriented design
  - SIMD/SIMT execution model
  - Why GPUs exist and what they're good at
  
- **NVIDIA GPU architecture hierarchy**
  - Graphics Processing Clusters (GPCs)
  - Streaming Multiprocessors (SMs) - the workhorses
  - CUDA cores, Tensor cores, RT cores (depending on generation)
  - Warp schedulers and dispatch units

- **Compute capability and generations**
  - Fermi, Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Hopper, Blackwell
  - What changes between generations
  - How to write portable code across compute capabilities

- **Resource limitations per SM**
  - Maximum threads per SM
  - Maximum blocks per SM
  - Register file size and allocation
  - Shared memory capacity
  - Warp slots and active warps

### 1.2 Memory Hierarchy Deep Dive
- **Memory types and characteristics**
  - Global memory (DRAM) - large, slow, visible to all
  - L2 cache - shared across SMs
  - L1 cache / Shared memory - per SM, fast
  - Registers - per thread, fastest
  - Constant cache - read-only, broadcast capability
  - Texture cache - read-only, spatial locality optimization

- **Memory bandwidth and latency**
  - HBM/GDDR specifications
  - Memory bandwidth as primary bottleneck
  - Latency numbers: registers (~1 cycle), shared memory (~20-30 cycles), global memory (200-400 cycles)
  - Why latency hiding is critical

- **Physical memory organization**
  - Memory channels and partitions
  - Burst transactions and alignment
  - Bank structure in shared memory
  - How memory controllers work

### 1.3 Execution Pipeline
- **Warp execution mechanics**
  - What happens when a warp executes
  - Instruction fetch, decode, issue, execute
  - Single Instruction Multiple Thread (SIMT) model
  - Warp divergence at hardware level

- **Occupancy and resource binding**
  - What occupancy really means
  - Register pressure and occupancy
  - Shared memory pressure and occupancy
  - Finding the sweet spot

- **Scheduling and context switching**
  - How SMs schedule warps
  - Zero-cost context switching
  - Latency hiding through multithreading
  - Stalls and bubbles in the pipeline

### 1.4 Querying and Understanding Your GPU
- **Device properties and capabilities**
  - Using `cudaGetDeviceProperties()`
  - Understanding every field returned
  - Interpreting compute capability
  - Multi-GPU systems and peer access

- **Theoretical limits**
  - Peak FLOPS calculation
  - Peak memory bandwidth
  - Arithmetic intensity and roofline model
  - When you're compute-bound vs memory-bound

**Key Outcome:** You should be able to explain why a 3090 behaves differently than an A100, and predict performance characteristics before writing code.

---

## Phase 2: CUDA Execution Model
*Foundation: How your code maps to hardware*

### 2.1 Thread Hierarchy
- **Thread organization philosophy**
  - Why three-level hierarchy (thread → block → grid)
  - Mapping to hardware (threads → warps → SMs)
  - Independence assumptions at each level
  
- **Thread, Block, Grid dimensions**
  - 1D, 2D, 3D indexing schemes
  - Built-in variables: threadIdx, blockIdx, blockDim, gridDim
  - Calculating global indices
  - When to use which dimensionality

- **Block independence guarantee**
  - Why blocks must be independent
  - Implication for algorithms
  - Comparison with MPI ranks (similar independence concept)
  - No inter-block synchronization (within single kernel)

### 2.2 Kernel Launch Mechanics
- **Launch configuration**
  - `<<<gridDim, blockDim, sharedMem, stream>>>` syntax
  - Choosing grid and block dimensions
  - Dynamic vs static shared memory
  - Stream association

- **Execution configuration rules**
  - Block size must be ≤ 1024 threads
  - Block dimensions must fit SM resources
  - Grid dimensions can be massive (billions)
  - Ceiling division pattern for coverage

- **Launch overhead and granularity**
  - Cost of kernel launch
  - Optimal kernel granularity
  - When to fuse vs split kernels

### 2.3 Warp-Level Understanding
- **Warp as execution unit**
  - 32 threads always (hardware constant)
  - SIMT lockstep execution
  - Warp divergence and serialization
  - Reconvergence points

- **Warp scheduling**
  - Multiple warps per SM
  - Eligible vs stalled warps
  - Instruction-level parallelism
  - Hiding memory latency with warp switching

- **Block size optimization**
  - Why multiples of 32 matter
  - Warp efficiency calculation
  - Tail effects and wasted threads
  - Balancing block size with occupancy

### 2.4 Synchronization Primitives
- **Within a block: `__syncthreads()`**
  - Barrier semantics
  - Use cases: shared memory coordination
  - Dangers: deadlock from conditional sync
  - Performance cost

- **Between blocks: Kernel boundaries**
  - Why no inter-block sync within kernel
  - Using multiple kernel launches
  - Persistent kernels (advanced)

- **Atomic operations**
  - Hardware support for atomics
  - Performance characteristics
  - When to use vs avoid
  - Atomic contention problems

- **Memory fences and visibility**
  - `__threadfence()`, `__threadfence_block()`, `__threadfence_system()`
  - Memory consistency model
  - Volatile semantics

**Key Outcome:** You should understand exactly what happens when you launch a kernel, how threads map to hardware, and why certain patterns perform well or poorly.

---

## Phase 3: Foundational Algorithms
*Application: Build understanding through implementation*

### 3.1 Vector Operations
- **Vector addition**
  - Simplest parallel pattern
  - 1D indexing and launch configuration
  - Handling arbitrary sizes (boundary checks)
  - Memory access pattern analysis
  
- **Element-wise operations**
  - Scaling, multiplication, division
  - Fused operations
  - Activation functions (ReLU, sigmoid, tanh)
  - Understanding memory bandwidth limits

- **Vector reductions**
  - Sum, max, min, mean
  - Comparison with MPI_Reduce
  - Tree-based reduction in shared memory
  - Multiple kernel approach vs single kernel
  - Warp shuffle intrinsics for efficiency

### 3.2 Matrix Addition and Subtraction
- **1D approach**
  - Treating matrix as flat array
  - Simple and effective
  - When to use this

- **2D approach**
  - Row/column indexing
  - Conceptual clarity
  - Preparing for more complex operations

- **Performance analysis**
  - Memory bandwidth as bottleneck
  - Measuring achieved bandwidth
  - Comparison with theoretical peak
  - Why this is memory-bound, not compute-bound

### 3.3 Matrix Multiplication - The Deep Dive
**This is your major learning vehicle - spend significant time here**

- **Naive implementation**
  - One thread per output element
  - Accessing row from A, column from B
  - Understanding why it's slow
  - Global memory traffic analysis

- **Tiled matrix multiplication**
  - Concept of blocking/tiling
  - Loading tiles into shared memory
  - Thread cooperation within block
  - Reducing global memory access
  - 2D block structure necessity

- **Optimization progression**
  - Step 1: Basic tiling (16×16 tiles)
  - Step 2: Handling non-multiple dimensions
  - Step 3: Rectangular tiles
  - Step 4: Increasing tile size (32×32)
  - Step 5: Register blocking
  - Step 6: Vectorized loads
  - Each step: measure, understand, improve

- **Understanding cuBLAS**
  - What library implementations do
  - Comparing your code to cuBLAS (SGEMM)
  - Realistic performance expectations
  - When to use libraries vs custom kernels

- **Different problem sizes**
  - Small matrices (fit in cache)
  - Medium matrices (typical)
  - Large matrices (out-of-cache)
  - Rectangular matrices
  - Batched matrix multiplication

**Key Outcome:** Matrix multiplication is where everything comes together. Spend time here until you deeply understand memory access patterns, shared memory, and optimization strategies.

### 3.4 Matrix Transpose
- **Naive transpose**
  - Simple implementation
  - Why it's slow (memory coalescing issues)

- **Shared memory transpose**
  - Using shared memory as staging area
  - Bank conflict problems
  - Padding to avoid conflicts
  - Achieving near-peak bandwidth

- **Lessons learned**
  - Coalesced access importance
  - Bank conflicts in shared memory
  - When to use shared memory

### 3.5 Convolution
- **1D convolution**
  - Simple signal processing example
  - Handling boundaries
  - Constant memory for kernel

- **2D convolution (image filtering)**
  - Sliding window pattern
  - Halo regions and ghost cells
  - Shared memory optimization
  - Separable vs non-separable kernels

**Key Outcome:** Build intuition for parallel patterns, memory optimization, and the interplay between threads and memory hierarchy.

---

## Phase 4: Debugging and Profiling
*Visibility: You can't optimize what you can't measure*

### 4.1 Error Handling and Debugging
- **Proper error checking**
  - Checking every CUDA call
  - `cudaGetLastError()` after kernel launches
  - Building error-checking wrappers
  - Handling errors gracefully

- **Common errors and debugging**
  - Illegal memory access
  - Launch failures
  - Out-of-bounds access
  - Race conditions
  - Uninitialized memory

- **Debugging tools**
  - `cuda-gdb` - step through kernels
  - `cuda-memcheck` - memory error detection
  - Nsight Debugger - GUI debugging
  - `printf()` in kernels (surprisingly useful)
  - Conditional debugging techniques

- **Strategies for debugging**
  - Start with small problem sizes
  - Compare with CPU reference
  - Validate intermediate results
  - Binary search for bugs
  - Isolate and simplify

### 4.2 Profiling with Command-Line Tools
- **nvprof basics** (legacy but still useful)
  - Timeline view
  - Metrics and events
  - Understanding output
  
- **Nsight Systems (nsys)**
  - System-wide view
  - CPU-GPU interaction
  - Kernel launches and streams
  - Memory transfers
  - Finding CPU-side bottlenecks

- **Nsight Compute (ncu)**
  - Deep kernel analysis
  - Roofline analysis
  - Memory throughput metrics
  - Warp execution efficiency
  - Occupancy analysis
  - Instruction mix

### 4.3 Key Metrics to Understand
- **Occupancy**
  - Achieved vs theoretical occupancy
  - Why higher isn't always better
  - Occupancy-performance relationship

- **Memory metrics**
  - Global memory throughput
  - L1/L2 cache hit rates
  - Memory bandwidth utilization
  - Coalescing efficiency

- **Compute metrics**
  - FLOPS achieved
  - Instruction throughput
  - Warp execution efficiency
  - Branch divergence statistics

- **Derived insights**
  - Compute-bound vs memory-bound
  - Where time is spent
  - What to optimize next

### 4.4 Performance Methodology
- **Roofline model**
  - Understanding arithmetic intensity
  - Plotting your kernel
  - Identifying limitations
  - Setting realistic goals

- **Measurement discipline**
  - Timing kernels properly
  - Excluding transfer overhead
  - Multiple runs and statistics
  - Comparing with theoretical limits

- **Optimization workflow**
  - Measure baseline
  - Identify bottleneck
  - Apply optimization
  - Measure again
  - Understand why (or why not) it helped

**Key Outcome:** You should be comfortable profiling any kernel, understanding where time is spent, identifying bottlenecks, and knowing what to optimize.

---

## Phase 5: Memory and Performance Optimization
*Mastery: Extracting maximum performance*

### 5.1 Memory Access Patterns
- **Coalesced memory access**
  - What coalescing means at hardware level
  - Aligned and sequential access importance
  - Stride effects
  - Structure of Arrays (SoA) vs Array of Structures (AoS)
  - Measuring coalescing efficiency

- **Cache behavior**
  - L1/L2 cache architecture
  - Cache line size (32-128 bytes)
  - Spatial and temporal locality
  - Cache thrashing
  - Read-only cache usage

- **Memory alignment**
  - 128-byte alignment for optimal coalescing
  - Padding structures
  - Alignment requirements for different types

### 5.2 Shared Memory Mastery
- **Shared memory as scratchpad**
  - Programmer-managed cache
  - 100x faster than global memory
  - Limited size (48-96 KB per SM)
  - Configuring L1/shared split

- **Bank conflicts**
  - 32 banks in shared memory
  - Conflict-free access patterns
  - Broadcast mechanism
  - Diagnosing and fixing conflicts
  - Padding techniques

- **Shared memory patterns**
  - Tiling for matrix operations
  - Reduction trees
  - Prefix sums
  - Transposition
  - Cooperative data loading

### 5.3 Register Optimization
- **Register pressure**
  - Limited registers per thread
  - Spilling to local memory (slow!)
  - Checking register usage at compile time
  - Balancing occupancy vs register usage

- **Compiler optimization**
  - `-use_fast_math` flag
  - Inlining functions
  - Loop unrolling
  - `#pragma unroll` directives
  - Optimization flags and their effects

### 5.4 Instruction-Level Optimization
- **Reducing divergence**
  - Restructuring conditionals
  - Data-dependent vs uniform branches
  - Minimizing branch granularity
  - Using predication

- **Instruction throughput**
  - Latency vs throughput
  - Dependent vs independent operations
  - Instruction-level parallelism (ILP)
  - Hiding instruction latency

- **Special function units**
  - Fast math functions (`__sin`, `__cos`, etc.)
  - Reciprocal approximations
  - When precision tradeoffs make sense

### 5.5 Occupancy Optimization
- **Resource balancing**
  - Registers, shared memory, threads per block
  - Finding optimal configuration
  - Using occupancy calculator
  - Launch bounds pragma

- **When occupancy doesn't matter**
  - Compute-bound kernels
  - Already saturated resources
  - Occupancy vs achieved performance

### 5.6 Advanced Memory Techniques
- **Pinned (page-locked) memory**
  - Faster host-device transfers
  - When to use `cudaMallocHost()`
  - Memory limitations

- **Unified Memory**
  - Automatic migration
  - Prefetching and hints
  - When it helps vs hurts performance

- **Zero-copy memory**
  - Direct GPU access to CPU memory
  - Use cases and limitations

**Key Outcome:** You should know the full optimization toolkit and when to apply each technique. You should be able to look at a kernel and identify optimization opportunities.

---

## Phase 6: Advanced Patterns
*Sophistication: Complex parallel algorithms*

### 6.1 Reduction Patterns
- **Parallel reduction fundamentals**
  - Associative operations only
  - Tree-based reduction
  - Multiple kernels vs single kernel

- **Optimization progression**
  - Naive reduction
  - Minimizing divergence
  - Sequential addressing
  - Bank conflict avoidance
  - Unrolling the last warp
  - Multiple elements per thread

- **Warp-level primitives**
  - `__shfl_down_sync()` and siblings
  - Lock-step warp operations
  - Modern reduction techniques
  - When to use vs shared memory

- **Segmented reductions**
  - Reducing multiple segments
  - Applications to grouped operations

### 6.2 Scan (Prefix Sum)
- **Scan fundamentals**
  - Inclusive vs exclusive scan
  - Importance in parallel algorithms
  - Applications: compaction, allocation, etc.

- **Work-efficient scan**
  - Blelloch scan algorithm
  - Up-sweep and down-sweep phases
  - Bank conflict considerations

- **Large array scan**
  - Multi-block scan
  - Block-level prefix sums
  - Handling arbitrary sizes

### 6.3 Sorting and Searching
- **Bitonic sort**
  - Comparison network
  - Power-of-2 sizes
  - Shared memory implementation

- **Radix sort**
  - Digit-by-digit sorting
  - Using scan for redistribution
  - Comparing with thrust::sort

- **Binary search**
  - Parallel search in sorted arrays
  - Warp-level cooperation

### 6.4 Histogram and Binning
- **Atomic histogram**
  - Simple but contention-prone
  - When it works well

- **Privatized histogram**
  - Per-block histograms
  - Reduction step
  - Better scalability

- **Sort-based histogram**
  - Using sorting for binning
  - When it's faster than atomics

### 6.5 Stencil Operations
- **Stencil patterns**
  - Jacobi iteration
  - Heat equation
  - Finite difference methods

- **Optimization strategies**
  - Shared memory for halo regions
  - Handling boundaries
  - 2.5D blocking techniques

### 6.6 Graph Algorithms
- **BFS and DFS on GPU**
  - Challenges of irregular parallelism
  - Work queues
  - Level synchronization

- **Understanding limitations**
  - When GPU isn't appropriate
  - Hybrid CPU-GPU approaches

### 6.7 Sparse Matrix Operations
- **Storage formats**
  - COO, CSR, CSC, ELL
  - Format selection criteria

- **SpMV (Sparse Matrix-Vector)**
  - Memory access patterns
  - Warp-level cooperation
  - Comparing with cuSPARSE

### 6.8 Streams and Concurrency
- **CUDA streams**
  - Asynchronous execution
  - Overlapping compute and transfer
  - Default stream vs explicit streams

- **Stream synchronization**
  - Events for timing and dependencies
  - Avoiding implicit synchronization

- **Multi-stream patterns**
  - Pipeline parallelism
  - Concurrent kernel execution
  - When it helps performance

### 6.9 Dynamic Parallelism
- **Device-side kernel launches**
  - Nested parallelism
  - Recursive algorithms
  - Use cases and overheads

### 6.10 Cooperative Groups
- **Modern synchronization**
  - Beyond `__syncthreads()`
  - Flexible thread grouping
  - Grid-wide synchronization
  - Warp-level operations

**Key Outcome:** Expand your parallel algorithm toolkit beyond simple patterns. Understand when and how to apply sophisticated techniques.

---

## Phase 7: Best Practices
*Wisdom: Building maintainable, production-ready code*

### 7.1 Code Organization
- **Separation of concerns**
  - Host code vs device code
  - Kernel launching abstractions
  - Memory management wrappers

- **Template metaprogramming**
  - Type-generic kernels
  - Compile-time optimization
  - SFINAE and constraints

- **Build systems**
  - CMake for CUDA projects
  - Separate compilation
  - Handling multiple architectures

### 7.2 Portability
- **Compute capability handling**
  - Feature detection
  - Graceful degradation
  - Architecture-specific optimizations

- **Multi-GPU systems**
  - Device enumeration
  - Load balancing strategies
  - Peer-to-peer transfers

### 7.3 Integration Patterns
- **Mixing with CPU code**
  - When to use GPU vs CPU
  - Amdahl's law implications
  - Minimizing transfer overhead

- **Library integration**
  - cuBLAS, cuFFT, cuSPARSE, cuDNN
  - When custom kernels make sense
  - Interoperability

- **Language interop**
  - Calling CUDA from C++
  - Python integration (pyCUDA, CuPy, Numba)
  - Fortran integration

### 7.4 Performance Patterns
- **Kernel fusion**
  - Combining operations
  - Reducing memory traffic
  - When fusion helps

- **Batching**
  - Processing multiple inputs
  - Amortizing overhead
  - Optimal batch sizes

- **Persistent threads**
  - Long-running kernels
  - Work stealing
  - When to use this pattern

### 7.5 Common Pitfalls
- **Race conditions**
  - Shared memory races
  - Global memory races
  - Debugging strategies

- **Memory leaks**
  - Tracking allocations
  - RAII patterns
  - Tools for detection

- **Integer overflow**
  - Index calculations
  - Large problem sizes
  - Using appropriate types

- **Floating point issues**
  - Precision considerations
  - Summation order matters
  - Catastrophic cancellation

### 7.6 Testing and Validation
- **Unit testing kernels**
  - Testing strategy
  - Handling GPU errors in tests
  - Continuous integration

- **Numerical validation**
  - Comparing with CPU reference
  - Relative vs absolute error
  - Handling floating point

- **Performance regression testing**
  - Tracking performance over time
  - Automated benchmarking

### 7.7 Documentation
- **Documenting kernel assumptions**
  - Input constraints
  - Memory requirements
  - Performance characteristics

- **Performance documentation**
  - Expected throughput
  - Scaling behavior
  - Known limitations

### 7.8 When NOT to Use GPU
- **Decision criteria**
  - Transfer overhead dominates
  - Insufficient parallelism
  - Memory capacity limits
  - Complex control flow

- **Hybrid approaches**
  - CPU for control, GPU for compute
  - Heterogeneous algorithms
  - Fallback strategies

**Key Outcome:** Write professional, maintainable CUDA code that others can use and extend. Know when to use GPU acceleration and when not to.

---

## 🎯 Learning Approach

### For Each Phase:

1. **Study deeply** - Read documentation, papers, blog posts
2. **Implement** - Write code, experiment, break things
3. **Measure** - Profile everything, understand results
4. **Iterate** - Optimize based on measurements
5. **Question** - Why does this work? Why is this faster?

### jfjgjgj:

- **60% implementation** - Writing and experimenting with code
- **20% reading** - Documentation, papers, resources
- **20% profiling/analysis** - Understanding performance

### Depth over Breadth:

- Spend significant time on matrix multiplication (Phase 3.3)
- Make it your "reference implementation"
- Return to it when learning new concepts
- Use it to test profiling tools
- Apply all optimizations to it

### MPI vs CUDA:

- **Compare concepts**: Blocks ↔ MPI ranks, Grids ↔ Communicators
- **Memory models**: Distributed (MPI) vs Hierarchical (CUDA)
- **Synchronization**: Point-to-point/collective ↔ Barriers/atomics
- **Decomposition**: Domain decomposition ↔ Thread decomposition

---

## 📚 Essential Resources

### Must-Read Documentation
- **NVIDIA CUDA C++ Programming Guide** - Your primary reference
- **NVIDIA CUDA C++ Best Practices Guide** - Optimization bible
- **GPU Architecture Whitepapers** - Understand your hardware

### Recommended Books
- **"Programming Massively Parallel Processors"** - Comprehensive textbook
- **"CUDA by Example"** - Practical introduction (quick review for you)

### Online Resources
- **NVIDIA Developer Blog** - Latest techniques and insights
- **CUDA Samples Repository** - Reference implementations
- **Nsight Documentation** - Master the profiling tools

### Community
- **NVIDIA Developer Forums** - Get help when stuck
- **GitHub** - Study production code
- **ArXiv** - Research papers on GPU algorithms

---
Date : 29.12.2025
