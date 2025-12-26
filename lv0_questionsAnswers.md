# CUDA Fundamentals Questionnaire - Complete Answer Key
## To be taken after the understanding of cuda fundamentals are covered. Vector addition, matrix addition programs are implemneted.
### To be taken before matrix multiplication, profiling are covered.
### 26th Dec. 2025
---

## Section 1: Basic Concepts

### Q1. What is the relationship between threads, blocks, and grids? Draw a simple hierarchy.

**Answer:**
```
Grid (Logical unit, created at kernel launch, destroyed at completion)
├── Contains Blocks in X, Y, Z dimensions
    └── Block
        ├── Contains Threads in X, Y, Z dimensions
        └── Thread (Executes kernel code)
```

**Key Points:**
- **Grid**: Logical organization of all blocks for a kernel launch
- **Block**: Group of threads that can cooperate (shared memory, synchronization)
- **Thread**: Individual execution unit that runs the kernel code
- All blocks in a grid have the same dimensions
- Each thread has access to:
  - `threadIdx.x/y/z` (position within block)
  - `blockIdx.x/y/z` (block's position in grid)
  - `blockDim.x/y/z` (size of block)
  - `gridDim.x/y/z` (size of grid)

---

### Q2. In the kernel code below, what values will `threadIdx.x`, `blockIdx.x`, and `blockDim.x` have for the thread that processes element 550?

```cuda
kernel<<<100, 256>>>(data, N);
// Inside kernel:
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

**Answer:**
- Total threads: 100 × 256 = 25,600
- For element 550: `550 = blockIdx.x * 256 + threadIdx.x`
- **blockIdx.x = 2** (because 550 / 256 = 2.148...)
- **threadIdx.x = 38** (because 550 % 256 = 38)
- **blockDim.x = 256** (constant for all threads)

**Verification:** 2 × 256 + 38 = 512 + 38 = 550 ✓

---

### Q3. True or False: `threadIdx` and `blockIdx` are variables you set when launching a kernel.

**Answer:** **False**

These are **built-in variables** automatically assigned by CUDA to each thread. The programmer sets:
- Number of blocks (which determines range of `blockIdx`)
- Threads per block (which determines range of `threadIdx`)

But the actual values are assigned by the CUDA runtime/hardware.

---

### Q4. True or False: The grid is a physical hardware component on the GPU.

**Answer:** **False**

The grid is a **logical organization** that exists only for the duration of a kernel launch. Physical components are:
- **Streaming Multiprocessors (SMs)**: The actual compute units
- **CUDA cores**: Processing elements within SMs
- **Memory hierarchy**: Global, shared, registers, L1/L2 cache

---

### Q5. What does this formula calculate? `int idx = blockIdx.x * blockDim.x + threadIdx.x;`

**Answer:**

This calculates the **global 1D thread index** - a unique identifier for each thread across the entire grid.

**How it works:**
- `blockIdx.x * blockDim.x`: Skips all threads in previous blocks
- `+ threadIdx.x`: Adds position within current block
- Result: Unique index from 0 to (total_threads - 1)

**Example:**
- Block 0: indices 0-255
- Block 1: indices 256-511
- Block 2: indices 512-767

---

### Q6. If you launch `kernel<<<50, 512>>>()`, how many total threads are created?

**Answer:** **25,600 threads**

Calculation: 50 blocks × 512 threads/block = 25,600 threads

---

### Q7. Can two threads from different blocks synchronize with each other using `__syncthreads()`?

**Answer:** **No**

`__syncthreads()` only synchronizes threads **within the same block**. Threads in different blocks:
- May run on different SMs
- May run at different times
- Cannot directly synchronize or communicate through shared memory

For cross-block synchronization, you need:
- Multiple kernel launches
- Atomic operations on global memory
- Cooperative groups (advanced feature)

---

### Q8. What is a Streaming Multiprocessor (SM)?

**Answer:**

An SM is a **physical compute unit** on the GPU that:
- Executes blocks of threads
- Contains:
  - Multiple CUDA cores (32-128 depending on architecture)
  - Warp schedulers
  - Register file
  - Shared memory / L1 cache
  - Texture/constant cache
- Executes threads in groups of 32 (warps)
- Modern GPUs have 10-130+ SMs depending on model

**Example:** A GPU with 80 SMs can execute up to 80 blocks simultaneously (if each SM runs one block).

---

### Q9. Do you as a programmer directly control which block runs on which SM?

**Answer:** **No**

The CUDA runtime and hardware scheduler automatically:
- Distribute blocks across available SMs
- Balance load dynamically
- Handle block scheduling as resources become available

The programmer only controls:
- Number of blocks and threads
- Resource usage (registers, shared memory)

---

### Q10. What is a warp and why is it important?

**Answer:**

A **warp** is a group of **32 threads** that execute together on an SM.

**Why important:**
1. **Execution unit**: SMs execute warps, not individual threads
2. **SIMT execution**: All threads in a warp execute the same instruction simultaneously
3. **Efficiency**: For best performance, block size should be a multiple of 32
4. **Divergence**: If threads in a warp take different code paths (if/else), execution is serialized, reducing efficiency

**Example:**
- Block with 256 threads = 8 warps (perfect efficiency)
- Block with 100 threads = 4 warps (but only uses 100/128 = 78% of warp slots)

---

## Section 2: Launch Configuration

### Q11. You need to process an array of 10,000 elements. Fill in the blanks:

```cuda
int N = 10000;
int threadsPerBlock = _______;  // Your choice
int blocksPerGrid = _______;    // Calculate this
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);
```

**Answer:**
```cuda
int N = 10000;
int threadsPerBlock = 256;  // Good choice (multiple of 32)
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  
// = (10000 + 255) / 256 = 10255 / 256 = 40 blocks

kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);
```

**Result:** 40 blocks × 256 threads = 10,240 threads (240 will be idle, handled by boundary check)

---

### Q12. Why is `(N + threadsPerBlock - 1) / threadsPerBlock` used instead of just `N / threadsPerBlock`?

**Answer:**

This is **ceiling division** - it rounds UP to ensure enough threads to cover all elements.

**Example with N=1000, threadsPerBlock=256:**
- Wrong: `1000 / 256 = 3` blocks → only 768 threads (not enough!)
- Right: `(1000 + 255) / 256 = 1255 / 256 = 4` blocks → 1024 threads ✓

**Formula explanation:**
- Adding `(threadsPerBlock - 1)` before dividing ensures any remainder causes rounding up
- Equivalent to `ceil(N / threadsPerBlock)` but uses only integer operations

---

### Q13. For a 512×512 matrix addition using 2D configuration with 16×16 thread blocks:

**Answers:**
- **Threads per block:** 16 × 16 = **256 threads**
- **Blocks in X direction:** (512 + 15) / 16 = 527 / 16 = **32 blocks**
- **Blocks in Y direction:** (512 + 15) / 16 = 527 / 16 = **32 blocks**
- **Total blocks:** 32 × 32 = **1,024 blocks**
- **Total threads launched:** 1,024 blocks × 256 threads = **262,144 threads**

**Note:** Elements needed = 512 × 512 = 262,144, so perfect match!

---

### Q14. What's wrong with this code?

```cuda
int threadsPerBlock = 2000;  // Want lots of threads!
int blocksPerGrid = (N + 1999) / 2000;
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);
```

**Answer:**

**Critical error:** `threadsPerBlock = 2000` **exceeds the maximum** of 1024 threads per block (on most GPUs). The kernel **will not launch** and will return an error.

**Additional issues if limit was higher:**
- 2000 is not a multiple of 32 (warp size)
- 2000 / 32 = 62.5 warps → uses 63 warps
- Wastes 16 thread slots per block (63 × 32 - 2000 = 16 wasted)
- Warp efficiency: 2000/2016 = 99.2%

**Fix:**
```cuda
int threadsPerBlock = 256;  // or 512, both within limit and multiple of 32
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
```

---

### Q15. Which is better for a block size and why?
- Option A: 250 threads per block
- Option B: 256 threads per block

**Answer:** **Option B (256 threads)**

**Reasons:**
1. **256 is a multiple of 32** (warp size)
   - 256 / 32 = 8 warps (perfect!)
   - 100% warp efficiency
   
2. **250 is not a multiple of 32**
   - 250 / 32 = 7.8125 warps → uses 8 warps
   - Actually uses 256 thread slots (8 × 32)
   - 6 thread slots wasted per block
   - Warp efficiency: 250/256 = 97.7%

**Rule:** Always use block sizes that are multiples of 32: {32, 64, 96, 128, 192, 256, 512, 1024}

---

### Q16. You have a 100×200 matrix. Which 2D block configuration is better?
- Option A: `dim3(10, 20)` = 200 threads
- Option B: `dim3(16, 16)` = 256 threads

**Answer:** **Option B: dim3(16, 16)**

**Analysis:**

**Option A (10×20 = 200 threads):**
- 200 / 32 = 6.25 warps → uses 7 warps
- Warp efficiency: 200/224 = 89.3%
- Not a standard size

**Option B (16×16 = 256 threads):**
- 256 / 32 = 8 warps (perfect!)
- Warp efficiency: 100%
- Standard, well-optimized size
- Square shape is good for matrix operations

**Blocks needed:**
- Option A: (200+9)/10 × (100+19)/20 = 20 × 5 = 100 blocks
- Option B: (200+15)/16 × (100+15)/16 = 13 × 7 = 91 blocks

Both work, but **Option B has better warp efficiency**.

---

### Q17. How many blocks should you aim to have relative to the number of SMs on your GPU?

**Answer:** **C) Many more than the number of SMs**

**Reasoning:**

**Why not A (exactly equal):**
- When a block finishes, its SM sits idle
- No load balancing
- Wastes GPU resources

**Why not B (a few more):**
- Better than A, but still limited
- Some SMs may finish early and idle

**Why C (many more):**
- ✓ When blocks finish, new blocks immediately start
- ✓ Load balancing across SMs
- ✓ Hides execution time variability
- ✓ Keeps all SMs busy throughout execution
- ✓ Better resource utilization

**Rule of thumb:** Have at least 10-100× more blocks than SMs. For a GPU with 80 SMs:
- Minimum: ~800 blocks
- Better: 1,000-8,000 blocks
- Excellent: 10,000+ blocks

---

### Q18. For a 1D launch with 300,000 elements and 256 threads per block: `int blocksPerGrid = (300000 + 255) / 256;` What is `blocksPerGrid`?

**Answer:** **1,173 blocks**

**Calculation:**
```
(300000 + 255) / 256
= 300255 / 256
= 1172.87109375
= 1173 (integer division rounds down, but we've already added 255 to round up)
```

**Verification:** 1173 × 256 = 300,288 threads (enough to cover 300,000 elements) ✓

---

### Q19. In 2D configuration, what does `blockIdx.x` correspond to - rows or columns?

**Answer:** **Columns**

**Convention:**
- `blockIdx.x` / `threadIdx.x` → **Columns** (X dimension)
- `blockIdx.y` / `threadIdx.y` → **Rows** (Y dimension)

**In kernel:**
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;  // Y = rows
int col = blockIdx.x * blockDim.x + threadIdx.x;  // X = columns
```

This matches mathematical matrix notation: M[row][col] or M[y][x]

---

### Q20. You launch `kernel<<<dim3(10, 5), dim3(8, 8)>>>()`. How many total threads are launched?

**Answer:** **3,200 threads**

**Breakdown:**
- `dim3(10, 5)` = blocks per grid
  - 10 blocks in X direction
  - 5 blocks in Y direction
  - Total blocks: 10 × 5 = **50 blocks**
  
- `dim3(8, 8)` = threads per block
  - 8 threads in X direction
  - 8 threads in Y direction
  - Threads per block: 8 × 8 = **64 threads**
  
- **Total threads:** 50 blocks × 64 threads/block = **3,200 threads**

---

## Section 3: 2D Grid and Block Configuration

### Q21. For this kernel:
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```
If you're in Block(2, 3) and Thread(5, 7) with block size 16×16, what are `row` and `col`?

**Answer:**

**Given:**
- Block(2, 3) means: blockIdx.x = 2, blockIdx.y = 3
- Thread(5, 7) means: threadIdx.x = 5, threadIdx.y = 7
- Block size 16×16 means: blockDim.x = 16, blockDim.y = 16

**Calculations:**
```
row = blockIdx.y * blockDim.y + threadIdx.y
    = 3 × 16 + 7
    = 48 + 7
    = 55

col = blockIdx.x * blockDim.x + threadIdx.x
    = 2 × 16 + 5
    = 32 + 5
    = 37
```

**Answer: row = 55, col = 37**

This thread processes element [55][37] in the matrix.

---

### Q22. You have a 1000×500 matrix. You choose `dim3 threadsPerBlock(32, 32)`. Calculate:

```cuda
dim3 blocksPerGrid(
    _______,  // X dimension
    _______   // Y dimension
);
```

**Answer:**

**Given:**
- Matrix: 1000 rows × 500 columns
- Threads per block: 32×32

**Calculations:**
```cuda
// X dimension covers columns
blocksPerGrid.x = (cols + blockDim.x - 1) / blockDim.x
                = (500 + 32 - 1) / 32
                = 531 / 32
                = 16 blocks (with remainder, so use 17)
                = 17 blocks

// Y dimension covers rows
blocksPerGrid.y = (rows + blockDim.y - 1) / blockDim.y
                = (1000 + 32 - 1) / 32
                = 1031 / 32
                = 32 blocks (with remainder, so use 33)
                = 33 blocks
```

**Answer:**
```cuda
dim3 blocksPerGrid(17, 33);  // 17 blocks in X, 33 blocks in Y
```

**Verification:**
- Total blocks: 17 × 33 = 561 blocks
- Threads per block: 32 × 32 = 1,024 threads
- Total threads: 561 × 1,024 = 574,464 threads
- Elements needed: 1000 × 500 = 500,000 elements ✓

---

### Q23. Why might `dim3(16, 16)` be better than `dim3(32, 8)` for matrix operations, even though both = 256 threads?

**Answer:**

Both have the same number of threads (256) and warp efficiency (256/32 = 8 warps), but **16×16 is often better** for these reasons:

**1. Square tiles are more natural for matrices:**
- Matrix operations often work on square regions
- 16×16 tile matches square matrix structure better

**2. Memory access patterns:**
- Matrix operations access both rows and columns
- 16×16 provides balanced access in both dimensions
- 32×8 is heavily biased toward one dimension

**3. Shared memory tiling (for matmul):**
- Square tiles (16×16) load square regions from A and B
- More efficient shared memory usage
- Better data reuse patterns

**4. Flexibility:**
- 16×16 works well for both square and rectangular matrices
- 32×8 may waste threads for narrow matrices

**Example:** For matrix multiplication with tiling:
- 16×16 block loads 16×16 tile from A, 16×16 from B (symmetric, efficient)
- 32×8 block loads 32×8 from A, awkward tiling from B

**Bottom line:** For general matrix operations, square blocks are preferred. Use rectangular blocks only when you have a specific reason (like very rectangular matrices).

---

### Q24. Draw (or describe) how a 64×64 matrix is divided among blocks when using 16×16 thread blocks.

**Answer:**

**Setup:**
- Matrix: 64 × 64 elements
- Block size: 16 × 16 threads
- Each thread processes one element

**Grid calculation:**
```
Blocks in X: 64 / 16 = 4 blocks
Blocks in Y: 64 / 16 = 4 blocks
Total blocks: 4 × 4 = 16 blocks
```

**Visual representation:**

```
64×64 Matrix divided into 16×16 tiles
┌────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Block(0,0)   Block(1,0)   Block(2,0)   Block(3,0)            │
│  [0-15,       [16-31,      [32-47,      [48-63,                │
│   0-15]        0-15]        0-15]        0-15]                 │
│                                                                 │
│  Block(0,1)   Block(1,1)   Block(2,1)   Block(3,1)            │
│  [0-15,       [16-31,      [32-47,      [48-63,                │
│   16-31]       16-31]       16-31]       16-31]                │
│                                                                 │
│  Block(0,2)   Block(1,2)   Block(2,2)   Block(3,2)            │
│  [0-15,       [16-31,      [32-47,      [48-63,                │
│   32-47]       32-47]       32-47]       32-47]                │
│                                                                 │
│  Block(0,3)   Block(1,3)   Block(2,3)   Block(3,3)            │
│  [0-15,       [16-31,      [32-47,      [48-63,                │
│   48-63]       48-63]       48-63]       48-63]                │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

Each block contains 16×16 = 256 threads
Each thread processes exactly one matrix element
Total: 16 blocks × 256 threads = 4,096 threads for 4,096 elements (perfect!)
```

**Example - Block(1,2):**
- Covers rows 32-47, columns 16-31
- Thread(0,0) in this block → element[32][16]
- Thread(15,15) in this block → element[47][31]

---

### Q25. For a 75×150 matrix with 16×16 thread blocks:

**Answers:**

**Calculations:**
```
Matrix: 75 rows × 150 columns = 11,250 elements

Blocks in X: (150 + 15) / 16 = 165 / 16 = 10.3... = 11 blocks
Blocks in Y: (75 + 15) / 16 = 90 / 16 = 5.6... = 6 blocks
Total blocks: 11 × 6 = 66 blocks

Threads per block: 16 × 16 = 256 threads
Total threads launched: 66 × 256 = 16,896 threads
```

**Answers:**
- **Threads launched:** 66 blocks × 256 threads/block = **16,896 threads**
- **Elements needing processing:** 75 × 150 = **11,250 elements**
- **Threads "wasted":** 16,896 - 11,250 = **5,646 threads**
- **Percentage wasted:** 5,646 / 16,896 = **33.4%**

**Is this a problem?**

**No, this is NOT a problem because:**

1. **Perfect warp alignment:** 256 threads = 8 warps (100% efficient)
2. **Boundary check handles it:** Idle threads exit early via `if (row < 75 && col < 150)`
3. **Minimal overhead:** Idle threads consume almost no resources
4. **Alternative would be worse:** Using non-standard block sizes would hurt warp efficiency more than the wasted threads
5. **Standard practice:** This is how CUDA is designed to work

**Bad alternative:**
```cuda
// Trying to avoid waste with custom block size
dim3 threadsPerBlock(15, 15);  // 225 threads
// 225 / 32 = 7.03 warps → uses 8 warps, warp efficiency = 225/256 = 87.9%
// WORSE than 33% thread waste!
```

**Key principle:** Prioritize warp efficiency over minimizing idle threads.

---

### Q26. Complete this 2D matrix addition kernel:

```cuda
__global__ void matAdd(float *A, float *B, float *C, int rows, int cols) {
    int row = _________;
    int col = _________;
    
    if (___________) {
        int idx = _________;
        C[idx] = A[idx] + B[idx];
    }
}
```

**Answer:**

```cuda
__global__ void matAdd(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
```

**Explanation:**
1. Calculate global row/col position for this thread
2. Boundary check to handle edge cases where matrix dimensions aren't perfect multiples
3. Convert 2D (row, col) to 1D index for memory access
4. Perform the addition

---

### Q27. What's the difference between these two indexing approaches in the kernel?

```cuda
// Approach 1
C[row * cols + col] = A[row * cols + col] + B[row * cols + col];

// Approach 2
int idx = row * cols + col;
C[idx] = A[idx] + B[idx];
```

**Answer:**

**No functional difference** - both produce identical results. The difference is in **code efficiency and readability:**

**Approach 1:**
- Calculates `row * cols + col` **three times**
- Compiler may optimize this, but not guaranteed
- Less readable - harder to see what's happening

**Approach 2:**
- Calculates `row * cols + col` **once**
- Stores in variable and reuses
- More efficient (guaranteed single calculation)
- More readable - clear separation of index calculation and operation
- **Recommended approach**

**Performance impact:** Minor for simple operations, but good practice for:
- Complex index calculations
- Readability and maintainability
- Ensuring optimization

---

### Q28. Can you use 2D blocks with a 1D grid? Can you use 1D blocks with a 2D grid?

**Answer:** **Yes to both!**

Blocks and grids have **independent dimensionality**.

**2D blocks with 1D grid:**
```cuda
dim3 threadsPerBlock(16, 16);  // 2D block: 256 threads
int blocksPerGrid = 100;       // 1D grid: 100 blocks

kernel<<<blocksPerGrid, threadsPerBlock>>>();

// In kernel:
int tx = threadIdx.x;  // 0-15
int ty = threadIdx.y;  // 0-15
int bx = blockIdx.x;   // 0-99
// blockIdx.y and blockIdx.z are 0
```

**1D blocks with 2D grid:**
```cuda
int threadsPerBlock = 256;     // 1D block: 256 threads
dim3 blocksPerGrid(10, 10);    // 2D grid: 100 blocks

kernel<<<blocksPerGrid, threadsPerBlock>>>();

// In kernel:
int tx = threadIdx.x;  // 0-255
// threadIdx.y and threadIdx.z are 0
int bx = blockIdx.x;   // 0-9
int by = blockIdx.y;   // 0-9
```

**When to use each:**
- Matrix ops: Usually 2D blocks + 2D grid
- Image processing: Often 2D blocks + 2D grid
- Volume processing: 3D blocks + 3D grid
- Stream processing: 1D blocks + 1D grid
- Mixed: Any combination that fits your problem

---

## Section 4: Warps and Efficiency

### Q29. What is the warp size on NVIDIA GPUs?

**Answer:** **32 threads**

This is a hardware constant on all current NVIDIA GPUs. All threads in a warp execute the same instruction simultaneously (SIMT - Single Instruction, Multiple Threads).

---

### Q30. A block has 100 threads. How many warps does it actually use on the hardware?

**Answer:** **4 warps**

**Calculation:**
- 100 threads / 32 threads per warp = 3.125 warps
- Must use complete warps → rounds up to **4 warps**
- 4 warps × 32 threads = **128 thread slots**
- **28 thread slots are wasted** (idle)

**Warp efficiency:** 100 / 128 = 78.125%

This is why block sizes should be multiples of 32!

---

### Q31. Which block size has better warp efficiency and why?
- A) 100 threads
- B) 128 threads

**Answer:** **B) 128 threads**

**Analysis:**

**Option A (100 threads):**
- 100 / 32 = 3.125 warps → uses 4 warps
- Thread slots used: 4 × 32 = 128
- Wasted slots: 128 - 100 = **28 slots**
- **Warp efficiency: 100/128 = 78.125%**

**Option B (128 threads):**
- 128 / 32 = 4 warps (exact!)
- Thread slots used: 4 × 32 = 128
- Wasted slots: **0 slots**
- **Warp efficiency: 128/128 = 100%**

**Conclusion:** 128 is better because it's a multiple of 32, achieving perfect warp utilization.

---

### Q32. You launch 1,000,000 threads but your GPU has only 80 SMs with max 2048 threads per SM. Will your kernel work? Explain.

**Answer:** **Yes, the kernel will work perfectly!**

**Explanation:**

**GPU capacity:**
- 80 SMs × 2048 threads/SM = 163,840 threads can be **resident** (active) at once

**Your launch:**
- 1,000,000 threads total

**How it works:**
1. GPU schedules blocks to SMs
2. At any moment, ~163,840 threads are active
3. As blocks complete, new blocks are scheduled
4. Process continues until all 1,000,000 threads have executed
5. **Time-multiplexing:** Not all threads run simultaneously, but all will run eventually

**Example with 256 threads/block:**
```
Blocks needed: 1,000,000 / 256 = 3,907 blocks

Blocks resident at once: 163,840 / 256 = 640 blocks
(if each SM can hold 8 blocks: 80 SMs × 8 = 640 blocks)

Execution:
- First wave: 640 blocks execute
- As blocks finish: Next 640 blocks start
- Continue until all 3,907 blocks complete
```

This is the **beauty of CUDA**: You can launch millions/billions of threads, and the hardware efficiently schedules them!

---

### Q33. For a 60×60 matrix (3,600 elements):

```cuda
// Configuration A
dim3 threadsPerBlock(16, 16);  // 256 threads
dim3 blocksPerGrid(4, 4);      // 16 blocks, 4096 total threads

// Configuration B
dim3 threadsPerBlock(20, 20);  // 400 threads
dim3 blocksPerGrid(3, 3);      // 9 blocks, 3600 total threads
```

Which is better and why?

**Answer:** **Configuration A is better**

**Detailed Analysis:**

**Configuration A:**
- Threads per block: 16 × 16 = 256
- Warps per block: 256 / 32 = **8 warps (perfect!)**
- **Warp efficiency: 100%**
- Total threads: 4 × 4 × 256 = 4,096
- Wasted threads: 4,096 - 3,600 = 496 (13.7%)
- Number of blocks: **16 blocks**

**Configuration B:**
- Threads per block: 20 × 20 = 400
- Warps per block: 400
