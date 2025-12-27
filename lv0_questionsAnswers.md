# CUDA Fundamentals Questionnaire - Complete Answer Key
## To be taken after the understanding of CUDA fundamentals is covered. Vector addition and matrix addition programs are implemented.
### To be taken before matrix multiplication, profiling is covered.
### 27th Dec. 2025
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
- **Grid**: Logical organisation of all blocks for a kernel launch
- **Block**: A group of threads that can cooperate (shared memory, synchronisation)
- **Thread**: An individual execution unit that runs the kernel code
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

The grid is a **logical organisation** that exists only for the duration of a kernel launch. Physical components are:
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

### Q7. Can two threads from different blocks synchronise with each other using `__syncthreads()`?

**Answer:** **No**

`__syncthreads()` only synchronizes threads **within the same block**. Threads in different blocks:
- May run on different SMs
- May run at different times
- Cannot directly synchronise or communicate through shared memory

For cross-block synchronisation, you need:
- Multiple kernel launches
- Atomic operations on global memory
- Cooperative groups (advanced feature)

---

### Q8. What is a Streaming Multiprocessor (SM)?

**Answer:**

An SM is a **physical compute unit** on the GPU that:
- Executes blocks of threads
- Contains:
  - Multiple CUDA cores (32-128, depending on architecture)
  - Warp schedulers
  - Register file
  - Shared memory / L1 cache
  - Texture/constant cache
- Executes threads in groups of 32 (warps)
- Modern GPUs have 10-130+ SMs depending on model

**Example:** A GPU with 80 SMs can execute up to 80 blocks simultaneously (if each SM runs one block).

---

### Q9. Do you as a programme,r directly control which block runs on which SM?

**Answer:** **No**

The CUDA runtime and hardware scheduler automatically:
- Distribute blocks across available SMs
- Balance load dynamically
- Handle block scheduling as resources become available

The programmer only controls:
- Number of blocks and threads
- Resource usage (registers, shared memory)

---

### Q10. What is a warp, and why is it important?

**Answer:**

A **warp** is a group of **32 threads** that execute together on an SM.

**Why important:**
1. **Execution unit**: SMs execute warps, not individual threads
2. **SIMT execution**: All threads in a warp execute the same instruction simultaneously
3. **Efficiency**: For best performance, block size should be a multiple of 32
4. **Divergence**: If threads in a warp take different code paths (if/else), execution is serialised, reducing efficiency

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

**Additional issues if the limit was higher:**
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
- Standard, well-optimised size
- A square shape is good for matrix operations

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
- ✓ Better resource utilisation

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

### Q19. In a 2D configuration, what does `blockIdx.x` correspond to - rows or columns?

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
  - 5 blocks in the Y direction
  - Total blocks: 10 × 5 = **50 blocks**
  
- `dim3(8, 8)` = threads per block
  - 8 threads in X direction
  - 8 threads in the Y direction
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

**Key principle:** Prioritise warp efficiency over minimising idle threads.

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
- Compiler may optimise this, but not guaranteed
- Less readable - harder to see what's happening

**Approach 2:**
- Calculates `row * cols + col` **once**
- Stores in a variable and reuses
- More efficient (guaranteed single calculation)
- More readable - clear separation of index calculation and operation
- **Recommended approach**

**Performance impact:** Minor for simple operations, but good practice for:
- Complex index calculations
- Readability and maintainability
- Ensuring optimisation

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

**Conclusion:** 128 is better because it's a multiple of 32, achieving perfect warp utilisation.

---

### Q32. You launch 1,000,000 threads, but your GPU has only 80 SMs with max 2048 threads per SM. Will your kernel work? Explain.

**Answer:** **Yes, the kernel will work perfectly!**

**Explanation:**

**GPU capacity:**
- 80 SMs × 2048 threads/SM = 163,840 threads can be **resident** (active) at once

**Your launch:**
- 1,000,000 threads total

**How it works:**
1. GPU schedules blocks to SMs
2. At any moment, ~163,840 threads are active
3. As blocks are completed, new blocks are scheduled
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
- Continue until all 3,907 blocks are complete
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
- Warps per block: 400 / 32 = 12.5 → **uses 13 warps**
- Thread slots: 13 × 32 = 416
- Wasted warp slots: 416 - 400 = 16 per block
- **Warp efficiency: 400/416 = 96.15%**
- Total threads: 3 × 3 × 400 = 3,600 (perfect!)
- Wasted threads: 0
- Number of blocks: **9 blocks**

**Winner: Configuration A**

**Reasons:**
1. ✓ **100% warp efficiency** vs 96.15%
2. ✓ **More blocks (16 vs 9)** = better SM utilization
3. ✓ **Standard block size (256)** = well-optimized
4. ✗ More wasted threads (496 vs 0) - but this is insignificant

**Key insight:** Warp efficiency and block count are more important than minimising wasted threads!

---

### Q34. Why is having "many more blocks than SMs" a good thing?

**Answer:**

Having many more blocks than SMs provides several benefits:

**1. Load Balancing:**
- Blocks may take different amounts of time to complete
- With many blocks, when one finishes, another immediately starts
- Keeps all SMs busy throughout execution

**2. Latency Hiding:**
- While some warps wait for memory access, others can execute
- More blocks = more warps = better latency hiding
- Improves overall throughput

**3. Resource Flexibility:**
- Not all blocks use the same resources
- More blocks give the scheduler flexibility to pack blocks efficiently
- Can adapt to varying resource availability

**4. Avoiding Idle SMs:**
- With a few blocks, when they finish, SMs sit idle
- With many blocks, there's always work queued

**Example:**
```
GPU with 80 SMs:

Bad: 80 blocks (1 per SM)
- When blocks finish at different times, SMs go idle
- No work left to schedule

Good: 8,000 blocks (100 per SM)
- As blocks finish, new ones immediately start
- SMs stay busy until all work is done
- Better utilization
```

**Rule of thumb:** Aim for at least 10-100× more blocks than SMs.

---

### Q35. True or False: You should always try to minimise the number of "wasted" threads, even if it means using non-standard block sizes.

**Answer:** **False**

**Explanation:**

Prioritise **warp efficiency** and **standard block sizes** over minimising wasted threads.

**Why "minimising waste" is the wrong goal:**

1. **Warp inefficiency is worse than idle threads**
   - Block with 100 threads: 28 warp slots wasted (worse!)
   - Block with 128 threads: 0 warp slots wasted, but might have idle threads (better!)

2. **Idle threads have minimal cost**
   - They exit early via boundary check
   - Don't consume significant resources
   - GPU is designed to handle this efficiently

3. **Non-standard sizes cause problems**
   - Poor warp utilization
   - Less predictable performance
   - May not be well-optimised by the compiler

4. **Standard sizes are optimised**
   - 128, 256, 512 are tested and optimised
   - Libraries and tools expect these sizes
   - Better overall performance

**Example:**
```cuda
// Array with 10,000 elements

// BAD: Trying to minimise waste
int threadsPerBlock = 200;  // 10,000 / 50 blocks
// Warp efficiency: 200/224 = 89.3% (bad!)

// GOOD: Use standard size
int threadsPerBlock = 256;  // Need 40 blocks, 240 threads wasted
// Warp efficiency: 100% (good!)
// The 240 "wasted" threads are insignificant compared to warp loss
```

**Priority order:**
1. Warp alignment (multiple of 32)
2. Standard block sizes (128-512)
3. Enough blocks for SM utilisation
4. Minimising wasted threads (least important)

---

## Section 5: Conceptual Understanding

### Q36. Explain why matrix addition doesn't benefit much from a 2D vs 1D configuration, but matrix multiplication does.

**Answer:**

**Matrix Addition:**
```cuda
C[i] = A[i] + B[i];  // Each element is independent
```

- **Element-wise operation:** Each output element needs exactly 2 inputs
- **No cooperation:** Threads don't need to share data
- **No spatial locality matters:** Thread can be organized any way
- **Memory access:** Read 2 values, write 1 value (simple)
- **1D or 2D makes no difference** - just different ways to organize the same work

**Matrix Multiplication:**
```cuda
C[i][j] = sum(A[i][k] * B[k][j] for all k)  // Needs entire row and column
```

- **Complex operation:** Each output needs N inputs (entire row × column)
- **Cooperation helps:** Threads in a block can share data via shared memory
- **Spatial locality critical:** Nearby threads access nearby memory
- **2D structure enables tiling:**
  - Load 16×16 tile of A into shared memory
  - Load 16×16 tile of B into shared memory
  - All threads in the block reuse this data
  - Reduces global memory access by ~16×

**Optimisation example (matmul):**
```cuda
// 2D block structure enables this:
__shared__ float tileA[16][16];
__shared__ float tileB[16][16];

// Threads cooperate to load tiles
tileA[ty][tx] = A[...];  // Load collaboratively
__syncthreads();         // Wait for all

// All threads reuse shared data
for (k = 0; k < 16; k++)
    sum += tileA[ty][k] * tileB[k][tx];
```

This only makes sense with a 2D organisation!

**Summary:**
- **Addition:** No benefit from 2D (use either)
- **Multiplication:** Major benefit from 2D (required for optimization)

---

### Q37. What would happen if you forgot the boundary check in this kernel?

```cuda
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Missing: if (idx < N)
    data[idx] = data[idx] * 2.0f;
}
```

**Answer:**

**You would get undefined behavior and likely crashes!**

**What goes wrong:**

1. **Out-of-bounds memory access:**
   - If N=1000 and you launch 1024 threads (4 blocks × 256)
   - Threads 1000-1023 will access `data[1000]` to `data[1023]`
   - These indices are **beyond the allocated memory**

2. **Possible outcomes:**
   - **Best case:** Corrupt other data in your program
   - **Common case:** Segmentation fault / CUDA error
   - **Worst case:** Silent data corruption (hardest to debug!)

3. **CUDA error:**
   - May trigger `cudaErrorIllegalAddress`
   - Subsequent CUDA calls will fail
   - The program may crash

**Example:**
```cuda
float *data;
cudaMalloc(&data, 1000 * sizeof(float));  // Allocate 1000 elements

kernel<<<4, 256>>>(data, 1000);  // Launch 1024 threads

// Threads 1000-1023 access data[1000] to data[1023]
// These are OUTSIDE allocated memory!
// Result: Undefined behaviour
```

**Correct version:**
```cuda
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  // ← Essential boundary check
        data[idx] = data[idx] * 2.0f;
    }
}
```

**Why we over-launch threads:**
- Simplifies block calculation (ceiling division)
- Minimal overhead for idle threads
- But **must** have boundary check!

---

### Q38. A friend says, "I'll use 1024 threads per block because more threads = faster!" What would you tell them?

**Answer:**

**This is a common misconception!** More threads per block is NOT always better. Here's why:

**Problems with 1024 threads/block:**

1. **Limits blocks per SM:**
   - If kernel uses many registers: Maybe only 1 block fits per SM
   - With 256 threads/block: Could fit 4-8 blocks per SM
   - **Fewer blocks = worse occupancy**

2. **Resource constraints:**
   ```
   SM resources are limited:
   - Max ~65k registers per SM
   - Max ~48-96 KB shared memory per SM
   
   1024 threads/block:
   - Uses more registers/shared memory
   - Fewer blocks can run simultaneously
   - Lower occupancy
   ```

3. **Reduced flexibility:**
   - Small problems might only need 100 blocks
   - With 1024 threads: Only 100 blocks total
   - With 256 threads: Could have 400 blocks
   - More blocks = better SM utilization

4. **No performance gain for simple kernels:**
   - Simple kernels don't benefit from large blocks
   - Memory bandwidth, not block size, is the bottleneck

**When 1024 is appropriate:**
- Kernel needs extensive cooperation within the block
- Using lots of shared memory already (so can't fit many blocks anyway)
- Specific algorithm requires it

**Better advice:**
```cuda
// Usually optimal:
dim3 threadsPerBlock(16, 16);  // 256 threads
// or
dim3 threadsPerBlock(16, 16);  // 256 threads
// or
int threadsPerBlock = 512;     // Good middle ground

// Use 1024 only when you have a specific reason
```

**The truth:** **256-512 threads per block is the sweet spot** for most kernels!

---

### Q39. You need to process 1,000,000 elements. Your GPU has 80 SMs. Roughly how many blocks should you launch? Walk through your reasoning.

**Answer:**

**Step 1: Choose threads per block**
```
Use standard size: 256 threads/block
(Could also use 128, 512, but 256 is common)
```

**Step 2: Calculate blocks needed**
```
blocksPerGrid = (1,000,000 + 255) / 256
              = 1,000,255 / 256
              = 3,907 blocks
```

**Step 3: Analyse relative to SMs**
```
GPU has: 80 SMs
Blocks: 3,907
Ratio: 3,907 / 80 ≈ 49 blocks per SM
```

**Step 4: Is this good?**

**YES! This is excellent!**
- ✓ **49× more blocks than SMs** (great load balancing)
- ✓ As blocks finish, new ones immediately start
- ✓ All SMs stay busy throughout execution
- ✓ Good occupancy

**What if blocks per SM varied:**

```
Scenario A: Only 80 blocks total (1 per SM)
- Bad: When blocks finish, SMs sit idle
- No load balancing

Scenario B: 800 blocks (10 per SM)
- OK: Some load balancing
- Could be better

Scenario C: 3,907 blocks (49 per SM)
- Excellent: Great load balancing ✓
- SMs always have work

Scenario D: 80,000 blocks (1000 per SM)
- Excellent: Even better! (if problem size supports it)
```

**Rule of thumb:**
- Minimum: 2-3× blocks per SM
- Good: 10-50× blocks per SM  ← **We're here!**
- Excellent: 100× blocks per SM

**Final answer:** Launch approximately **3,900-4,000 blocks** (varies with threadsPerBlock choice). This provides ~49 blocks per SM, which is excellent for keeping the GPU busy!

---

### Q40. Explain in your own words: What's the relationship between blocks, SMs, and warps?

**Answer:**

**The Hierarchy:**

```
Physical Hardware (GPU)
└── Streaming Multiprocessor (SM) - Physical compute unit
    ├── Executes multiple blocks simultaneously (if resources allow)
    └── Block (Logical unit scheduled on SM)
        ├── Cannot span multiple SMs
        ├── Contains multiple warps
        └── Warp (Group of 32 threads)
            ├── Execution unit of the SM
            ├── All 32 threads execute the same instruction (SIMT)
            └── Thread (Individual execution context)
```

**How they relate:**

**1. Blocks → SMs (Scheduling)**
- GPU scheduler assigns blocks to SMs
- One block runs on exactly one SM (never split)
- One SM can run multiple blocks simultaneously
- As blocks finish, new blocks are scheduled

**2. Blocks → Warps (Composition)**
- Each block is divided into warps of 32 threads
- Block with 256 threads = 8 warps
- Block with 100 threads = 4 warps (with 28 wasted slots)

**3. SMs → Warps (Execution)**
- SMs execute warps, not individual threads
- Warp scheduler picks which warp executes each cycle
- Multiple warps per SM allow latency hiding
- While one warp waits for memory, another executes

**Concrete example:**

```
GPU Configuration:
- 80 SMs
- Each SM can hold up to 2048 threads and 16 blocks

Kernel launch: <<<1000, 256>>>
- 1000 blocks total
- 256 threads per block = 8 warps per block

Execution:
1. Scheduler distributes 1000 blocks across 80 SMs
2. Each SM might run 8-12 blocks simultaneously (if resources allow)
3. Each block's 8 warps are scheduled by that SM's warp scheduler
4. At each cycle, SM executes 1-2 warps (depends on architecture)
5. As blocks are completed, new blocks from the 1000 are scheduled
```

**Key relationships:**
- **Blocks organize threads** (programmer's view)
- **SMs execute blocks** (hardware scheduling)
- **Warps are the execution unit** (hardware execution)

**Memory hierarchy connection:**
- Registers: Per-thread
- Shared memory: Per-block (all threads in the block can access)
- L1 cache: Per-SM
- L2 cache: Shared across all SMs
- Global memory: Shared across the entire GPU

---

## Bonus Challenge Questions

### Q41. Debug this code:

```cuda
int N = 1000;
int threadsPerBlock = 256;
int blocksPerGrid = N / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerGrid>>>(data, N);
```

**Answer:**

**Bug:** `blocksPerGrid = N / threadsPerBlock` uses integer division without ceiling.

**What goes wrong:**
```
N / threadsPerBlock = 1000 / 256 = 3.90625
Integer division: = 3 blocks

3 blocks × 256 threads = 768 threads
But we need 1000 threads!
Elements 768-999 are NOT processed! 
```

**Fixed code:**
```cuda
int N = 1000;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // = 4 blocks
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);

// Now: 4 blocks × 256 threads = 1024 threads ✓
// Threads 1000-1023 exit via boundary check
```

**In kernel, must have:**
```cuda
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  // Essential!
        data[idx] = /* ... */;
    }
}
```

**The fix:** Always use ceiling division: `(N + threadsPerBlock - 1) / threadsPerBlock`

---

### Q42. You have a 1024×2048 matrix. Design a complete launch configuration (include both 1D and 2D approaches) and explain which you'd prefer.

**Answer:**

**Matrix dimensions:** 1024 rows × 2048 columns = 2,097,152 elements

---

**Approach 1: 1D Configuration**

```cuda
// Treat as flat array
int N = 1024 * 2048;  // 2,097,152 elements
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
// = (2097152 + 255) / 256 = 8193 blocks

kernel_1D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

// Kernel:
__global__ void kernel_1D(float *A, float *B, float *C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = rows * cols;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // For addition
    }
}
```

**Stats:**
- Blocks: 8,193
- Threads per block: 256
- Total threads: 2,097,408
- Wasted threads: 256

---

**Approach 2: 2D Configuration**

```cuda
// Treat as 2D matrix
dim3 threadsPerBlock(16, 16);  // 256 threads
dim3 blocksPerGrid(
    (2048 + 15) / 16,  // Columns: 2063/16 = 129 blocks
    (1024 + 15) / 16   // Rows: 1039/16 = 65 blocks
);
// Total: 129 × 65 = 8,385 blocks

kernel_2D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

// Kernel:
__global__ void kernel_2D(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];  // For addition
    }
}
```

**Stats:**
- Blocks: 8,385 (129 × 65)
- Threads per block: 256
- Total threads: 2,146,560
- Wasted threads: 49,408 (2.3%)

---

**Alternative 2D: Using 32×8 blocks**

```cuda
dim3 threadsPerBlock(32, 8);  // 256 threads
dim3 blocksPerGrid(
    (2048 + 31) / 32,  // 2079/32 = 65 blocks
    (1024 + 7) / 8     // 1031/8 = 129 blocks
);
// Total: 65 × 129 = 8,385 blocks (same as before)
```

---

**Which do I prefer?**

**For matrix addition: 1D approach**
- ✓ Simpler code
- ✓ Simpler launch configuration
- ✓ Fewer wasted threads (256 vs 49,408)
- ✓ No conceptual benefit from 2D structure
- Performance: Essentially identical

**For matrix multiplication: 2D approach (16×16)**
- ✓ Required for tiling optimisation
- ✓ Natural structure for shared memory
- ✓ Square blocks work well for matrix tiles
- ✓ Enables thread cooperation
- Performance: Much better with optimisation

**For this specific problem (1024×2048):**

**If just doing addition:** Use **1D**
```cuda
kernel_1D<<<8193, 256>>>(d_A, d_B, d_C, 1024, 2048);
```

**If doing multiplication or other complex ops:** Use **2D with 16×16 blocks**
```cuda
kernel_2D<<<dim3(128, 64), dim3(16, 16)>>>(d_A, d_B, d_C, 1024, 2048);
```

**My choice for general matrix code:** 2D with 16×16 blocks - it's the standard, works for all matrix operations, and is slightly more readable for matrix code, even if performance is identical for simple operations.

---

### Q43. A kernel processes 100,000 elements. You profile it and find that only 30% of your GPU's SMs are active. What might be wrong,g and how would you investigate?

**Answer:**

**Problem:** Only 30% SM utilisation means 70% of the GPU is idle - very inefficient!

---

**Possible Causes:**

**1. Too few blocks launched**

```cuda
// BAD: Only 24 blocks
kernel<<<24, 4096>>>(data, 100000);  
// Won't work - 4096 > 1024 max!

// BAD: Only 100 blocks
kernel<<<100, 1000>>>(data, 100000);
// On 80-SM GPU, many SMs are idle

// GOOD: Many blocks
kernel<<<391, 256>>>(data, 100000);
// 391 blocks keep all SMs busy
```

**2. Resource exhaustion per block**

```cuda
__global__ void resourceHungryKernel() {
    // Uses many registers
    double temp[100];  // 800 bytes of registers per thread!
    
    // Uses lots of shared memory
    __shared__ float data[10000];  // 40 KB per block
    
    // Result: Only 1-2 blocks can fit per SM
    // → Low occupancy
}
```

**3. Long-running blocks with uneven workload**

```cuda
__global__ void unevenKernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Some blocks do lots of work
    if (idx < 1000) {
        for (int i = 0; i < 1000000; i++) { /* heavy */ }
    }
    // Other blocks do little work
    else {
        data[idx] = idx;  // Quick!
    }
    // Result: Some SMs finish early and sit idle
}
```

**4. Synchronization issues**

```cuda
// Multiple kernel launches with small work
for (int i = 0; i < 1000; i++) {
    smallKernel<<<10, 256>>>(data);  // Launches only 10 blocks each time
    cudaDeviceSynchronize();  // Waits for completion
}
// Better: Launch once with 10,000 blocks
```

---

**How to Investigate:**

**1. Check launch configuration**

```cuda
// Add diagnostic prints
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
printf("Launching %d blocks with %d threads each\n", blocksPerGrid, threadsPerBlock);
printf("Total threads: %d\n", blocksPerGrid * threadsPerBlock);
printf("Elements to process: %d\n", N);
```

**2. Use NVIDIA profiler**

```bash
# Check occupancy
nvprof --metrics achieved_occupancy ./program

# Check SM utilisation
nvprof --metrics sm_efficiency ./program

# See detailed metrics
nvprof --print-gpu-trace ./program
```

**3. Check resource usage at compile time**

```bash
nvcc --ptxas-options=-v kernel.cu

# Output shows:
# - Registers used per thread
# - Shared memory used per block
# - Estimated occupancy
```

**4. Use Nsight Compute for detailed analysis**

```bash
ncu --set full -o profile ./program
# Opens GUI with detailed metrics:
# - Warp execution efficiency
# - Memory throughput
# - SM occupancy over time
```

**5. Check for errors**

```cuda
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
```

---

**Solutions based on cause:**

**If too few blocks:**
```cuda
// Increase blocks, decrease threads per block
kernel<<<391, 256>>>(data, 100000);  // Instead of <<<100, 1000>>>
```

**If resource exhaustion:**
```cuda
// Reduce resource usage
__global__ void optimizedKernel() {
    // Use fewer local variables
    float temp;  // Instead of temp[100]
    
    // Use smaller shared memory
    __shared__ float data[1024];  // Instead of [10000]
}
```

**If uneven workload:**
```cuda
// Balance work distribution
// Use more blocks with smaller chunks
// Or use dynamic parallelism for irregular work
```

**Specific investigation for 100,000 elements:**

```cuda
// What you might have (BAD):
kernel<<<100, 1000>>>(data, 100000);
// Issue: Only 100 blocks on 80-SM GPU
// Some SMs get 2 blocks, some get 1, many get 0

// What you should have (GOOD):
int threadsPerBlock = 256;
int blocksPerGrid = (100000 + 255) / 256;  // = 391 blocks
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, 100000);
// Now: 391 blocks across 80 SMs ≈ 5 blocks per SM
// All SMs stay busy!
```

---

## Summary: Key Formulas to Memorize

**1D Launch:**
```cuda
int threadsPerBlock = 256;  // Choose: 128, 256, or 512
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);
```

**2D Launch:**
```cuda
dim3 threadsPerBlock(16, 16);  // 256 threads
dim3 blocksPerGrid(
    (cols + 15) / 16,  // X dimension
    (rows + 15) / 16   // Y dimension
);
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, rows, cols);
```

**Global Index Calculations:**
```cuda
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * cols + col;
```

**Key Principles:**
1. Always use ceiling division for blocks
2. Block size should be a# CUDA Fundamentals Questionnaire - Complete Answer Key

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
- Warps per block: 400 / 32 = 12.5 → **uses 13 warps**
- Thread slots: 13 × 32 = 416
- Wasted warp slots: 416 - 400 = 16 per block
- **Warp efficiency: 400/416 = 96.15%**
- Total threads: 3 × 3 × 400 = 3,600 (perfect!)
- Wasted threads: 0
- Number of blocks: **9 blocks**

**Winner: Configuration A**

**Reasons:**
1. ✓ **100% warp efficiency** vs 96.15%
2. ✓ **More blocks (16 vs 9)** = better SM utilization
3. ✓ **Standard block size (256)** = well-optimized
4. ✗ More wasted threads (496 vs 0) - but this is insignificant

**Key insight:** Warp efficiency and block count are more important than minimizing wasted threads!

---

### Q34. Why is having "many more blocks than SMs" a good thing?

**Answer:**

Having many more blocks than SMs provides several benefits:

**1. Load Balancing:**
- Blocks may take different amounts of time to complete
- With many blocks, when one finishes, another immediately starts
- Keeps all SMs busy throughout execution

**2. Latency Hiding:**
- While some warps wait for memory access, others can execute
- More blocks = more warps = better latency hiding
- Improves overall throughput

**3. Resource Flexibility:**
- Not all blocks use the same resources
- More blocks gives the scheduler flexibility to pack blocks efficiently
- Can adapt to varying resource availability

**4. Avoiding Idle SMs:**
- With few blocks, when they finish, SMs sit idle
- With many blocks, there's always work queued

**Example:**
```
GPU with 80 SMs:

Bad: 80 blocks (1 per SM)
- When blocks finish at different times, SMs go idle
- No work left to schedule

Good: 8,000 blocks (100 per SM)
- As blocks finish, new ones immediately start
- SMs stay busy until all work is done
- Better utilization
```

**Rule of thumb:** Aim for at least 10-100× more blocks than SMs.

---

### Q35. True or False: You should always try to minimize the number of "wasted" threads, even if it means using non-standard block sizes.

**Answer:** **False**

**Explanation:**

Prioritize **warp efficiency** and **standard block sizes** over minimizing wasted threads.

**Why "minimizing waste" is the wrong goal:**

1. **Warp inefficiency is worse than idle threads**
   - Block with 100 threads: 28 warp slots wasted (worse!)
   - Block with 128 threads: 0 warp slots wasted, but might have idle threads (better!)

2. **Idle threads have minimal cost**
   - They exit early via boundary check
   - Don't consume significant resources
   - GPU is designed to handle this efficiently

3. **Non-standard sizes cause problems**
   - Poor warp utilization
   - Less predictable performance
   - May not be well-optimized by compiler

4. **Standard sizes are optimized**
   - 128, 256, 512 are tested and optimized
   - Libraries and tools expect these sizes
   - Better overall performance

**Example:**
```cuda
// Array with 10,000 elements

// BAD: Trying to minimize waste
int threadsPerBlock = 200;  // 10,000 / 50 blocks
// Warp efficiency: 200/224 = 89.3% (bad!)

// GOOD: Use standard size
int threadsPerBlock = 256;  // Need 40 blocks, 240 threads wasted
// Warp efficiency: 100% (good!)
// The 240 "wasted" threads are insignificant compared to warp loss
```

**Priority order:**
1. Warp alignment (multiple of 32)
2. Standard block sizes (128-512)
3. Enough blocks for SM utilization
4. Minimizing wasted threads (least important)

---

## Section 5: Conceptual Understanding

### Q36. Explain why matrix addition doesn't benefit much from 2D vs 1D configuration, but matrix multiplication does.

**Answer:**

**Matrix Addition:**
```cuda
C[i] = A[i] + B[i];  // Each element is independent
```

- **Element-wise operation:** Each output element needs exactly 2 inputs
- **No cooperation:** Threads don't need to share data
- **No spatial locality matters:** Thread can be organized any way
- **Memory access:** Read 2 values, write 1 value (simple)
- **1D or 2D makes no difference** - just different ways to organize the same work

**Matrix Multiplication:**
```cuda
C[i][j] = sum(A[i][k] * B[k][j] for all k)  // Needs entire row and column
```

- **Complex operation:** Each output needs N inputs (entire row × column)
- **Cooperation helps:** Threads in a block can share data via shared memory
- **Spatial locality critical:** Nearby threads access nearby memory
- **2D structure enables tiling:**
  - Load 16×16 tile of A into shared memory
  - Load 16×16 tile of B into shared memory
  - All threads in block reuse this data
  - Reduces global memory access by ~16×

**Optimization example (matmul):**
```cuda
// 2D block structure enables this:
__shared__ float tileA[16][16];
__shared__ float tileB[16][16];

// Threads cooperate to load tiles
tileA[ty][tx] = A[...];  // Load collaboratively
__syncthreads();         // Wait for all

// All threads reuse shared data
for (k = 0; k < 16; k++)
    sum += tileA[ty][k] * tileB[k][tx];
```

This only makes sense with 2D organization!

**Summary:**
- **Addition:** No benefit from 2D (use either)
- **Multiplication:** Major benefit from 2D (required for optimization)

---

### Q37. What would happen if you forgot the boundary check in this kernel?

```cuda
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Missing: if (idx < N)
    data[idx] = data[idx] * 2.0f;
}
```

**Answer:**

**You would get undefined behavior and likely crashes!**

**What goes wrong:**

1. **Out-of-bounds memory access:**
   - If N=1000 and you launch 1024 threads (4 blocks × 256)
   - Threads 1000-1023 will access `data[1000]` to `data[1023]`
   - These indices are **beyond the allocated memory**

2. **Possible outcomes:**
   - **Best case:** Corrupt other data in your program
   - **Common case:** Segmentation fault / CUDA error
   - **Worst case:** Silent data corruption (hardest to debug!)

3. **CUDA error:**
   - May trigger `cudaErrorIllegalAddress`
   - Subsequent CUDA calls will fail
   - Program may crash

**Example:**
```cuda
float *data;
cudaMalloc(&data, 1000 * sizeof(float));  // Allocate 1000 elements

kernel<<<4, 256>>>(data, 1000);  // Launch 1024 threads

// Threads 1000-1023 access data[1000] to data[1023]
// These are OUTSIDE allocated memory!
// Result: Undefined behavior
```

**Correct version:**
```cuda
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  // ← Essential boundary check
        data[idx] = data[idx] * 2.0f;
    }
}
```

**Why we over-launch threads:**
- Simplifies block calculation (ceiling division)
- Minimal overhead for idle threads
- But **must** have boundary check!

---

### Q38. A friend says: "I'll use 1024 threads per block because more threads = faster!" What would you tell them?

**Answer:**

**This is a common misconception!** More threads per block is NOT always better. Here's why:

**Problems with 1024 threads/block:**

1. **Limits blocks per SM:**
   - If kernel uses many registers: Maybe only 1 block fits per SM
   - With 256 threads/block: Could fit 4-8 blocks per SM
   - **Fewer blocks = worse occupancy**

2. **Resource constraints:**
   ```
   SM resources are limited:
   - Max ~65k registers per SM
   - Max ~48-96 KB shared memory per SM
   
   1024 threads/block:
   - Uses more registers/shared memory
   - Fewer blocks can run simultaneously
   - Lower occupancy
   ```

3. **Reduced flexibility:**
   - Small problems might only need 100 blocks
   - With 1024 threads: Only 100 blocks total
   - With 256 threads: Could have 400 blocks
   - More blocks = better SM utilization

4. **No performance gain for simple kernels:**
   - Simple kernels don't benefit from large blocks
   - Memory bandwidth, not block size, is the bottleneck

**When 1024 is appropriate:**
- Kernel needs extensive cooperation within block
- Using lots of shared memory already (so can't fit many blocks anyway)
- Specific algorithm requires it

**Better advice:**
```cuda
// Usually optimal:
dim3 threadsPerBlock(16, 16);  // 256 threads
// or
dim3 threadsPerBlock(16, 16);  // 256 threads
// or
int threadsPerBlock = 512;     // Good middle ground

// Use 1024 only when you have a specific reason
```

**The truth:** **256-512 threads per block is the sweet spot** for most kernels!

---

### Q39. You need to process 1,000,000 elements. Your GPU has 80 SMs. Roughly how many blocks should you launch? Walk through your reasoning.

**Answer:**

**Step 1: Choose threads per block**
```
Use standard size: 256 threads/block
(Could also use 128, 512, but 256 is common)
```

**Step 2: Calculate blocks needed**
```
blocksPerGrid = (1,000,000 + 255) / 256
              = 1,000,255 / 256
              = 3,907 blocks
```

**Step 3: Analyze relative to SMs**
```
GPU has: 80 SMs
Blocks: 3,907
Ratio: 3,907 / 80 ≈ 49 blocks per SM
```

**Step 4: Is this good?**

**YES! This is excellent!**
- ✓ **49× more blocks than SMs** (great load balancing)
- ✓ As blocks finish, new ones immediately start
- ✓ All SMs stay busy throughout execution
- ✓ Good occupancy

**What if blocks per SM varied:**

```
Scenario A: Only 80 blocks total (1 per SM)
- Bad: When blocks finish, SMs sit idle
- No load balancing

Scenario B: 800 blocks (10 per SM)
- OK: Some load balancing
- Could be better

Scenario C: 3,907 blocks (49 per SM)
- Excellent: Great load balancing ✓
- SMs always have work

Scenario D: 80,000 blocks (1000 per SM)
- Excellent: Even better! (if problem size supports it)
```

**Rule of thumb:**
- Minimum: 2-3× blocks per SM
- Good: 10-50× blocks per SM  ← **We're here!**
- Excellent: 100× blocks per SM

**Final answer:** Launch approximately **3,900-4,000 blocks** (varies with threadsPerBlock choice). This provides ~49 blocks per SM, which is excellent for keeping the GPU busy!

---

### Q40. Explain in your own words: What's the relationship between blocks, SMs, and warps?

**Answer:**

**The Hierarchy:**

```
Physical Hardware (GPU)
└── Streaming Multiprocessor (SM) - Physical compute unit
    ├── Executes multiple blocks simultaneously (if resources allow)
    └── Block (Logical unit scheduled on SM)
        ├── Cannot span multiple SMs
        ├── Contains multiple warps
        └── Warp (Group of 32 threads)
            ├── Execution unit of the SM
            ├── All 32 threads execute same instruction (SIMT)
            └── Thread (Individual execution context)
```

**How they relate:**

**1. Blocks → SMs (Scheduling)**
- GPU scheduler assigns blocks to SMs
- One block runs on exactly one SM (never split)
- One SM can run multiple blocks simultaneously
- As blocks finish, new blocks are scheduled

**2. Blocks → Warps (Composition)**
- Each block is divided into warps of 32 threads
- Block with 256 threads = 8 warps
- Block with 100 threads = 4 warps (with 28 wasted slots)

**3. SMs → Warps (Execution)**
- SMs execute warps, not individual threads
- Warp scheduler picks which warp executes each cycle
- Multiple warps per SM allow latency hiding
- While one warp waits for memory, another executes

**Concrete example:**

```
GPU Configuration:
- 80 SMs
- Each SM can hold up to 2048 threads and 16 blocks

Kernel launch: <<<1000, 256>>>
- 1000 blocks total
- 256 threads per block = 8 warps per block

Execution:
1. Scheduler distributes 1000 blocks across 80 SMs
2. Each SM might run 8-12 blocks simultaneously (if resources allow)
3. Each block's 8 warps are scheduled by that SM's warp scheduler
4. At each cycle, SM executes 1-2 warps (depends on architecture)
5. As blocks complete, new blocks from the 1000 are scheduled
```

**Key relationships:**
- **Blocks organize threads** (programmer's view)
- **SMs execute blocks** (hardware scheduling)
- **Warps are the execution unit** (hardware execution)

**Memory hierarchy connection:**
- Registers: Per-thread
- Shared memory: Per-block (all threads in block can access)
- L1 cache: Per-SM
- L2 cache: Shared across all SMs
- Global memory: Shared across entire GPU

---

## Bonus Challenge Questions

### Q41. Debug this code:

```cuda
int N = 1000;
int threadsPerBlock = 256;
int blocksPerGrid = N / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerGrid>>>(data, N);
```

**Answer:**

**Bug:** `blocksPerGrid = N / threadsPerBlock` uses integer division without ceiling.

**What goes wrong:**
```
N / threadsPerBlock = 1000 / 256 = 3.90625
Integer division: = 3 blocks

3 blocks × 256 threads = 768 threads
But we need 1000 threads!
Elements 768-999 are NOT processed! ❌
```

**Fixed code:**
```cuda
int N = 1000;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // = 4 blocks
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);

// Now: 4 blocks × 256 threads = 1024 threads ✓
// Threads 1000-1023 exit via boundary check
```

**In kernel, must have:**
```cuda
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  // Essential!
        data[idx] = /* ... */;
    }
}
```

**The fix:** Always use ceiling division: `(N + threadsPerBlock - 1) / threadsPerBlock`

---

### Q42. You have a 1024×2048 matrix. Design a complete launch configuration (include both 1D and 2D approaches) and explain which you'd prefer.

**Answer:**

**Matrix dimensions:** 1024 rows × 2048 columns = 2,097,152 elements

---

**Approach 1: 1D Configuration**

```cuda
// Treat as flat array
int N = 1024 * 2048;  // 2,097,152 elements
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
// = (2097152 + 255) / 256 = 8193 blocks

kernel_1D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

// Kernel:
__global__ void kernel_1D(float *A, float *B, float *C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = rows * cols;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // For addition
    }
}
```

**Stats:**
- Blocks: 8,193
- Threads per block: 256
- Total threads: 2,097,408
- Wasted threads: 256

---

**Approach 2: 2D Configuration**

```cuda
// Treat as 2D matrix
dim3 threadsPerBlock(16, 16);  // 256 threads
dim3 blocksPerGrid(
    (2048 + 15) / 16,  // Columns: 2063/16 = 129 blocks
    (1024 + 15) / 16   // Rows: 1039/16 = 65 blocks
);
// Total: 129 × 65 = 8,385 blocks

kernel_2D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

// Kernel:
__global__ void kernel_2D(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];  // For addition
    }
}
```

**Stats:**
- Blocks: 8,385 (129 × 65)
- Threads per block: 256
- Total threads: 2,146,560
- Wasted threads: 49,408 (2.3%)

---

**Alternative 2D: Using 32×8 blocks**

```cuda
dim3 threadsPerBlock(32, 8);  // 256 threads
dim3 blocksPerGrid(
    (2048 + 31) / 32,  // 2079/32 = 65 blocks
    (1024 + 7) / 8     // 1031/8 = 129 blocks
);
// Total: 65 × 129 = 8,385 blocks (same as before)
```

---

**Which do I prefer?**

**For matrix addition: 1D approach**
- ✓ Simpler code
- ✓ Simpler launch configuration
- ✓ Fewer wasted threads (256 vs 49,408)
- ✓ No conceptual benefit from 2D structure
- Performance: Essentially identical

**For matrix multiplication: 2D approach (16×16)**
- ✓ Required for tiling optimization
- ✓ Natural structure for shared memory
- ✓ Square blocks work well for matrix tiles
- ✓ Enables thread cooperation
- Performance: Much better with optimization

**For this specific problem (1024×2048):**

**If just doing addition:** Use **1D**
```cuda
kernel_1D<<<8193, 256>>>(d_A, d_B, d_C, 1024, 2048);
```

**If doing multiplication or other complex ops:** Use **2D with 16×16 blocks**
```cuda
kernel_2D<<<dim3(128, 64), dim3(16, 16)>>>(d_A, d_B, d_C, 1024, 2048);
```

**My choice for general matrix code:** 2D with 16×16 blocks - it's the standard, works for all matrix operations, and is slightly more readable for matrix code even if performance is identical for simple operations.

---

### Q43. A kernel processes 100,000 elements. You profile it and find only 30% of your GPU's SMs are active. What might be wrong and how would you investigate?

**Answer:**

**Problem:** Only 30% SM utilization means 70% of GPU is idle - very inefficient!

---

**Possible Causes:**

**1. Too few blocks launched**

```cuda
// BAD: Only 24 blocks
kernel<<<24, 4096>>>(data, 100000);  
// Won't work - 4096 > 1024 max!

// BAD: Only 100 blocks
kernel<<<100, 1000>>>(data, 100000);
// On 80-SM GPU, many SMs idle

// GOOD: Many blocks
kernel<<<391, 256>>>(data, 100000);
// 391 blocks keeps all SMs busy
```

**2. Resource exhaustion per block**

```cuda
__global__ void resourceHungryKernel() {
    // Uses many registers
    double temp[100];  // 800 bytes of registers per thread!
    
    // Uses lots of shared memory
    __shared__ float data[10000];  // 40 KB per block
    
    // Result: Only 1-2 blocks can fit per SM
    // → Low occupancy
}
```

**3. Long-running blocks with uneven workload**

```cuda
__global__ void unevenKernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Some blocks do lots of work
    if (idx < 1000) {
        for (int i = 0; i < 1000000; i++) { /* heavy */ }
    }
    // Other blocks do little work
    else {
        data[idx] = idx;  // Quick!
    }
    // Result: Some SMs finish early and sit idle
}
```

**4. Synchronization issues**

```cuda
// Multiple kernel launches with small work
for (int i = 0; i < 1000; i++) {
    smallKernel<<<10, 256>>>(data);  // Launches only 10 blocks each time
    cudaDeviceSynchronize();  // Waits for completion
}
// Better: Launch once with 10,000 blocks
```

---

**How to Investigate:**

**1. Check launch configuration**

```cuda
// Add diagnostic prints
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
printf("Launching %d blocks with %d threads each\n", blocksPerGrid, threadsPerBlock);
printf("Total threads: %d\n", blocksPerGrid * threadsPerBlock);
printf("Elements to process: %d\n", N);
```

**2. Use NVIDIA profiler**

```bash
# Check occupancy
nvprof --metrics achieved_occupancy ./program

# Check SM utilization
nvprof --metrics sm_efficiency ./program

# See detailed metrics
nvprof --print-gpu-trace ./program
```

**3. Check resource usage at compile time**

```bash
nvcc --ptxas-options=-v kernel.cu

# Output shows:
# - Registers used per thread
# - Shared memory used per block
# - Estimated occupancy
```

**4. Use Nsight Compute for detailed analysis**

```bash
ncu --set full -o profile ./program
# Opens GUI with detailed metrics:
# - Warp execution efficiency
# - Memory throughput
# - SM occupancy over time
```

**5. Check for errors**

```cuda
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
```

---

**Solutions based on cause:**

**If too few blocks:**
```cuda
// Increase blocks, decrease threads per block
kernel<<<391, 256>>>(data, 100000);  // Instead of <<<100, 1000>>>
```

**If resource exhaustion:**
```cuda
// Reduce resource usage
__global__ void optimizedKernel() {
    // Use fewer local variables
    float temp;  // Instead of temp[100]
    
    // Use smaller shared memory
    __shared__ float data[1024];  // Instead of [10000]
}
```

**If uneven workload:**
```cuda
// Balance work distribution
// Use more blocks with smaller chunks
// Or use dynamic parallelism for irregular work
```

**Specific investigation for 100,000 elements:**

```cuda
// What you might have (BAD):
kernel<<<100, 1000>>>(data, 100000);
// Issue: Only 100 blocks on 80-SM GPU
// Some SMs get 2 blocks, some get 1, many get 0

// What you should have (GOOD):
int threadsPerBlock = 256;
int blocksPerGrid = (100000 + 255) / 256;  // = 391 blocks
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, 100000);
// Now: 391 blocks across 80 SMs ≈ 5 blocks per SM
// All SMs stay busy!
```

---

## Summary: Key Formulas to Memorize

**1D Launch:**
```cuda
int threadsPerBlock = 256;  // Choose: 128, 256, or 512
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);
```

**2D Launch:**
```cuda
dim3 threadsPerBlock(16, 16);  // 256 threads
dim3 blocksPerGrid(
    (cols + 15) / 16,  // X dimension
    (rows + 15) / 16   // Y dimension
);
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, rows, cols);
```

**Global Index Calculations:**
```cuda
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * cols + col;
```

**Key Principles:**
1. Always use ceiling division for blocks
2. Block size should be multiple of 32 (warp size)
3. Aim for 256-512 threads per block
4. Launch many more blocks than SMs (10-100×)
5. Always include boundary checks in kernels
6. Don't worry about "wasted" threads if warps are efficient

---

## Grading Rubric

- **Section 1 (Basic Concepts):** /10 points
- **Section 2 (Launch Configuration):** /10 points
- **Section 3 (2D Configuration):** /8 points
- **Section 4 (Warps and Efficiency):** /7 points
- **Section 5 (Conceptual):** /5 points
- **Bonus (Challenge):** /3 points

**Total:** /43 points

**Grade Scale:**
- 40-43: A+ (Excellent)
- 35-39: A (Very Good)
- 30-34: B (Good)
- 25-29: C (Satisfactory)
- 20-24: D (Needs Improvement)
- <20: F (Significant gaps)

---

*End of Questionnaire*# CUDA Fundamentals Questionnaire - Complete Answer Key

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
- Warps per block: 400 / 32 = 12.5 → **uses 13 warps**
- Thread slots: 13 × 32 = 416
- Wasted warp slots: 416 - 400 = 16 per block
- **Warp efficiency: 400/416 = 96.15%**
- Total threads: 3 × 3 × 400 = 3,600 (perfect!)
- Wasted threads: 0
- Number of blocks: **9 blocks**

**Winner: Configuration A**

**Reasons:**
1. ✓ **100% warp efficiency** vs 96.15%
2. ✓ **More blocks (16 vs 9)** = better SM utilization
3. ✓ **Standard block size (256)** = well-optimized
4. ✗ More wasted threads (496 vs 0) - but this is insignificant

**Key insight:** Warp efficiency and block count are more important than minimizing wasted threads!

---

### Q34. Why is having "many more blocks than SMs" a good thing?

**Answer:**

Having many more blocks than SMs provides several benefits:

**1. Load Balancing:**
- Blocks may take different amounts of time to complete
- With many blocks, when one finishes, another immediately starts
- Keeps all SMs busy throughout execution

**2. Latency Hiding:**
- While some warps wait for memory access, others can execute
- More blocks = more warps = better latency hiding
- Improves overall throughput

**3. Resource Flexibility:**
- Not all blocks use the same resources
- More blocks gives the scheduler flexibility to pack blocks efficiently
- Can adapt to varying resource availability

**4. Avoiding Idle SMs:**
- With few blocks, when they finish, SMs sit idle
- With many blocks, there's always work queued

**Example:**
```
GPU with 80 SMs:

Bad: 80 blocks (1 per SM)
- When blocks finish at different times, SMs go idle
- No work left to schedule

Good: 8,000 blocks (100 per SM)
- As blocks finish, new ones immediately start
- SMs stay busy until all work is done
- Better utilization
```

**Rule of thumb:** Aim for at least 10-100× more blocks than SMs.

---

### Q35. True or False: You should always try to minimize the number of "wasted" threads, even if it means using non-standard block sizes.

**Answer:** **False**

**Explanation:**

Prioritize **warp efficiency** and **standard block sizes** over minimizing wasted threads.

**Why "minimizing waste" is the wrong goal:**

1. **Warp inefficiency is worse than idle threads**
   - Block with 100 threads: 28 warp slots wasted (worse!)
   - Block with 128 threads: 0 warp slots wasted, but might have idle threads (better!)

2. **Idle threads have minimal cost**
   - They exit early via boundary check
   - Don't consume significant resources
   - GPU is designed to handle this efficiently

3. **Non-standard sizes cause problems**
   - Poor warp utilization
   - Less predictable performance
   - May not be well-optimized by compiler

4. **Standard sizes are optimized**
   - 128, 256, 512 are tested and optimized
   - Libraries and tools expect these sizes
   - Better overall performance

**Example:**
```cuda
// Array with 10,000 elements

// BAD: Trying to minimize waste
int threadsPerBlock = 200;  // 10,000 / 50 blocks
// Warp efficiency: 200/224 = 89.3% (bad!)

// GOOD: Use standard size
int threadsPerBlock = 256;  // Need 40 blocks, 240 threads wasted
// Warp efficiency: 100% (good!)
// The 240 "wasted" threads are insignificant compared to warp loss
```

**Priority order:**
1. Warp alignment (multiple of 32)
2. Standard block sizes (128-512)
3. Enough blocks for SM utilization
4. Minimizing wasted threads (least important)

---

## Section 5: Conceptual Understanding

### Q36. Explain why matrix addition doesn't benefit much from 2D vs 1D configuration, but matrix multiplication does.

**Answer:**

**Matrix Addition:**
```cuda
C[i] = A[i] + B[i];  // Each element is independent
```

- **Element-wise operation:** Each output element needs exactly 2 inputs
- **No cooperation:** Threads don't need to share data
- **No spatial locality matters:** Thread can be organized any way
- **Memory access:** Read 2 values, write 1 value (simple)
- **1D or 2D makes no difference** - just different ways to organize the same work

**Matrix Multiplication:**
```cuda
C[i][j] = sum(A[i][k] * B[k][j] for all k)  // Needs entire row and column
```

- **Complex operation:** Each output needs N inputs (entire row × column)
- **Cooperation helps:** Threads in a block can share data via shared memory
- **Spatial locality critical:** Nearby threads access nearby memory
- **2D structure enables tiling:**
  - Load 16×16 tile of A into shared memory
  - Load 16×16 tile of B into shared memory
  - All threads in block reuse this data
  - Reduces global memory access by ~16×

**Optimization example (matmul):**
```cuda
// 2D block structure enables this:
__shared__ float tileA[16][16];
__shared__ float tileB[16][16];

// Threads cooperate to load tiles
tileA[ty][tx] = A[...];  // Load collaboratively
__syncthreads();         // Wait for all

// All threads reuse shared data
for (k = 0; k < 16; k++)
    sum += tileA[ty][k] * tileB[k][tx];
```

This only makes sense with 2D organization!

**Summary:**
- **Addition:** No benefit from 2D (use either)
- **Multiplication:** Major benefit from 2D (required for optimization)

---

### Q37. What would happen if you forgot the boundary check in this kernel?

```cuda
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Missing: if (idx < N)
    data[idx] = data[idx] * 2.0f;
}
```

**Answer:**

**You would get undefined behavior and likely crashes!**

**What goes wrong:**

1. **Out-of-bounds memory access:**
   - If N=1000 and you launch 1024 threads (4 blocks × 256)
   - Threads 1000-1023 will access `data[1000]` to `data[1023]`
   - These indices are **beyond the allocated memory**

2. **Possible outcomes:**
   - **Best case:** Corrupt other data in your program
   - **Common case:** Segmentation fault / CUDA error
   - **Worst case:** Silent data corruption (hardest to debug!)

3. **CUDA error:**
   - May trigger `cudaErrorIllegalAddress`
   - Subsequent CUDA calls will fail
   - Program may crash

**Example:**
```cuda
float *data;
cudaMalloc(&data, 1000 * sizeof(float));  // Allocate 1000 elements

kernel<<<4, 256>>>(data, 1000);  // Launch 1024 threads

// Threads 1000-1023 access data[1000] to data[1023]
// These are OUTSIDE allocated memory!
// Result: Undefined behavior
```

**Correct version:**
```cuda
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  // ← Essential boundary check
        data[idx] = data[idx] * 2.0f;
    }
}
```

**Why we over-launch threads:**
- Simplifies block calculation (ceiling division)
- Minimal overhead for idle threads
- But **must** have boundary check!

---

### Q38. A friend says: "I'll use 1024 threads per block because more threads = faster!" What would you tell them?

**Answer:**

**This is a common misconception!** More threads per block is NOT always better. Here's why:

**Problems with 1024 threads/block:**

1. **Limits blocks per SM:**
   - If kernel uses many registers: Maybe only 1 block fits per SM
   - With 256 threads/block: Could fit 4-8 blocks per SM
   - **Fewer blocks = worse occupancy**

2. **Resource constraints:**
   ```
   SM resources are limited:
   - Max ~65k registers per SM
   - Max ~48-96 KB shared memory per SM
   
   1024 threads/block:
   - Uses more registers/shared memory
   - Fewer blocks can run simultaneously
   - Lower occupancy
   ```

3. **Reduced flexibility:**
   - Small problems might only need 100 blocks
   - With 1024 threads: Only 100 blocks total
   - With 256 threads: Could have 400 blocks
   - More blocks = better SM utilization

4. **No performance gain for simple kernels:**
   - Simple kernels don't benefit from large blocks
   - Memory bandwidth, not block size, is the bottleneck

**When 1024 is appropriate:**
- Kernel needs extensive cooperation within block
- Using lots of shared memory already (so can't fit many blocks anyway)
- Specific algorithm requires it

**Better advice:**
```cuda
// Usually optimal:
dim3 threadsPerBlock(16, 16);  // 256 threads
// or
dim3 threadsPerBlock(16, 16);  // 256 threads
// or
int threadsPerBlock = 512;     // Good middle ground

// Use 1024 only when you have a specific reason
```

**The truth:** **256-512 threads per block is the sweet spot** for most kernels!

---

### Q39. You need to process 1,000,000 elements. Your GPU has 80 SMs. Roughly how many blocks should you launch? Walk through your reasoning.

**Answer:**

**Step 1: Choose threads per block**
```
Use standard size: 256 threads/block
(Could also use 128, 512, but 256 is common)
```

**Step 2: Calculate blocks needed**
```
blocksPerGrid = (1,000,000 + 255) / 256
              = 1,000,255 / 256
              = 3,907 blocks
```

**Step 3: Analyze relative to SMs**
```
GPU has: 80 SMs
Blocks: 3,907
Ratio: 3,907 / 80 ≈ 49 blocks per SM
```

**Step 4: Is this good?**

**YES! This is excellent!**
- ✓ **49× more blocks than SMs** (great load balancing)
- ✓ As blocks finish, new ones immediately start
- ✓ All SMs stay busy throughout execution
- ✓ Good occupancy

**What if blocks per SM varied:**

```
Scenario A: Only 80 blocks total (1 per SM)
- Bad: When blocks finish, SMs sit idle
- No load balancing

Scenario B: 800 blocks (10 per SM)
- OK: Some load balancing
- Could be better

Scenario C: 3,907 blocks (49 per SM)
- Excellent: Great load balancing ✓
- SMs always have work

Scenario D: 80,000 blocks (1000 per SM)
- Excellent: Even better! (if problem size supports it)
```

**Rule of thumb:**
- Minimum: 2-3× blocks per SM
- Good: 10-50× blocks per SM  ← **We're here!**
- Excellent: 100× blocks per SM

**Final answer:** Launch approximately **3,900-4,000 blocks** (varies with threadsPerBlock choice). This provides ~49 blocks per SM, which is excellent for keeping the GPU busy!

---

### Q40. Explain in your own words: What's the relationship between blocks, SMs, and warps?

**Answer:**

**The Hierarchy:**

```
Physical Hardware (GPU)
└── Streaming Multiprocessor (SM) - Physical compute unit
    ├── Executes multiple blocks simultaneously (if resources allow)
    └── Block (Logical unit scheduled on SM)
        ├── Cannot span multiple SMs
        ├── Contains multiple warps
        └── Warp (Group of 32 threads)
            ├── Execution unit of the SM
            ├── All 32 threads execute same instruction (SIMT)
            └── Thread (Individual execution context)
```

**How they relate:**

**1. Blocks → SMs (Scheduling)**
- GPU scheduler assigns blocks to SMs
- One block runs on exactly one SM (never split)
- One SM can run multiple blocks simultaneously
- As blocks finish, new blocks are scheduled

**2. Blocks → Warps (Composition)**
- Each block is divided into warps of 32 threads
- Block with 256 threads = 8 warps
- Block with 100 threads = 4 warps (with 28 wasted slots)

**3. SMs → Warps (Execution)**
- SMs execute warps, not individual threads
- Warp scheduler picks which warp executes each cycle
- Multiple warps per SM allow latency hiding
- While one warp waits for memory, another executes

**Concrete example:**

```
GPU Configuration:
- 80 SMs
- Each SM can hold up to 2048 threads and 16 blocks

Kernel launch: <<<1000, 256>>>
- 1000 blocks total
- 256 threads per block = 8 warps per block

Execution:
1. Scheduler distributes 1000 blocks across 80 SMs
2. Each SM might run 8-12 blocks simultaneously (if resources allow)
3. Each block's 8 warps are scheduled by that SM's warp scheduler
4. At each cycle, SM executes 1-2 warps (depends on architecture)
5. As blocks complete, new blocks from the 1000 are scheduled
```

**Key relationships:**
- **Blocks organize threads** (programmer's view)
- **SMs execute blocks** (hardware scheduling)
- **Warps are the execution unit** (hardware execution)

**Memory hierarchy connection:**
- Registers: Per-thread
- Shared memory: Per-block (all threads in block can access)
- L1 cache: Per-SM
- L2 cache: Shared across all SMs
- Global memory: Shared across entire GPU

---

## Bonus Challenge Questions

### Q41. Debug this code:

```cuda
int N = 1000;
int threadsPerBlock = 256;
int blocksPerGrid = N / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerGrid>>>(data, N);
```

**Answer:**

**Bug:** `blocksPerGrid = N / threadsPerBlock` uses integer division without ceiling.

**What goes wrong:**
```
N / threadsPerBlock = 1000 / 256 = 3.90625
Integer division: = 3 blocks

3 blocks × 256 threads = 768 threads
But we need 1000 threads!
Elements 768-999 are NOT processed! ❌
```

**Fixed code:**
```cuda
int N = 1000;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // = 4 blocks
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);

// Now: 4 blocks × 256 threads = 1024 threads ✓
// Threads 1000-1023 exit via boundary check
```

**In kernel, must have:**
```cuda
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  // Essential!
        data[idx] = /* ... */;
    }
}
```

**The fix:** Always use ceiling division: `(N + threadsPerBlock - 1) / threadsPerBlock`

---

### Q42. You have a 1024×2048 matrix. Design a complete launch configuration (include both 1D and 2D approaches) and explain which you'd prefer.

**Answer:**

**Matrix dimensions:** 1024 rows × 2048 columns = 2,097,152 elements

---

**Approach 1: 1D Configuration**

```cuda
// Treat as flat array
int N = 1024 * 2048;  // 2,097,152 elements
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
// = (2097152 + 255) / 256 = 8193 blocks

kernel_1D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

// Kernel:
__global__ void kernel_1D(float *A, float *B, float *C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = rows * cols;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // For addition
    }
}
```

**Stats:**
- Blocks: 8,193
- Threads per block: 256
- Total threads: 2,097,408
- Wasted threads: 256

---

**Approach 2: 2D Configuration**

```cuda
// Treat as 2D matrix
dim3 threadsPerBlock(16, 16);  // 256 threads
dim3 blocksPerGrid(
    (2048 + 15) / 16,  // Columns: 2063/16 = 129 blocks
    (1024 + 15) / 16   // Rows: 1039/16 = 65 blocks
);
// Total: 129 × 65 = 8,385 blocks

kernel_2D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

// Kernel:
__global__ void kernel_2D(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];  // For addition
    }
}
```

**Stats:**
- Blocks: 8,385 (129 × 65)
- Threads per block: 256
- Total threads: 2,146,560
- Wasted threads: 49,408 (2.3%)

---

**Alternative 2D: Using 32×8 blocks**

```cuda
dim3 threadsPerBlock(32, 8);  // 256 threads
dim3 blocksPerGrid(
    (2048 + 31) / 32,  // 2079/32 = 65 blocks
    (1024 + 7) / 8     // 1031/8 = 129 blocks
);
// Total: 65 × 129 = 8,385 blocks (same as before)
```

---

**Which do I prefer?**

**For matrix addition: 1D approach**
- ✓ Simpler code
- ✓ Simpler launch configuration
- ✓ Fewer wasted threads (256 vs 49,408)
- ✓ No conceptual benefit from 2D structure
- Performance: Essentially identical

**For matrix multiplication: 2D approach (16×16)**
- ✓ Required for tiling optimization
- ✓ Natural structure for shared memory
- ✓ Square blocks work well for matrix tiles
- ✓ Enables thread cooperation
- Performance: Much better with optimization

**For this specific problem (1024×2048):**

**If just doing addition:** Use **1D**
```cuda
kernel_1D<<<8193, 256>>>(d_A, d_B, d_C, 1024, 2048);
```

**If doing multiplication or other complex ops:** Use **2D with 16×16 blocks**
```cuda
kernel_2D<<<dim3(128, 64), dim3(16, 16)>>>(d_A, d_B, d_C, 1024, 2048);
```

**My choice for general matrix code:** 2D with 16×16 blocks - it's the standard, works for all matrix operations, and is slightly more readable for matrix code even if performance is identical for simple operations.

---

### Q43. A kernel processes 100,000 elements. You profile it and find only 30% of your GPU's SMs are active. What might be wrong and how would you investigate?

**Answer:**

**Problem:** Only 30% SM utilization means 70% of GPU is idle - very inefficient!

---

**Possible Causes:**

**1. Too few blocks launched**

```cuda
// BAD: Only 24 blocks
kernel<<<24, 4096>>>(data, 100000);  
// Won't work - 4096 > 1024 max!

// BAD: Only 100 blocks
kernel<<<100, 1000>>>(data, 100000);
// On 80-SM GPU, many SMs idle

// GOOD: Many blocks
kernel<<<391, 256>>>(data, 100000);
// 391 blocks keeps all SMs busy
```

**2. Resource exhaustion per block**

```cuda
__global__ void resourceHungryKernel() {
    // Uses many registers
    double temp[100];  // 800 bytes of registers per thread!
    
    // Uses lots of shared memory
    __shared__ float data[10000];  // 40 KB per block
    
    // Result: Only 1-2 blocks can fit per SM
    // → Low occupancy
}
```

**3. Long-running blocks with uneven workload**

```cuda
__global__ void unevenKernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Some blocks do lots of work
    if (idx < 1000) {
        for (int i = 0; i < 1000000; i++) { /* heavy */ }
    }
    // Other blocks do little work
    else {
        data[idx] = idx;  // Quick!
    }
    // Result: Some SMs finish early and sit idle
}
```

**4. Synchronization issues**

```cuda
// Multiple kernel launches with small work
for (int i = 0; i < 1000; i++) {
    smallKernel<<<10, 256>>>(data);  // Launches only 10 blocks each time
    cudaDeviceSynchronize();  // Waits for completion
}
// Better: Launch once with 10,000 blocks
```

---

**How to Investigate:**

**1. Check launch configuration**

```cuda
// Add diagnostic prints
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
printf("Launching %d blocks with %d threads each\n", blocksPerGrid, threadsPerBlock);
printf("Total threads: %d\n", blocksPerGrid * threadsPerBlock);
printf("Elements to process: %d\n", N);
```

**2. Use NVIDIA profiler**

```bash
# Check occupancy
nvprof --metrics achieved_occupancy ./program

# Check SM utilization
nvprof --metrics sm_efficiency ./program

# See detailed metrics
nvprof --print-gpu-trace ./program
```

**3. Check resource usage at compile time**

```bash
nvcc --ptxas-options=-v kernel.cu

# Output shows:
# - Registers used per thread
# - Shared memory used per block
# - Estimated occupancy
```

**4. Use Nsight Compute for detailed analysis**

```bash
ncu --set full -o profile ./program
# Opens GUI with detailed metrics:
# - Warp execution efficiency
# - Memory throughput
# - SM occupancy over time
```

**5. Check for errors**

```cuda
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
```

---

**Solutions based on cause:**

**If too few blocks:**
```cuda
// Increase blocks, decrease threads per block
kernel<<<391, 256>>>(data, 100000);  // Instead of <<<100, 1000>>>
```

**If resource exhaustion:**
```cuda
// Reduce resource usage
__global__ void optimizedKernel() {
    // Use fewer local variables
    float temp;  // Instead of temp[100]
    
    // Use smaller shared memory
    __shared__ float data[1024];  // Instead of [10000]
}
```

**If uneven workload:**
```cuda
// Balance work distribution
// Use more blocks with smaller chunks
// Or use dynamic parallelism for irregular work
```

**Specific investigation for 100,000 elements:**

```cuda
// What you might have (BAD):
kernel<<<100, 1000>>>(data, 100000);
// Issue: Only 100 blocks on 80-SM GPU
// Some SMs get 2 blocks, some get 1, many get 0

// What you should have (GOOD):
int threadsPerBlock = 256;
int blocksPerGrid = (100000 + 255) / 256;  // = 391 blocks
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, 100000);
// Now: 391 blocks across 80 SMs ≈ 5 blocks per SM
// All SMs stay busy!
```

---

## Summary: Key Formulas to Memorize

**1D Launch:**
```cuda
int threadsPerBlock = 256;  // Choose: 128, 256, or 512
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);
```

**2D Launch:**
```cuda
dim3 threadsPerBlock(16, 16);  // 256 threads
dim3 blocksPerGrid(
    (cols + 15) / 16,  // X dimension
    (rows + 15) / 16   // Y dimension
);
kernel<<<blocksPerGrid, threadsPerBlock>>>(data, rows, cols);
```

**Global Index Calculations:**
```cuda
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * cols + col;
```

**Key Principles:**
1. Always use ceiling division for blocks
2. Block size should be multiple of 32 (warp size)
3. Aim for 256-512 threads per block
4. Launch many more blocks than SMs (10-100×)
5. Always include boundary checks in kernels
6. Don't worry about "wasted" threads if warps are efficient

---

## Grading Rubric

- **Section 1 (Basic Concepts):** /10 points
- **Section 2 (Launch Configuration):** /10 points
- **Section 3 (2D Configuration):** /8 points
- **Section 4 (Warps and Efficiency):** /7 points
- **Section 5 (Conceptual):** /5 points
- **Bonus (Challenge):** /3 points

**Total:** /43 points

**Grade Scale:**
- 40-43: A+ (Excellent)
- 35-39: A (Very Good)
- 30-34: B (Good)
- 25-29: C (Satisfactory)
- 20-24: D (Needs Improvement)
- <20: F (Significant gaps)

---

*End of Questionnaire* multiple of 32 (warp size)
3. Aim for 256-512 threads per block
4. Launch many more blocks than SMs (10-100×)
5. Always include boundary checks in kernels
6. Don't worry about "wasted" threads if warps are efficient

---

## Grading Rubric

- **Section 1 (Basic Concepts):** /10 points
- **Section 2 (Launch Configuration):** /10 points
- **Section 3 (2D Configuration):** /8 points
- **Section 4 (Warps and Efficiency):** /7 points
- **Section 5 (Conceptual):** /5 points
- **Bonus (Challenge):** /3 points

**Total:** /43 points

**Grade Scale:**
- 40-43: A+ (Excellent)
- 35-39: A (Very Good)
- 30-34: B (Good)
- 25-29: C (Satisfactory)
- 20-24: D (Needs Improvement)
- <20: F (Significant gaps)

---

*End of Questionnaire*
