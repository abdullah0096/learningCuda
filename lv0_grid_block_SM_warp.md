# CUDA Execution Model — Summary Notes

This note summarizes the core execution concepts in CUDA: **threads, warps, blocks, grids, and Streaming Multiprocessors (SMs)**, and how they relate to each other.

---

## 1. Thread

- **Smallest logical unit** in CUDA.
- Executes the kernel code.
- Has:
  - Its own registers
  - Its own program counter
- Identified by:
  - `threadIdx.{x,y,z}` (within a block)

🔹 A single thread is **not** an independent hardware execution unit.

---

## 2. Warp

- **Smallest hardware execution unit**.
- A warp consists of **32 threads**.
- All threads in a warp:
  - Execute the **same instruction** at the same time (SIMT model)
  - May diverge due to branches (performance penalty)

Key points:
- Warp size is fixed at **32** on all NVIDIA GPUs.
- Threads are grouped into warps **in order** inside a block.
- Memory coalescing and branch divergence happen at the **warp level**.

🔹 **Warps are what the GPU actually executes.**

---

## 3. Block (Thread Block)

- A **logical grouping of threads**.
- Defined by the programmer via `threadsPerBlock`.
- Identified by:
  - `blockIdx.{x,y,z}`
- Properties:
  - Runs entirely on **one SM**
  - Cannot migrate to another SM
  - Can use **shared memory**
  - Threads can synchronize using `__syncthreads()`

Block characteristics:
- Size ≤ 1024 threads (architecture limit)
- Block size determines:
  - Number of warps
  - Resource usage (registers, shared memory)
  - Occupancy

🔹 **Block is the unit of scheduling.**

---

## 4. Grid

- A **collection of blocks**.
- Defined by the programmer via `blocksPerGrid`.
- Created when a kernel is launched.

Properties:
- Blocks in a grid:
  - Can execute in any order
  - Can run concurrently or sequentially
- No implicit synchronization between blocks.

🔹 **Grid exists only at launch level; it has no direct hardware representation.**

---

## 5. Streaming Multiprocessor (SM)

- The **main hardware processing unit** of a GPU.
- Contains:
  - CUDA cores
  - Warp schedulers
  - Registers
  - Shared memory / L1 cache

Responsibilities:
- Accepts blocks from the scheduler
- Breaks blocks into warps
- Executes warps
- Time-multiplexes warps to hide memory latency

Example (RTX 2080):
- 46 SMs
- Each SM:
  - Executes multiple warps concurrently
  - Hosts multiple resident blocks (resource-limited)

🔹 **SMs execute warps, not blocks or grids.**

---

## 6. How Everything Maps Together

```text
Grid (logical)
 └── Block (logical, scheduling unit)
      └── Warp (hardware execution unit)
           └── Thread (lane inside warp)
