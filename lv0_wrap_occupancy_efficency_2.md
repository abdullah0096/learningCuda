# CUDA Occupancy and Efficiency — Warp-Centric View

This note explains how to think about **occupancy and efficiency in CUDA**, and why **warps** are the correct level of abstraction.

---

## Core Idea (Very Important)

**Occupancy is a measure of how many warps are active per SM.**  
**Efficiency problems also occur at the warp level.**

Even though CUDA exposes threads, blocks, and grids, the hardware:
- Schedules warps
- Executes warps
- Hides latency using warps

---

## What Occupancy Actually Means

### Definition

Occupancy = Active warps per SM / Maximum warps per SM

Example (RTX 2080 – Turing):
- Maximum warps per SM = 32
- Active warps per SM = 16

Occupancy = 16 / 32 = 50%

---

## Why Warps Matter More Than Threads

- Threads are logical
- Warps are hardware execution units

The GPU:
- Never schedules individual threads
- Never issues instructions to blocks
- Only issues instructions to warps

Therefore, latency hiding, instruction throughput, and memory behavior all depend on warp availability.

---

## How Block Size Translates to Warps

| Threads per Block | Warps per Block |
|------------------|-----------------|
| 64 | 2 |
| 128 | 4 |
| 256 | 8 |
| 512 | 16 |
| 1024 | 32 |

Occupancy depends on warps per block and how many blocks fit on one SM.

---

## What Limits the Number of Active Warps per SM

### 1. Registers

Registers per SM / Registers per thread

High register usage reduces the number of active warps.

---

### 2. Shared Memory

Shared memory per SM / Shared memory per block

Large shared memory usage reduces the number of resident blocks and warps.

---

### 3. Block Size

- Too small: not enough warps
- Too large: fewer blocks fit

---

### 4. Architectural Limits

- Maximum warps per SM
- Maximum blocks per SM
- Maximum threads per SM

---

## Efficiency Problems Are Warp-Level Problems

### Branch Divergence
Occurs within a warp. Different execution paths cause serialization.

### Memory Coalescing
Evaluated per warp. Poor access patterns cause multiple memory transactions.

### Latency Hiding
SM switches between ready warps. Too few warps lead to stalls.

---

## Important Nuance

100% occupancy does NOT guarantee maximum performance.

Reasons:
- Compute units may already be saturated
- Memory bandwidth may be the bottleneck
- Extra warps may not improve latency hiding

In practice:
- 50–75% occupancy is often sufficient
- Some kernels perform best at lower occupancy

---

## How to Think When Designing a Kernel

Ask:
1. How many warps per block?
2. How many blocks per SM can fit?
3. How many active warps per SM does this give?
4. Is this enough to hide latency?
5. Am I register-bound or memory-bound?

---

## Practical Mental Checklist

- Think in warps
- Tune block size
- Monitor register usage
- Monitor shared memory usage
- Use occupancy as a diagnostic, not a goal

---

## One-Line Rules

Occupancy is a warp metric; efficiency is a warp problem.

You program blocks, but the GPU executes warps on SMs.
