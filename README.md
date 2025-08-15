# GPU-Accelerated Image Processing & Convolution

---

## Project overview

This repository contains a CUDA-based image processing project that implements high-throughput, multi-channel image-transformation kernels. Implementations include:

* an **inverted grayscale** kernel, and
* a **custom Thomas transformation** (user-defined multi-channel transform).

The solution is optimized for real GPUs (NVIDIA T4 used for benchmarking) and focuses on maximizing throughput for megapixel-scale images by using:

* shared memory and constant memory where appropriate,
* coalesced global memory access patterns, and
* techniques to minimize shared-memory bank conflicts.

**Benchmark (representative):** \~**16×** speedup vs. a sequential single-threaded Intel i5 (10th-gen) CPU on megapixel images (measured on an NVIDIA T4).

---

## Repository layout

```
.
├── input/                 # Test input images (put test cases here)
├── output/                # Expected output images (same filenames as input)
├── submit/                # Place the project source to be compiled and run
│   ├── main.cu            # <-- main CUDA file (required)
│   └── compile.sh         # <-- compile script (required)
├── logFile                # Execution and comparison log (created by run.sh)
├── timing_logFile         # Timings recorded for each test (created by run.sh)
├── run.sh                 # Runs all testcases, compares outputs, records timings
└── README.md              # <-- this file
```

> **Important**: Put your `main.cu` and `compile.sh` inside the `submit/` folder. `run.sh` expects `submit/compile.sh` to produce a runnable binary (see *Compile & run*).

---

## Dependencies

* **NVIDIA GPU + drivers** compatible with the CUDA Toolkit you will use (e.g., NVIDIA T4 was used for testing).
* **CUDA Toolkit** (nvcc). CUDA 11.x or newer recommended.
* **bash** (for `run.sh` and `compile.sh`).
* Standard Unix utilities (`diff`, `time`, etc.) for comparison and timing (these are used by the provided `run.sh`).

---

## How to compile

`submit/compile.sh` should compile `submit/main.cu` into an executable (example name `image_proc`). A recommended nvcc command (you can place this in `compile.sh`):

```bash
#!/usr/bin/env bash
set -e
# Example compile command (adjust -arch according to your GPU)
nvcc -O3 -arch=sm_75 submit/main.cu -o submit/image_proc
```

* `sm_75` was used for NVIDIA T4 (change if you target different hardware).
* Ensure `compile.sh` is executable: `chmod +x submit/compile.sh`.

---

## How to run (automated)

`run.sh` automates the full test flow:

1. Calls `submit/compile.sh` to compile the CUDA binary.
2. Iterates over files in `input/`.
3. For each input, runs the compiled binary to produce an output (or writes to a temporary output file).
4. Compares the produced output with the expected file in `output/` (filename should match).
5. Records per-test execution time into `timing_logFile`.
6. Appends result details (pass/fail, diffs, errors) to `logFile`.

Basic usage:

```bash
# make sure scripts are executable
chmod +x submit/compile.sh run.sh

# run the full test-suite
./run.sh
```

After execution:

* Check `logFile` for pass/fail and diagnostics.
* Check `timing_logFile` for per-test timings and aggregated stats.

---

## Test case / file format

* Place **input** test images (one test per file) in the `input/` directory.
* Place the **expected** output images (same filenames) in the `output/` directory.
* `run.sh` expects filenames to match between `input/` and `output/`. If your executable outputs files to a different naming convention, update `run.sh` accordingly.

> If your project uses a non-image binary format, ensure `run.sh`'s comparison step matches the produced file format (text `diff` vs. binary `cmp` or custom comparator).

---

## What to edit / extend

* `submit/main.cu` — where kernels and host-side orchestration live. Update or add kernels, change grid/block configuration, or modify image I/O as needed.
* `submit/compile.sh` — update compilation flags, architecture, or linking requirements.
* `run.sh` — modify how inputs/outputs are located, alter comparator logic, or change logging/timing behavior.

---

## Logging & timings

* `logFile` contains the per-test run details and comparison results (which tests passed or failed and any relevant messages).
* `timing_logFile` contains timing information per test (useful for per-image throughput and aggregate benchmarking).
* Typical timing fields: test filename, execution time (s), GPU device used, any additional stats your binary prints.

---

## Performance notes & tips

* Tune block/grid dimensions for your GPU architecture; try different block sizes and measure with `timing_logFile`.
* Use shared memory to stage neighbor loads for convolution-like operations.
* Use constant memory for small lookup tables or transform constants.
* Align host buffers and use pinned (page-locked) memory for faster host-to-device transfers if your workload is transfer-bound.
* Profile with NVIDIA tools (nsight, nvprof, nvvp) to identify bottlenecks (memory-bound vs compute-bound).

---

## Troubleshooting

* **nvcc not found**: make sure CUDA Toolkit is installed and `nvcc` is on `PATH`.
* **No CUDA device visible**: ensure NVIDIA driver is installed and `nvidia-smi` reports your GPU.
* **Wrong arch/arch warnings**: set `-arch` / `-gencode` flags appropriate for your GPU.
* **Mismatched outputs**: check that `output/` files exactly match the binary's produced outputs (byte-wise or via expected tolerances).

---

## License

This repository is provided for educational/demo purposes. You can add an explicit license here (e.g., MIT) if you want reuse permissions.

---

## Contact / Notes

* Instructor: Prof. Rupesh Nasre (GPU Programming course).
* For modifications: edit `submit/main.cu` and `submit/compile.sh`. `run.sh` is the central harness that compiles, executes, compares, and logs results.

---
