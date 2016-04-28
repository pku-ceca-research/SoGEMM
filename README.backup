
# GEMM SDSoC

This is part of the whole SDNN (SDSoC Neural Network) project.

For detailed information about how **blocked matrix multiplication** works, you could find this [website](http://csapp.cs.cmu.edu/2e/waside/waside-blocking.pdf) useful.

## Files

- `gemm_grid.*`: Blocked matrix multiplication
- `gemm.*`: matrix multiplication
- `test_gemm_grid.c`: Test suite for gemm_grid. 

## TODO

### About GEMM standalone

- [x] `gemm_mmult` baseline
- [x] `gemm_madd` baseline
- [x] `trans_to_blocked` and `trans_from_blocked` CPU functions
- [x] On-the-fly version implementation and test
- [x] `gemm_grid_nn` implementation and test
- [ ] `gemm_grid_nt` implementation and test
- [ ] `gemm_grid_tn` and `gemm_grid_tt` implementation and test
- [ ] Darknet integration and test - *Baseline*
- [ ] Make GEMM fully utilise FPGA
- [ ] Darknet integration and test - *MaxPerf*

### About CNN and other layers

TBD

## Installation

Run:

```
make test
```
You can get a x86 CPU version of `test_gemm_grid`

```
make test SDS=1
```

ARM CPU version: `test_gemm_grid.elf` and `sd_card`

```
make test SDS=1 HW=1
```

ARM CPU-FPGA version.

## Run

```
./test_gemm_grid[.elf] -t [test_names] -m [Matrix M width] -n [Matrix N width] -k [Matrix K width] -i [iteration]
```

`test_names` are a comma-seperated string. Available test names:

1. `TRANS`: Test whether the transformation functions works - trans to and trans back.
2. `NORMAL_MMULT`: `gemm_grid_nn` in one iteration
3. `NORMAL_MMULT_PROF`: `gemm_grid_nn` in many iteration (specified in the command line)

