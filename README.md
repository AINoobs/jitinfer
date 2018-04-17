# JIT(Just-In-Time) Inference library
JIT Inference is a deep fusion library focusing on deep learning inference workload, supporting on AVX, AVX2 and AVX512 instructions and AVX512 VNNI later.

## Build
You can just run [makell.sh](./makeall.sh) to build and install, or run as below:

``` bash
mkdir -p build && cd build
# debug cmake
# cmake .. -DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_INSTALL_PREFIX=./tmp # -DWITH_VERBOSE=ON -DWITH_COLD_CACHE=ON

# release cmake
cmake .. -DCMAKE_INSTALL_PREFIX=./yourfolder # -DWITH_VERBOSE=ON -DWITH_DUMP_CODE=ON # -DWITH_COLD_CACHE=OFF

# minmal release
# This will only generate jitinfer library without any benchmark utilities and gtests
# cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel

make clean
make -j `nproc`
make test
make install
```

## Use

### How to Benchmark
Add `-DWITH_BENCHMARK=ON` in cmake option, then rebuild and you can run`./build/benchmark/bench_concat`.

### How to profiling
Add `-DWITH_VERBOSE=ON` in cmake options, then export env:
``` bash
export JITINFER_VERBOSE=1
```
This will print the submit time of each operator at every iteration.

### How to dump code
Add `-DWITH_DUMP_CODE=ON` in cmake options, then export env:
``` bash
export JITINFER_DUMP_CODEE=1
```
Then, when run some apps, you can get some file like `jit_dump_jit_concat_kernel.0.bin`, then use `xed` to view the ASM. For exapmle:
```
$xed -ir jit_dump_jit_concat_kernel.0.bin
XDIS 0: PUSH      BASE       53                       push ebx
XDIS 1: PUSH      BASE       55                       push ebp
XDIS 2: BINARY    BASE       41                       inc ecx
XDIS 3: PUSH      BASE       54                       push esp
XDIS 4: BINARY    BASE       41                       inc ecx
XDIS 5: PUSH      BASE       55                       push ebp
XDIS 6: BINARY    BASE       41                       inc ecx
XDIS 7: PUSH      BASE       56                       push esi
XDIS 8: BINARY    BASE       41                       inc ecx
XDIS 9: PUSH      BASE       57                       push edi

```

### Cmake Options
- `-DWITH_BENCHMARK=ON`
- `-DWITH_VERBOSE=ON`
- `WITH_DUMP_CODE`

This three options are supported. They are enabled in Debug mode and disabled in Release mode by default.

### MinSizeRel
This will only generate jitinfer library without any benchmark utilities and gtests. All the cmake options shown above are disabled. \
`cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel`

## Docker env
Docker images are provided for compiling and debuging.
 - `docker pull tensortang/ubuntu` for gcc8.0, gdb and some necessary env.
 - `docker pull tensortang/ubuntu:16.04` for gcc5.4

## Supported Operators

### 1. Concat and relu fusion.
Support on AVX, AVX2 and AVX512

### 2. Conv fusion
conv relu and conv1x1relu fusion (will support VNNI).
 - fuse: conv + relu + conv(with 1x1 weight) + relu
 - supported multi channel scales
 - supported various data type

  | Memory | Supported Data Type |
  |---|--- |
  | src | u8 |
  | weight | s8 |
  | bias | u8/s8/s32/f32 |
  | dst | u8/s8/s32/f32 |

## Third party
Xbyak and Intel(R) MKLML are the only two necessary dependencies for Jitinfer library.

Below are the third parties:
- [Xbyak](https://github.com/herumi/xbyak), for JIT kernels.
- [Intel(R) MKLML](https://github.com/intel/mkl-dnn/releases/download/v0.13/mklml_lnx_2018.0.2.20180127.tgz), for Intel OpenMP library.
- [Intel(R) MKL-DNN](https://github.com/intel/mkl-dnn), for benchmark comparion and gtest refernce.
- [gtest](https://github.com/google/googletest), for test cases framework.
- [gflags](https://github.com/gflags/gflags), for benchmark arguments.


## [How to contribute](./doc/HowToContribute.md)
