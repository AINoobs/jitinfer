# JIT(Just-In-Time) Inference library
JIT Inference is a deep fusion library focusing on deep learning inference workload, supporting on AVX, AVX2 and AVX512 instructions and AVX512 VNNI later.

## Compile
1. mkdir -p build && cd build
2. cmake ..
3. make -j `nproc`
4. make test
5. make install

It will download, compile and install all dependencies, including:
- [Xbyak](https://github.com/herumi/xbyak), used for JIT kernels.
- [Intel(R) MKL-DNN](https://github.com/intel/mkl-dnn), used for benchmark comparion and refernce in gtest.
- [Intel(R) MKLML](https://github.com/intel/mkl-dnn/releases/download/v0.13/mklml_lnx_2018.0.2.20180127.tgz), used for Intel OpenMP library.
- [gtest](https://github.com/google/googletest)
- [gflags](https://github.com/gflags/gflags)

### Benchmark
`cmake -DWITH_BENCHMARK=ON`, default is enabled. \
Then can run `sh ./build/benchmark/bench_concat`

### How to profiling
`cmake -DWITH_VERBOSE=ON` and `export JITINFER_VERBOSE=1` \
This `WITH_VERBOSE`, cmake option, default is enabled in DEBUG and disabled in Release.

### How to dump code
`cmake -DWITH_DUMP_CODE=ON` and `export JITINFER_DUMP_CODEE=1` \
This `WITH_DUMP_CODE`, cmake option, default is enabled in DEBUG and disabled in Release.

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

### MinSizeRel
This will only generate jitinfer library without any benchmark utilities and gtests. \
`cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel`

## Operators

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

## Docker
Docker images is provied for compiling and debuging.
 - `docker pull tensortang/ubuntu` for gcc8.0, gdb and some necessary env.
 - `docker pull tensortang/ubuntu:16.04` for gcc5.4

## [How to contribute](./doc/HowToContribute.md)
