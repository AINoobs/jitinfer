# How to contribute

## Pre commit
Please follow below instructions to install `pre-commit`, this will check the code style.
1. `pip install pre-commit`, about pre-commit check [here](http://pre-commit.com/#plugins).
2. `pre-commit install`. (use `pre-commit uninstall` to uninstall)
3. Then git commit ...

Try your best to follow this format `conv: benchmark: xxx commit` when push your commit.

## Add new operator
Adding any new feature should contain at least 3 components: src, test and benchmark

### 1. src
Source files must be placed in `./src`. \
This should contain 2 parts: `op` and `jit`.

For example:
 - op: `op_concat.cc` and `op_concat.h`, defining operator and omp part.
 - jit: `jit_concat_kernel.cc` and `jit_concat_kernel.h`, focusing on jit kernel code.

@Note: The API must only be added in `./include/jitinfer.h`.

### 2. test
Test file must be placed in `./test`. \
Adding any new file with `*.cc` will automatically generate corresponding executable file. \
For axample, adding `test_conv.cc` will generate `./build/test/test_conv`.

### 3. benchmark
This should focus on comparing your performance with Intel  MKL-DNN. \
Please add your file in `./benchmark`. \
For axample, adding `bench_conv.cc` will generate `./build/benchmark/bench_conv`.

## Update readme
At last, please add some description about yout operator in [README.md](../README.md).
