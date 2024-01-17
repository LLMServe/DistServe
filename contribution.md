# Contribution Guide
If you want to contribute to the project, please fork the repo and file a Pull Request for your modification.

## Modify Python code
If you installed `fastserve` in the develop mode, your modification will take effect immediately.

## Modify C++ code
If you have built `SwiftTransformer` previously, in subsequent build (i.e. after modifying some code), you only have to run `cmake --build build -j$(nproc)` under the `SwiftTransformer/` folder.

## Test
We provide various unit tests in `fastserve/tests/` and `SwiftTransformer/src/unittest` to test the correctness. For Python code, please run `python -m pytest` in the `fastserve` directory. For C++ code, please compile the project, and then execute `bin/unittest_XXX` in the `SwiftTransformer/build` directory.

## Development

### C++ Code Structure

Currently, the source code is organized as follows:

```text
src
├── CMakeLists.txt
├── csrc
│   ├── CMakeLists.txt
│   ├── kernel
│   ├── layer
│   ├── model
│   └── util
└── unittest
    ├── CMakeLists.txt
    ├── kernel
    ├── layer
    ├── model
    ├── unittest_torch_utils.h
    ├── unittest_utils.h
    └── util
```

The `csrc` folder contains the core implementation of the model, including the kernel, layer and model.

The `unittest` folder contains unit tests for the components in `csrc`. The `kernel`, `layer`, `model`, and `util` folders under the `unittest` folder contain the implementation of the corresponding components. For example, `src/unittest/layer/attention.cc` contains the unit test for the `Attention` layer, which is implemented in `src/csrc/layer/attention.cc`.

Note for vscode users: If you encounter `#include errors detected. Please update your includePath.`, you may need to update include path in `.vscode/c_cpp_properties.json`.

### Design Philosophy

- **Well-documented.** We strongly believe that a well-documented codebase boosts the efficiency of research. Therefore we try our best to document every function and class. **Typically we explain the purpose and meanings of arguments of a function before its implementation in the `.cc` file.**
- **POP-styled design.** Different from FasterTransformer which adopts an Object-oriented programming (OOP) design, we adopt a more Procedure-Oriented Programming (POP) style. We believe that POP is more suitable for research projects, since it is easier to extend and modify the code. Think why we need OOP, and you will find the answer is "to hide the details". However in research projects, we need to know, and alter the details. Therefore all kernels and layers are implemented in POP style.
- **Extensive unit tests.** Every kernel and layer is paired with a unit test. We believe that unit tests are essential for research projects, since they can help us to verify the correctness of our implementation. We use [googletest](https://github.com/google/googletest) as our unit test framework. With the help of `TYPED_TEST` from googletest, we can test our kernels and layers with different data types (e.g. `float` and `half`) without writing redundant code.
- **LibTorch for reference in unit tests.** For the "reference" part in unittests, we use LibTorch to implement the same kernel or layer. This is because LibTorch is well-tested, and we can use it as a reference to verify the correctness of our implementation.
- **Raw pointers instead of `at::Tensor`.** We prefer the raw pointer in C over `at::Tensor` (The tensor class provided by LibTorch, the C++ frontend of PyTorch), since we need fine-grained control over the memory layout.

### Prerequisite Knowledge

Time for you to get your hands on! Here are some tutorials to help you get started:

- CMake: You need to know what `target` is and what `target_link_libraries` does. Here is a tutorial: [An Introduction to Modern CMake
](https://cliutils.gitlab.io/modern-cmake/)
- GoogleTest: To write unit tests, a brief look at googletest's manual is necessary: [The googletest primer](https://google.github.io/googletest/primer.html)
- CUDA: Interested in CUDA? Here are some tutorials:
  - For beginners: [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/), [CUDA Tutorial](https://www.tutorialspoint.com/cuda/index.htm).
  - Performance insight and optimization: [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html), [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
  - Interesting stuff: [A HISTORY OF NVIDIA STREAM MULTIPROCESSOR](https://fabiensanglard.net/cuda/)
