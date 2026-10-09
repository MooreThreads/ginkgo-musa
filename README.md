# ginkgo-musa

`ginkgo-musa` is the Moore Threads MUSA maintenance line of
[Ginkgo](https://github.com/ginkgo-project/ginkgo), a high-performance linear
algebra library for scientific and engineering applications.

## Source and version

- Upstream project: Ginkgo
- Inherited version: `v2.0.0`
- Upstream baseline: `8e9859e5840a6f47e3b1027bbd0dc2cc4a583c33`
- Maintainer: Moore Threads

The MUSA line adds MUSA device and kernel integration while retaining the
upstream reference, CUDA, HIP, DPC++ and CPU backends, tests, and public API.

## Requirements and build

Use CMake 3.16 or newer, a C++17-capable compiler, and the MUSA SDK when
selecting the MUSA backend. A typical build is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Enable the backend and optional components using the CMake options documented
by the project. External BLAS, MPI, and accelerator libraries are supplied by
the target environment.

## Layout and contribution

Library sources are under `core/`, `omp/`, and backend directories; tests,
examples, and CMake modules preserve the upstream organization. Keep backend
changes isolated and include API documentation for new behavior.

Moore Threads maintains this source line. See [LICENSE](LICENSE) for the
applicable declaration.
