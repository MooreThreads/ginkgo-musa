# Ginkgo MUSA integration

MUSA support is added as a device/backend path alongside Ginkgo's inherited
CPU, CUDA, HIP, and other implementations. CMake selects the MUSA compiler,
runtime, and target architecture while preserving the public linear-algebra
interfaces. The MUSA SDK is supplied by the deployment environment.
