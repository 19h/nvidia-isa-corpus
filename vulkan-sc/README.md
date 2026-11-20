# Vulkan SC compute corpus

**78 of 97 programs** built as validated compute shaders. All 78 produced offline Vulkan SC caches on 29 NVIDIA chip targets, for **2,262 cache files**. PCC 595.99.02 rejected builds for the two other advertised targets, `ga10b` and `ga10f`, with an NVVM compilation error.

The files here port workloads from [programs/](../programs/) to Vulkan compute. NVIDIA's pipeline cache compiler consumes the validated SPIR-V and the pipeline description to produce an offline Vulkan SC cache. The runner loads the cache through the Vulkan SC loader using its pipeline UUID.

- [Shader coverage for all original programs](shader-status.tsv)
- [Offline cache coverage by NVIDIA chip](cache-status.tsv)
- [RTX 4090 Vulkan SC runtime checks](smoke-test.tsv)
- [SPIR-V](spirv/), [pipeline descriptions](pipelines/), and [offline caches](cache/)

## Build

The build uses [clspv](https://github.com/google/clspv) for OpenCL C shader sources, `glslc` for GLSL, and SPIRV-Tools for validation against the Vulkan 1.2 environment. Cache generation uses `nvidia-pcc` from the NVIDIA driver. The [NVIDIA Vulkan SC SDK](https://developer.nvidia.com/vulkan-sc) provides the Vulkan SC headers and loader for the runner.

```sh
python3 vulkan-sc/build.py --chip ad102
python3 vulkan-sc/build.py --chip all --reuse
```

Each compiler can be selected via a command-line option. The included outputs used a clspv Compiler Explorer build dated 2026-11-18, SPIRV-Tools 2026.1, and NVIDIA PCC 595.99.02. `--chip all` queries PCC for its chip list. The script records individual failures and returns nonzero if any shader or cache build fails. Each accepted cache is checked for the Vulkan SC header and its pipeline identifier.

The sources under [opencl/](opencl/) were adapted from the original C or C++ files by [port.py](port.py). The [workloads/](workloads/) and [glsl/](glsl/) directories contain ports with individually adapted algorithms or interfaces. The C++ automaton's constants are emitted from the original compile-time tables by [automaton-tables.cpp](automaton-tables.cpp).

## Run

Set `VKSC_SDK` to the SDK directory that contains `include/` and `lib64/`.

```sh
cc -std=c11 -O2 -I"$VKSC_SDK/include" vulkan-sc/run.c \
  -L"$VKSC_SDK/lib64" -lvulkansc -o /tmp/corpus-vksc-run
LD_LIBRARY_PATH="$VKSC_SDK/lib64" python3 vulkan-sc/verify.py /tmp/corpus-vksc-run --chip ad102
```

The runner dispatches one workgroup with one invocation. Descriptor set 0, binding 0 contains a storage buffer with a 64-word result area followed by zero-initialized scratch storage. Word zero is the workload status. Generated checksum outputs put one little-endian byte in each word and start at words 1, 17, and 33 for successive sinks. The hello-world shader returns its text in words 1 through 14. The Fibonacci shader accepts an index in word 63 and reports both implementations for indices through 30. From 31 through 92 it reports the memoized result.

The runner accepts up to six optional unsigned inputs, starting at word 63 and descending. The RTTI workload uses them for the native C++ implementation's six `type_info::hash_code()` low bytes. The polymorphic allocator workload accepts four optional object sizes for its allocation accounting and uses the x86-64 reference sizes by default.

## Scope

The ports preserve the workload computations where possible. They use scalar SIMD fallbacks, index-based pointer chasing, bounded stacks and arenas, and tagged dispatch for function pointers and C++ objects. Host printing is represented by result-buffer values for the ported checksums and text. The denormal workload decodes its IEEE-754 samples as bits and scales them to normal values before GPU arithmetic. The atomic workloads use Vulkan atomics and buffer barriers with packed storage for subword fields. Native CPU ABI behavior, OS interactions, and instruction encodings depend on the original host target.

The shader matrix records the 19 remaining gaps. They include POSIX syscalls, CPU instructions, and programs that still require dedicated ports of their host libraries, allocation, or runtime facilities. The cache matrix records rejected chip–shader pairs. Runtime checks use the local RTX 4090 and Vulkan SC SDK 1.0.15.
