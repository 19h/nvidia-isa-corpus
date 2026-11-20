# Vulkan SC compute corpus

**87 of 97 programs** built as validated compute shaders. All 87 produced offline Vulkan SC caches for each of **34 NVIDIA chip targets**, for **2,958 cache files**.

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

The SPIR-V outputs used a clspv Compiler Explorer build dated 2026-11-18 and SPIRV-Tools 2026.1. PCC 595.99.02 built the caches for 29 targets. The Jetson Linux 36.5.2 PCC (540.5.0) generated the `ga10b` and `ga10f` caches under AArch64 user-mode emulation. PCC 620.73.04 generated `gb10c`, `gr100`, and `gr102`. [NVIDIA Jetson Linux package repository](https://repo.download.nvidia.com/jetson/) · [NVIDIA 620.73.04 driver archive](https://download.nvidia.com/XFree86/Linux-x86_64/620.73.04/)

`--chip all` queries the default PCC and includes any additional targets supplied with `--pcc-for-chip`. Set the following paths to executables or wrapper scripts for the two alternate compilers to reproduce the included target matrix.

```sh
python3 vulkan-sc/build.py --chip all --reuse \
  --pcc-for-chip "ga10b=$JETSON_PCC" --pcc-for-chip "ga10f=$JETSON_PCC" \
  --pcc-for-chip "gb10c=$NEWER_PCC" --pcc-for-chip "gr100=$NEWER_PCC" \
  --pcc-for-chip "gr102=$NEWER_PCC"
```

The script records individual failures and returns nonzero if any shader or cache build fails. It validates each accepted cache's Vulkan SC header and pipeline identifier.

The sources under [opencl/](opencl/) were adapted from the original C or C++ files by [port.py](port.py). The [workloads/](workloads/) and [glsl/](glsl/) directories contain ports with individually adapted algorithms or interfaces. The C++ automaton's constants are emitted from the original compile-time tables by [automaton-tables.cpp](automaton-tables.cpp). The build generates shader headers from the native regex and JSON fixtures, assembles the four VM demos with the original C assembler, and instantiates the C++ template showcase to obtain its compile-time results. The host compilers for these steps can be set with `--cc` and `--cxx`. [Fixture generators](fixtures/)

## Run

Set `VKSC_SDK` to the SDK directory that contains `include/` and `lib64/`.

```sh
cc -std=c11 -O2 -I"$VKSC_SDK/include" vulkan-sc/run.c \
  -L"$VKSC_SDK/lib64" -lvulkansc -o /tmp/corpus-vksc-run
LD_LIBRARY_PATH="$VKSC_SDK/lib64" python3 vulkan-sc/verify.py /tmp/corpus-vksc-run --chip ad102
```

The runner dispatches one workgroup. Most shaders use one invocation. The thread-pool shader uses 64 invocations and a shared atomic queue. The neural layer, physics simulation, and image filter use 128 invocations. Descriptor set 0, binding 0 contains a storage buffer with a 64-word result area followed by zero-initialized scratch storage. Word zero is the workload status. Generated checksum outputs put one little-endian byte in each word and start at words 1, 17, and 33 for successive sinks. The hello-world shader returns its text in words 1 through 14. The Fibonacci shader accepts an index in word 63 and reports both implementations for indices through 30. From 31 through 92 it reports the memoized result.

The runner accepts up to six optional unsigned inputs, starting at word 63 and descending. The RTTI workload uses them for the native C++ implementation's six `type_info::hash_code()` low bytes. The polymorphic allocator workload accepts four optional object sizes for its allocation accounting and uses the x86-64 reference sizes by default. The bytecode VM uses word 63 to select arithmetic (0), factorial (1), Fibonacci (2), or nested calls (3). Words 62 through 58 serve as input for its `READ` opcode. The result includes the instruction count, final stack pointer, and a hash of the text emitted by the VM. The image filter uses word 63 to select the scalar (0), SSE2 (1), or NEON (2) algorithm.

The two standalone challenge validators return their packed native verdict in word 1. Pass up to 24 candidate bytes packed into the runner's six input words, least significant byte first. The RAROG runner fixture uses the name `GPU Corpus`. A host that sets word 64 to a nonzero value can pass a null-terminated candidate starting at word 65 for the crackme or a name at word 65 and a sigil at word 81 for RAROG. The buffer allows up to 159 candidate bytes for the crackme and 63 bytes each for the RAROG fields. The original programs provide the terminal UI.

## Scope

The ports preserve the workload computations where possible. They use scalar SIMD fallbacks, index-based pointer chasing, bounded stacks and arenas, and tagged dispatch for function pointers and C++ objects. Host printing is represented by result-buffer values for the ported checksums and text. The denormal workload decodes its IEEE-754 samples as bits and scales them to normal values before GPU arithmetic. The atomic workloads use Vulkan atomics and buffer barriers with packed storage for subword fields. Native CPU ABI behavior, OS interactions, and instruction encodings depend on the original host target.

The JSON shader uses bounded node and string storage and preserves the source spelling of parsed numbers during serialization. The runtime check compares parsed values and the pretty-printing overhead with the native program, and checks that both shader serializations parse back to the same values. The BST uses ordered IDs for the string fixtures. The thread-pool benchmark models callable results and task completion on device. Host timing and C++ exception propagation remain properties of the native program.

The shader matrix records the 10 remaining gaps. Nine exercise POSIX kernel interfaces or architecture-specific CPU instructions. The other is a third-party activation-key generator. The cache matrix records rejected chip–shader pairs. Runtime checks use the local RTX 4090 and Vulkan SC SDK 1.0.15.
