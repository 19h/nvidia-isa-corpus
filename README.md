# NVCC build results

The build produced 2,883 CUDA binaries and 2,826 PTX modules from 65 of the 97 inputs. The CUDA binaries span 47 SM targets, from `sm_10` through `sm_121f`, including `sm_107`, `sm_107a`, and `sm_107f`. Thirty-one programs built for every target. PTX ISA versions emitted were 1.4, 5.0, 6.5, 7.8, 8.8, 9.3, and 9.5.

- [Coverage by SM](coverage.tsv) and [coverage by program](port-check.tsv)
- [Build matrix with artifact and diagnostic paths](matrix.tsv)
- [Targets and default NVCC toolchains](targets.tsv)
- [RTX 4090 smoke tests](smoke-test.tsv)
- [CUDA adaptations](cuda-src/), [PTX](ptx/), and [CUDA binaries](cubin/)

The original files are in `programs/`. The CUDA adaptations qualify portable functions and globals for device execution and expose a `program_entry` kernel. Launch with one block and one thread and pass a device pointer to an `int` that receives the return code. `02_fibonacci` takes two additional parameters, the Fibonacci index and the nominal GPU clock frequency in kHz, and uses device clock cycles for elapsed time estimates.

The SIMD adaptations use the sources' scalar or portable paths. The first-generation targets below `sm_13` use the legacy compiler's double-to-float mapping. `cf_musttail` compiles as a workload, with tail-call behavior determined by NVCC. Target-specific failures include unsupported recursion, device I/O, variadics, and dynamic stack allocation. Programs without a device build hit CPU assembly, OS interfaces, unsupported language features, or host library dependencies. The matrix points to the compiler diagnostics for each gap.

Modern builds used GCC 13 as the host compiler. Legacy toolchains came from the NVIDIA package archives and were extracted under `/tmp/nvidia-legacy` without a system-wide installation. CUDA 11.8 supplied the `sm_35` and `sm_37` builds of `int_bitfields` and `int_int128_ops`. The build matrix records the actual NVCC version per successful build.

Each CUDA binary was checked for its ELF machine type, SM number, and kernel symbol. Each corresponding PTX module was checked for its target and entry point. Four `sm_89` workloads ran on the RTX 4090 and their device results matched CPU references.

## Vulkan SC

The [Vulkan SC build](vulkan-sc/README.md) contains compute shader ports, validated SPIR-V, and NVIDIA device-specific offline pipeline caches.
