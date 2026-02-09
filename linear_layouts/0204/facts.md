Tailored Golden Facts you should explicitly bake into your Claim/Evidence ledgers (microarch)

These are the kinds of “reviewer‑doesn’t‑argue” primary facts that should appear as VERIFIED claims early:

1) **Local memory & spilling meaning (NVIDIA official):** local memory is thread‑private and is used when variables don’t fit in registers or when “register spilling” occurs. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
2) **NCU can report register allocation per thread (`launch__registers_per_thread`)** and provides spill‑related metrics such as `derived__local_spilling_requests` and SASS instruction counts like `sass__inst_executed_register_spilling`. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=openai))  
3) **Interpretation caveat:** `launch__registers_per_thread` can exceed max live registers due to allocation “holes” and ABI/hardware constraints; don’t over‑interpret it without corroborating evidence. ([docs.nvidia.com](https://docs.nvidia.com/nsight-compute/2022.4/NsightCompute/index.html?utm_source=openai))  
4) **TileIR backend reality (as of Jan 30, 2026):** enabling TileIR backend is done via `ENABLE_TILE=1`; prerequisites include CUDA 13.1+ and Blackwell GPUs; the backend is in active development with limitations. ([developer.nvidia.com](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/?utm_source=openai))  
5) **TileIR backend tuning knob (SOTA):** the README describes an **occupancy hint (1–32)** as “critical,” and also explicitly mentions register spilling problems for some norms due to missing `num_warps` exposure. ([github.com](https://github.com/triton-lang/Triton-to-tile-IR?tab=readme-ov-file))  
6) **Workload harness anchor:** TritonBench is a real suite of PyTorch operators to evaluate Triton performance, with runnable commands (`python run.py --op ...`). ([github.com](https://github.com/meta-pytorch/tritonbench?utm_source=openai))  

These VERIFIED claims are what later let Stage 3 read like a serious paper proposal instead of wishful thinking.
