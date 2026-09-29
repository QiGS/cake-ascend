"""CAKE-Ascend: compiler–agent co-design for Ascend kernel evolution.

A from-scratch implementation of the CAKE paper's architecture
(arXiv:2608.12629, "Compiler–Agent Co-Design for Frontier Kernel Evolution"),
re-targeted from NVIDIA CUDA to the Huawei Ascend execution model:

- typed, hardware-explicit schedule IR with declarative resources
  (UB pools / L0C accumulators / roles / pipelines / events)
- an evolvable compiler harness: localized pre-compile gates, numerical
  validation, a calibrated cost model, optimization hints
- a four-stage kernel-agent loop (generate -> filter -> evaluate -> route)
- compiler evolution: recurring failures distilled into verifier rules and
  cost-model calibration, gated by the kernel corpus
- a separate generalization stage (dispatcher portfolios over shape domains)
"""

__version__ = "0.1.0"
