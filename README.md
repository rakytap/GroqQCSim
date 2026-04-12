# GroqQCSim

A quantum circuit simulator targeting Groq's Tensor Streaming Processor (TSP) hardware. It compiles quantum gate programs using the Groq SDK and executes state vector simulations directly on Groq GroqChips, enabling high-throughput simulation of circuits with up to ~21 qubits.

This repository was presented at:

- **GPU Day 2023** — Budapest, Hungary, May 15–16, 2023. Talk: *Simulation of quantum computers on tensor streaming processors*, Péter Rakyta. [https://gpuday.com/gpu-day-2023](https://gpuday.com/gpu-day-2023)
- **ICCS 2023** (International Conference on Computational Science), Quantum Computing Workshop (QCW) track — Prague, Czech Republic, July 3, 2023, 5:40 PM. *Simulation of quantum computers on tensor streaming processors*, Gregory Morse, Peter Rakyta, Tamás Kozsik. [Program link](https://easychair.org/smart-program/ICCS2023/2023-07-03.html#talk:223488)

---

## Overview

GroqQCSim implements a state vector simulator where each quantum gate is compiled into an IOP (I/O Program) targeting the Groq TSP. The simulator supports single-qubit U3 gates and two-qubit controlled gates (CNOT, CRY) across up to 21 qubits. Programs are compiled once with the Groq SDK and can be deployed repeatedly against different input state vectors and gate sequences.

The project integrates with the [SQUANDER](https://github.com/rakytap/sequential-quantum-gate-decomposer) quantum gate decomposition library to support variational algorithms such as VQE (Variational Quantum Eigensolver).

### Gate Categories

Qubits are partitioned into three ranges that correspond to different memory layouts on the TSP chip:

| Label | Qubit range | Notes |
|-------|-------------|-------|
| `S`   | 0–7         | Handled entirely within VXM (MXM) planes; no permutor needed |
| `L`   | 8–15        | Requires memory address remapping |
| `H`   | 16–21       | Requires additional control-qubit memory maps |

Each gate program is named by a `(target)(control)` pair drawn from the labels above (e.g. `gate_SL` = target qubit in S range, control qubit in L range). Uncontrolled single-qubit gates omit the control letter (e.g. `gate_S`, `gate_L`, `gate_H`).

---

## Repository Structure

| File / Directory | Purpose |
|------------------|---------|
| `groqQCsim.py` | Top-level compiler: creates the `ProgramPackage` and compiles all gate programs into IOP files |
| `QCsim_IOP.py` | `Gate_IOP` class — inherits all gate mixins and owns program compilation entry points |
| `QCsim_IO.py` | I/O helper programs (upload state, download state, upload gates) |
| `memory_layouts.py` | `MemoryLayouts` — defines all on-chip memory addresses and gather/scatter maps |
| `gate_<XY>.py` | One file per gate type; each implements the TSP dataflow for that (target, control) qubit-range combination |
| `gate_<XY>_base.py` | Shared base logic for each target-qubit range |
| `gate_counters.py` | Gate counter / permute-map selector programs |
| `generate_static_data.py` | Generates static constants (distributor maps, permutation matrices) embedded in the IOP |
| `state_transform.py` | End-to-end test: generates a random circuit, compiles to IOP, runs on device via SQUANDER integration |
| `groqQCsim_8_21.py` | Variant targeting a single gate type across qubits 8–21 |
| `qcsim_runtime_main.c` | C runtime shim that wraps the Groq driver; compiled to `build/libsvDFE.so` |
| `sincos.py` | Hardware sin/cos implementation on the TSP (used for gate angle evaluation) |
| `Makefile` | Builds the C runtime library and optional test binary |
| `heisenberg-16-20.qasm` | Example OpenQASM 2.0 circuit (16-qubit Heisenberg model) |

---

## Dependencies

- [Groq SDK](https://groq.com/) (tested with runtime 0.9.3; set `GROQ_DIR` in the Makefile if installed elsewhere)
- Python 3.8+
- NumPy
- [SQUANDER](https://github.com/rakytap/sequential-quantum-gate-decomposer) (required for `state_transform.py` and VQE examples; build the `gm_dfe_sv` branch)
- Intel TBB (required by SQUANDER)

---

## Build and Usage

### 1. Build the C runtime library

```bash
# Optionally override the Groq SDK path (default: /opt/groq/runtime.0.9.3)
export GROQ_DIR=/path/to/groq/runtime

make all        # compiles build/libsvDFE.so
make test       # also compiles build/testsv for standalone testing
```

### 2. Compile gate programs to IOP

Run the compiler to generate the IOP file used at inference time:

```bash
python groqQCsim.py
# The output IOP is written to build_iop/
# Rename it to encode the circuit parameters, e.g.:
mv build_iop/QCsim_multi_program.0.iop build_iop/QCsim_multi_program.17.768.iop
```

The filename convention is `QCsim_multi_program.<num_qubits>.<max_gates>.iop`.

### 3. Run with SQUANDER integration

```bash
git clone <SQUANDER repo> -b gm_dfe_sv

# Build SQUANDER (example for the Groq server)
export TBB_LIB_DIR=/usr/lib/x86_64-linux-gnu
export TBB_INC_DIR=/usr/include
export QGD_DFE=1
python setup.py build_ext

# Edit examples/VQE/Heisenberg_VQE.py to set qbit_num = 17 (or your target)
LD_LIBRARY_PATH=/path/to/GroqQCSim/build:$LD_LIBRARY_PATH \
    python examples/VQE/Heisenberg_VQE.py
```

### 4. Standalone simulation test

```bash
python state_transform.py
```

This generates a random quantum circuit, compiles it, and runs a state vector simulation on the Groq device, comparing results against a CPU reference.

---

## License

GPL-3.0 — see [LICENSE](LICENSE).
