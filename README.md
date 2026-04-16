# CogGML

CogPy cognitive architecture modules + [ggml](https://github.com/ggml-org/ggml) tensor library.

## CogPy Header-Only C++11 Library Suite

A unified, header-only C++11 library implementing all CogPy cognitive architecture modules with **zero external dependencies**.

### Modules

| Module | Namespace | Description |
|--------|-----------|-------------|
| **core** | `cog` | AtomSpace, Atom, TruthValue, AttentionValue, Arena, Spinlock |
| **plan9** | `cog::plan9` | Plan 9 cognitive OS — 9P2000 protocol, CogFS, MachSpace |
| **pilot** | `cog::pilot` | Deep Tree Echo reservoir — A000081, B-Series, J-Surface, P-System, ESN |
| **mach** | `cog::mach` | Mach microkernel cognitive — Q16.16 fixed-point, IPC, VM, kernel AtomSpace |
| **lux** | `cog::lux` | Cognitive node graph — typed nodes/edges, BFS/DFS, PageRank, DOT export |
| **glow** | `cog::glow` | Neural network compiler — graph IR, optimization passes, interpreter |
| **gml** | `cog::gml` | Tensor library — N-dim tensors, Q4/Q8 quantization, Adam/L-BFGS |
| **prime** | `cog::prime` | AGI architecture — cognitive cycle, PLN, pattern matching, memory systems |
| **webvm** | `cog::webvm` | Web AtomSpace VM — S-expression parser, Scheme REPL, JSON serialization |
| **fowler** | `cog::fowler` | Thomas Fowler's 1840 balanced ternary calculating machine — BalancedTernary arithmetic, FowlerMachine simulation |
| **tq** | `cog::tq` | Log2(3) ternary quantization — 5 trits/byte packing, BlockTQ, TernaryMLP, entropy analysis |
| **dte** | `cog::dte` | Dense MLP autoencoder — DTE identity backup/restore, AAR vector serialization, online SGD |
| **npu** | `cog::npu` | Tree-Polytope NPU — HarmonicKernel, MatulaDecoder, SkillmISA, SimplexGeometry |
| **inference** | `cog::inference` | Unified multi-model DTE inference engine — 4-model Echobeats pipeline, DualPoolReservoir, AutognosisCycle |

### Quick Start

```cpp
#include <cog/cog.hpp>  // Everything

int main() {
    cog::AtomSpace as;
    auto cat = as.add_node(cog::AtomType::CONCEPT_NODE, "cat");
    auto animal = as.add_node(cog::AtomType::CONCEPT_NODE, "animal");
    as.add_link(cog::AtomType::INHERITANCE_LINK, {cat, animal});
    return 0;
}
```

```bash
g++ -std=c++11 -I include -o myapp myapp.cpp
```

## ggml — Tensor Library for Machine Learning

[Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205)

***Note that this project is under active development. \
Some of the development is currently happening in the [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) repos***

### Features

- Low-level cross-platform implementation
- Integer quantization support
- Broad hardware support
- Automatic differentiation
- ADAM and L-BFGS optimizers
- No third-party dependencies
- Zero memory allocations during runtime

## Build & Test

```bash
# Build everything (ggml + CogPy)
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8

# Run CogPy tests
ctest

# Or build CogPy headers-only test directly
g++ -std=c++11 -I include -o test_all test/test_all.cpp && ./test_all
```

## GPT inference (example)

```bash
# run the GPT-2 small 117M model
../examples/gpt-2/download-ggml-model.sh 117M
./bin/gpt-2-backend -m models/gpt-2-117M/ggml-model.bin -p "This is an example"
```

For more information, checkout the corresponding programs in the [examples](examples) folder.

## Resources

- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [The GGUF file format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## License

MIT
