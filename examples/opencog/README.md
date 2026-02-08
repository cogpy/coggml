# OpenCog GGML Examples

This directory contains examples demonstrating the OpenCog cognitive architecture implemented using pure GGML tensor operations.

## Overview

OpenCog is a cognitive architecture for artificial general intelligence (AGI). This implementation uses GGML's tensor operations to provide efficient, hardware-accelerated cognitive processing.

## Architecture Components

### 1. AtomSpace
The core knowledge representation system that stores atoms (concepts, predicates, links) with:
- **Embeddings**: Vector representations using GGML tensors
- **Truth Values**: Probabilistic strength and confidence
- **Relationships**: Directed graph structure with incoming/outgoing links

### 2. Probabilistic Logic Networks (PLN)
Advanced reasoning operations:
- **Deduction**: A→B, B→C ⇒ A→C
- **Induction**: A→B, A→C ⇒ B→C (weaker inference)
- **Abduction**: A→C, B→C ⇒ A→B (hypothesis generation)
- **Revision**: Combining evidence from multiple sources
- **Modus Ponens**: A→B, A ⇒ B

### 3. Economic Attention Network (ECAN)
Attention allocation system using:
- **STI** (Short-Term Importance): Current relevance/focus
- **LTI** (Long-Term Importance): Historical importance
- **Attention Spreading**: Importance flows to connected atoms

### 4. CogServer & MindAgents
- **CogServer**: Manages cognitive cycles
- **MindAgents**: Autonomous processes that operate on the AtomSpace

## Examples

### opencog-simple
Basic demonstration of:
- Creating atoms (concepts, predicates, links)
- Querying by name and type
- Simple PLN deduction and induction
- Running MindAgents in cognitive cycles

```bash
./bin/opencog-simple
```

### opencog-reasoning
Advanced reasoning demonstration:
- Building taxonomic knowledge hierarchies
- Syllogistic reasoning (finding inference chains)
- Pattern matching and similarity detection
- Multi-cycle knowledge expansion

```bash
./bin/opencog-reasoning
```

### opencog-advanced
Comprehensive feature demonstration:
- **Embedding-based similarity**: Using cosine similarity on atom embeddings
- **Advanced PLN operations**: Abduction, revision, modus ponens
- **ECAN attention allocation**: Managing cognitive resources
- **Knowledge graph reasoning**: AI and cognition domain

```bash
./bin/opencog-advanced
```

### opencog-hebbian
Hebbian learning demonstration ("Neurons that fire together, wire together"):
- **Semantic association learning**: Strengthening connections through co-activation
- **Link-based learning**: Propagating similarity through relationships
- **Embedding normalization**: Maintaining numerical stability
- **Multi-concept scenarios**: Learning complex patterns from experience

```bash
./bin/opencog-hebbian
```

## Key Features

### Tensor-Based Embeddings
Each atom has a learned vector embedding that captures semantic meaning:
- Initialized from type embeddings + random perturbation
- Links blend embeddings of their target atoms
- Used for pattern matching and similarity computation

### Pattern Matching
Efficient similarity-based pattern matching using:
- Cosine similarity between embeddings
- Threshold-based filtering
- Sorted results by relevance

### Attention Dynamics
ECAN manages limited cognitive resources:
- High STI atoms get more processing
- LTI accumulates over time for important atoms
- Attention spreads through graph connections

### Hebbian Learning
Adaptive learning mechanism based on co-activation:
- **Hebbian principle**: "Neurons that fire together, wire together"
- **Embedding updates**: Strengthen connections between co-activated atoms
- **Link propagation**: Learn from structural relationships in the graph
- **Normalization**: Maintain numerical stability during learning

## Implementation Details

### Memory Efficiency
- Embeddings stored as std::vector<float> for efficiency
- GGML tensors used for batch operations
- Type embeddings shared across atoms

### Reasoning Accuracy
- Truth values use strength [0,1] and confidence [0,1]
- PLN formulas balance inference quality with uncertainty
- Confidence decreases for weaker inference types

### Performance
- O(1) atom access by ID
- O(n) queries by name/type with indexing
- O(n²) similarity computation (can be optimized with ANN)

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build . --target opencog-simple opencog-reasoning opencog-advanced
```

## Future Enhancements

Potential improvements to continue implementation:
- [x] Hebbian learning for embedding updates
- [ ] Temporal reasoning and event sequences
- [ ] Goal-directed attention allocation
- [ ] Pattern mining and concept formation
- [ ] Integration with neural networks
- [ ] Distributed AtomSpace
- [ ] GGML compute graph integration for inference
- [ ] Quantized embeddings for memory efficiency

## References

- [OpenCog Foundation](https://opencog.org/)
- [PLN Book](https://wiki.opencog.org/w/PLNBook)
- [ECAN Paper](https://wiki.opencog.org/w/ECAN)
- [AtomSpace Design](https://wiki.opencog.org/w/AtomSpace)
