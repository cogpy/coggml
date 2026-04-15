# DTE Special Tokens

The custom DTE BPE tokenizer includes 35 special tokens that encode primitives of the Deep Tree Echo cognitive architecture. These tokens are essential for training the model to understand and generate text related to its own internal structure and processes.

| Category | Token | Description |
| :--- | :--- | :--- |
| **Control** | `<|pad|>` | Padding token. |
| | `<|endoftext|>` | End of text token. |
| | `<|startoftext|>` | Start of text token. |
| **Cognitive** | `<|echo|>` | Represents the core echo process. |
| | `<|deep_tree|>` | Refers to the Deep Tree Echo architecture. |
| | `<|reservoir|>` | Represents the echo state network reservoir. |
| | `<|membrane|>` | Refers to the cognitive membrane. |
| | `<|hypergraph|>` | Represents the hypergraph memory system. |
| | `<|atomspace|>` | Refers to the OpenCog AtomSpace. |
| **AAR Triad** | `<|agent|>` | The agent component of the Agent-Arena-Relation triad. |
| | `<|arena|>` | The arena component of the AAR triad. |
| | `<|relation|>` | The relation component of the AAR triad. |
| **Cognitive State** | `<|introspect|>` | Triggers an introspection cycle. |
| | `<|perceive|>` | Represents the perception process. |
| | `<|act|>` | Represents the action process. |
| | `<|simulate|>` | Triggers a simulation or planning process. |
| **Echobeat** | `<|echobeat_start|>` | Marks the beginning of an echobeat cycle. |
| | `<|echobeat_end|>` | Marks the end of an echobeat cycle. |
| | `<|step|>` | Represents a single step within an echobeat. |
| **Persona** | `<|persona|>` | General persona marker. |
| | `<|cognitive|>` | Cognitive persona dimension. |
| | `<|adaptive|>` | Adaptive persona dimension. |
| | `<|recursive|>` | Recursive persona dimension. |
| | `<|synergistic|>` | Synergistic persona dimension. |
| | `<|holographic|>` | Holographic persona dimension. |
| **Structural** | `<|feedback|>` | Represents a feedback loop. |
| | `<|feedforward|>` | Represents a feedforward process. |
| | `<|resonance|>` | Refers to harmonic resonance in the ESN. |
| | `<|entelechy|>` | Represents the self-organizing principle. |
| **Conversation** | `<|user|>` | Marks user input. |
| | `<|echo_self|>` | Marks model output. |
| | `<|system|>` | Marks system messages. |
| **Architecture** | `<|b9|>` | b-series rooted trees (Plan 9). |
| | `<|p9|>` | p-system nested scopes (Plan 9). |
| | `<|j9|>` | j-surface elementary differentials (Plan 9). |
