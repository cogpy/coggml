#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

// OpenCog Atom Types
enum ggml_opencog_atom_type {
    GGML_OPENCOG_CONCEPT_NODE = 0,
    GGML_OPENCOG_PREDICATE_NODE = 1,
    GGML_OPENCOG_EVALUATION_LINK = 2,
    GGML_OPENCOG_INHERITANCE_LINK = 3,
    GGML_OPENCOG_SIMILARITY_LINK = 4,
    GGML_OPENCOG_ATOM_TYPE_COUNT = 5
};

// Truth Value structure for atoms
struct ggml_opencog_truth_value {
    float strength;     // [0.0, 1.0] - confidence in the truth
    float confidence;   // [0.0, 1.0] - amount of evidence
};

// Atom structure using GGML tensors
struct ggml_opencog_atom {
    uint64_t id;                                // Unique atom identifier
    enum ggml_opencog_atom_type type;          // Type of atom
    char name[256];                            // Name/label for the atom
    struct ggml_opencog_truth_value tv;       // Truth value
    struct ggml_tensor* embedding;             // Vector embedding of the atom
    std::vector<uint64_t> outgoing;            // IDs of atoms this links to
    std::vector<uint64_t> incoming;            // IDs of atoms that link to this
};

// AtomSpace - the core knowledge representation
struct ggml_opencog_atomspace {
    struct ggml_context* ctx;                  // GGML context for tensors
    ggml_backend_t backend;                    // Backend for computation
    ggml_backend_buffer_t buffer;              // Memory buffer
    
    std::unordered_map<uint64_t, std::unique_ptr<ggml_opencog_atom>> atoms;
    std::unordered_map<std::string, std::vector<uint64_t>> name_index;
    std::unordered_map<enum ggml_opencog_atom_type, std::vector<uint64_t>> type_index;
    
    uint64_t next_atom_id;                     // Next available atom ID
    int embedding_dim;                         // Dimension of atom embeddings
    
    struct ggml_tensor* type_embeddings;       // Learnable type embeddings
    struct ggml_tensor* atom_matrix;           // Matrix of all atom embeddings
};

// MindAgent interface for cognitive processes
struct ggml_opencog_mind_agent {
    char name[128];
    void (*process)(struct ggml_opencog_atomspace* atomspace);
    int frequency;                             // How often to run (in cycles)
    int last_run;                             // Last cycle this agent ran
};

// CogServer for managing agents and atomspace
struct ggml_opencog_cogserver {
    struct ggml_opencog_atomspace* atomspace;
    std::vector<struct ggml_opencog_mind_agent*> agents;
    int cycle_count;
    bool running;
};

// AtomSpace functions
struct ggml_opencog_atomspace* ggml_opencog_atomspace_new(int embedding_dim);
void ggml_opencog_atomspace_free(struct ggml_opencog_atomspace* atomspace);

// Atom manipulation
uint64_t ggml_opencog_add_atom(struct ggml_opencog_atomspace* atomspace,
                               enum ggml_opencog_atom_type type,
                               const char* name,
                               struct ggml_opencog_truth_value tv,
                               const std::vector<uint64_t>& outgoing);

struct ggml_opencog_atom* ggml_opencog_get_atom(struct ggml_opencog_atomspace* atomspace, uint64_t id);
bool ggml_opencog_remove_atom(struct ggml_opencog_atomspace* atomspace, uint64_t id);

// Query functions
std::vector<uint64_t> ggml_opencog_get_atoms_by_name(struct ggml_opencog_atomspace* atomspace, const char* name);
std::vector<uint64_t> ggml_opencog_get_atoms_by_type(struct ggml_opencog_atomspace* atomspace, enum ggml_opencog_atom_type type);

// Pattern matching
std::vector<uint64_t> ggml_opencog_pattern_match(struct ggml_opencog_atomspace* atomspace,
                                                  const struct ggml_tensor* pattern);

// Reasoning operations
struct ggml_opencog_truth_value ggml_opencog_pln_deduction(struct ggml_opencog_truth_value premise1,
                                                           struct ggml_opencog_truth_value premise2);

struct ggml_opencog_truth_value ggml_opencog_pln_induction(struct ggml_opencog_truth_value premise1,
                                                          struct ggml_opencog_truth_value premise2);

// CogServer functions
struct ggml_opencog_cogserver* ggml_opencog_cogserver_new(struct ggml_opencog_atomspace* atomspace);
void ggml_opencog_cogserver_free(struct ggml_opencog_cogserver* server);
void ggml_opencog_cogserver_add_agent(struct ggml_opencog_cogserver* server, struct ggml_opencog_mind_agent* agent);
void ggml_opencog_cogserver_run_cycle(struct ggml_opencog_cogserver* server);
void ggml_opencog_cogserver_start(struct ggml_opencog_cogserver* server);
void ggml_opencog_cogserver_stop(struct ggml_opencog_cogserver* server);

#ifdef __cplusplus
}
#endif