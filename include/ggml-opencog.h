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
    GGML_OPENCOG_TIME_NODE = 5,
    GGML_OPENCOG_SEQUENTIAL_LINK = 6,
    GGML_OPENCOG_SIMULTANEOUS_LINK = 7,
    GGML_OPENCOG_AT_TIME_LINK = 8,
    GGML_OPENCOG_ATOM_TYPE_COUNT = 9
};

// Time interval structure for temporal reasoning
struct ggml_opencog_time_interval {
    int64_t start_time;  // Unix timestamp in milliseconds
    int64_t end_time;    // Unix timestamp in milliseconds
    bool is_point;       // True if this represents a point in time (start == end)
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
    struct ggml_tensor* embedding;             // Vector embedding of the atom (can be nullptr)
    std::vector<float> embedding_data;         // Direct storage of embedding data
    std::vector<uint64_t> outgoing;            // IDs of atoms this links to
    std::vector<uint64_t> incoming;            // IDs of atoms that link to this
    
    // ECAN (Economic Attention Network) values
    float sti;                                 // Short-term importance (attention)
    float lti;                                 // Long-term importance (memory worthiness)
    float vlti;                                // Very long-term importance
    
    // Temporal information (optional)
    struct ggml_opencog_time_interval* time_interval;  // Time interval for temporal atoms
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

struct ggml_opencog_truth_value ggml_opencog_pln_abduction(struct ggml_opencog_truth_value premise1,
                                                          struct ggml_opencog_truth_value premise2);

struct ggml_opencog_truth_value ggml_opencog_pln_revision(struct ggml_opencog_truth_value tv1,
                                                         struct ggml_opencog_truth_value tv2);

struct ggml_opencog_truth_value ggml_opencog_pln_modus_ponens(struct ggml_opencog_truth_value implication,
                                                              struct ggml_opencog_truth_value antecedent);

// Similarity computation using embeddings
float ggml_opencog_compute_similarity(struct ggml_opencog_atomspace* atomspace,
                                     uint64_t atom1_id,
                                     uint64_t atom2_id);

// Attention allocation (ECAN)
void ggml_opencog_update_attention(struct ggml_opencog_atomspace* atomspace,
                                  uint64_t atom_id,
                                  float sti_delta,
                                  float lti_delta);

// Hebbian learning for embedding updates
// Strengthens the embedding connection between two co-activated atoms
void ggml_opencog_hebbian_update(struct ggml_opencog_atomspace* atomspace,
                                uint64_t atom1_id,
                                uint64_t atom2_id,
                                float learning_rate);

// Apply Hebbian learning to all atoms in a link
// Updates embeddings based on co-activation patterns
void ggml_opencog_hebbian_update_link(struct ggml_opencog_atomspace* atomspace,
                                     uint64_t link_id,
                                     float learning_rate);

// Normalize embedding vectors to unit length
void ggml_opencog_normalize_embedding(struct ggml_opencog_atomspace* atomspace,
                                     uint64_t atom_id);

// Temporal reasoning functions
// Add a time interval to an atom
void ggml_opencog_set_time_interval(struct ggml_opencog_atomspace* atomspace,
                                   uint64_t atom_id,
                                   int64_t start_time,
                                   int64_t end_time);

// Get atoms that overlap with a time interval
std::vector<uint64_t> ggml_opencog_get_atoms_at_time(struct ggml_opencog_atomspace* atomspace,
                                                      int64_t time);

std::vector<uint64_t> ggml_opencog_get_atoms_in_interval(struct ggml_opencog_atomspace* atomspace,
                                                         int64_t start_time,
                                                         int64_t end_time);

// Temporal ordering queries
bool ggml_opencog_happens_before(struct ggml_opencog_atomspace* atomspace,
                                 uint64_t atom1_id,
                                 uint64_t atom2_id);

bool ggml_opencog_happens_during(struct ggml_opencog_atomspace* atomspace,
                                uint64_t atom1_id,
                                uint64_t atom2_id);

bool ggml_opencog_happens_simultaneously(struct ggml_opencog_atomspace* atomspace,
                                        uint64_t atom1_id,
                                        uint64_t atom2_id,
                                        int64_t tolerance_ms);

// Temporal inference - derive sequential relationships
struct ggml_opencog_truth_value ggml_opencog_temporal_induction(
                                                struct ggml_opencog_truth_value before_link,
                                                struct ggml_opencog_truth_value after_link);

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