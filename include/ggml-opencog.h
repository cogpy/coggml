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
    // Extended atom types for richer knowledge representation
    GGML_OPENCOG_VARIABLE_NODE = 9,      // Variable atom for pattern matching
    GGML_OPENCOG_NUMBER_NODE = 10,       // Numeric value atom
    GGML_OPENCOG_AND_LINK = 11,          // Logical AND of members
    GGML_OPENCOG_OR_LINK = 12,           // Logical OR of members
    GGML_OPENCOG_NOT_LINK = 13,          // Logical NOT of single member
    GGML_OPENCOG_IMPLICATION_LINK = 14,  // Probabilistic implication A -> B
    GGML_OPENCOG_LIST_LINK = 15,         // Ordered list of atoms
    GGML_OPENCOG_ATOM_TYPE_COUNT = 16
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

// Variable binding for pattern matching with VariableNode atoms
struct ggml_opencog_binding {
    std::unordered_map<uint64_t, uint64_t> bindings;  // variable_id -> matched_atom_id

    bool bind(uint64_t var_id, uint64_t atom_id) {
        auto it = bindings.find(var_id);
        if (it != bindings.end()) {
            return it->second == atom_id;  // consistent if already bound to same
        }
        bindings[var_id] = atom_id;
        return true;
    }

    uint64_t get(uint64_t var_id) const {
        auto it = bindings.find(var_id);
        return (it != bindings.end()) ? it->second : 0;
    }

    bool has(uint64_t var_id) const {
        return bindings.find(var_id) != bindings.end();
    }

    size_t size() const { return bindings.size(); }
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

// Extended PLN rules: conjunction, disjunction, negation
struct ggml_opencog_truth_value ggml_opencog_pln_conjunction(
    struct ggml_opencog_truth_value tv1,
    struct ggml_opencog_truth_value tv2);

struct ggml_opencog_truth_value ggml_opencog_pln_disjunction(
    struct ggml_opencog_truth_value tv1,
    struct ggml_opencog_truth_value tv2);

struct ggml_opencog_truth_value ggml_opencog_pln_negation(
    struct ggml_opencog_truth_value tv);

// Forward chaining: iteratively apply PLN deduction to derive new InheritanceLinks.
// Returns IDs of newly derived atoms. Stops when no new derivations are possible
// or max_iterations is reached.
std::vector<uint64_t> ggml_opencog_forward_chain(
    struct ggml_opencog_atomspace* atomspace,
    int max_iterations);

// Backward chaining: attempt to prove that goal_atom can be derived from existing
// knowledge using PLN deduction chains up to max_depth. Fills derivation_path with
// intermediate atom IDs used in the proof. Returns true if goal can be derived.
bool ggml_opencog_backward_chain(
    struct ggml_opencog_atomspace* atomspace,
    uint64_t goal_id,
    int max_depth,
    std::vector<uint64_t>& derivation_path);

// Pattern matching with variable binding: check whether pattern_id (which may
// contain VARIABLE_NODE atoms) structurally matches candidate_id. Fills binding
// with any variable assignments. Returns true on match.
bool ggml_opencog_match_with_binding(
    struct ggml_opencog_atomspace* atomspace,
    uint64_t pattern_id,
    uint64_t candidate_id,
    struct ggml_opencog_binding* binding);

// Find all atoms in the atomspace that match pattern_id (may contain variables).
// Returns vector of (matched_atom_id, binding) pairs.
std::vector<std::pair<uint64_t, struct ggml_opencog_binding>>
ggml_opencog_find_matching(
    struct ggml_opencog_atomspace* atomspace,
    uint64_t pattern_id);

// Get the total number of atoms in the atomspace
size_t ggml_opencog_atom_count(struct ggml_opencog_atomspace* atomspace);

#ifdef __cplusplus
}
#endif