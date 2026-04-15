#include "ggml-opencog.h"
#include "ggml-cpu.h"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>

// Helper function to compute cosine similarity
static float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);
    
    if (norm_a < 1e-8f || norm_b < 1e-8f) return 0.0f;
    
    return dot / (norm_a * norm_b);
}

// Initialize AtomSpace
struct ggml_opencog_atomspace* ggml_opencog_atomspace_new(int embedding_dim) {
    auto* atomspace = new ggml_opencog_atomspace();
    
    atomspace->next_atom_id = 1;
    atomspace->embedding_dim = embedding_dim;
    
    // Initialize GGML backend
    atomspace->backend = ggml_backend_cpu_init();
    
    // Calculate memory requirements
    size_t ctx_size = 0;
    ctx_size += GGML_OPENCOG_ATOM_TYPE_COUNT * embedding_dim * sizeof(float); // type embeddings
    ctx_size += 10000 * embedding_dim * sizeof(float); // space for 10k atoms initially
    ctx_size += ggml_tensor_overhead() * 2; // tensors
    ctx_size += 1024; // overhead
    
    struct ggml_init_params params;
    params.mem_size = ctx_size;
    params.mem_buffer = NULL;
    params.no_alloc = true;
    
    atomspace->ctx = ggml_init(params);
    
    // Create type embeddings tensor
    atomspace->type_embeddings = ggml_new_tensor_2d(atomspace->ctx, GGML_TYPE_F32, 
                                                    embedding_dim, GGML_OPENCOG_ATOM_TYPE_COUNT);
    
    // Create atom matrix (will grow as needed)
    atomspace->atom_matrix = ggml_new_tensor_2d(atomspace->ctx, GGML_TYPE_F32, 
                                               embedding_dim, 10000);
    
    // Allocate tensors
    atomspace->buffer = ggml_backend_alloc_ctx_tensors(atomspace->ctx, atomspace->backend);
    
    // Initialize type embeddings with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    std::vector<float> type_init_data(GGML_OPENCOG_ATOM_TYPE_COUNT * embedding_dim);
    for (size_t i = 0; i < type_init_data.size(); i++) {
        type_init_data[i] = dist(gen);
    }
    
    ggml_backend_tensor_set(atomspace->type_embeddings, type_init_data.data(), 0, 
                           ggml_nbytes(atomspace->type_embeddings));
    
    return atomspace;
}

void ggml_opencog_atomspace_free(struct ggml_opencog_atomspace* atomspace) {
    if (!atomspace) return;
    
    atomspace->atoms.clear();
    atomspace->name_index.clear();
    atomspace->type_index.clear();
    
    ggml_backend_buffer_free(atomspace->buffer);
    ggml_backend_free(atomspace->backend);
    ggml_free(atomspace->ctx);
    
    delete atomspace;
}

// Add an atom to the atomspace
uint64_t ggml_opencog_add_atom(struct ggml_opencog_atomspace* atomspace,
                               enum ggml_opencog_atom_type type,
                               const char* name,
                               struct ggml_opencog_truth_value tv,
                               const std::vector<uint64_t>& outgoing) {
    
    uint64_t id = atomspace->next_atom_id++;
    
    auto atom = std::make_unique<ggml_opencog_atom>();
    atom->id = id;
    atom->type = type;
    strncpy(atom->name, name, sizeof(atom->name) - 1);
    atom->name[sizeof(atom->name) - 1] = '\0';
    atom->tv = tv;
    atom->outgoing = outgoing;
    
    // Initialize ECAN attention values
    atom->sti = 0.0f;    // Neutral attention initially
    atom->lti = 0.0f;    // No long-term importance yet
    atom->vlti = 0.0f;   // No very long-term importance yet
    
    // Initialize temporal information
    atom->time_interval = nullptr;  // No time interval by default
    
    // Initialize embedding based on type and name
    std::vector<float> embedding_data(atomspace->embedding_dim);
    
    // Get type embedding
    std::vector<float> type_embedding(atomspace->embedding_dim);
    ggml_backend_tensor_get(atomspace->type_embeddings, type_embedding.data(), 
                           type * atomspace->embedding_dim * sizeof(float), 
                           atomspace->embedding_dim * sizeof(float));
    
    // Hash-based initialization for name
    std::hash<std::string> hasher;
    size_t name_hash = hasher(name);
    std::mt19937 gen(name_hash);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (int i = 0; i < atomspace->embedding_dim; i++) {
        embedding_data[i] = type_embedding[i] + dist(gen);
    }
    
    // For links, incorporate embeddings of connected atoms
    if (!outgoing.empty()) {
        std::vector<float> combined_embedding(atomspace->embedding_dim, 0.0f);
        for (uint64_t target_id : outgoing) {
            auto target_it = atomspace->atoms.find(target_id);
            if (target_it != atomspace->atoms.end() && !target_it->second->embedding_data.empty()) {
                const auto& target_emb = target_it->second->embedding_data;
                for (int i = 0; i < atomspace->embedding_dim; i++) {
                    combined_embedding[i] += target_emb[i];
                }
            }
        }
        
        // Average and blend with type embedding
        float blend_factor = 0.7f;
        for (int i = 0; i < atomspace->embedding_dim; i++) {
            embedding_data[i] = blend_factor * embedding_data[i] + 
                              (1.0f - blend_factor) * combined_embedding[i] / outgoing.size();
        }
    }
    
    // Store embedding data directly in the atom for efficient access.
    // We use std::vector<float> rather than ggml_tensor* to avoid
    // the complexity of managing individual tensor contexts and buffers
    // for potentially thousands of atoms. This approach:
    // - Simplifies memory management (no buffer allocation per atom)
    // - Provides faster access for similarity computations
    // - Allows batch operations when needed via atom_matrix tensor
    atom->embedding_data = embedding_data;
    atom->embedding = nullptr;  // Tensor field reserved for future batch ops
    
    // Update incoming links for target atoms
    for (uint64_t target_id : outgoing) {
        auto it = atomspace->atoms.find(target_id);
        if (it != atomspace->atoms.end()) {
            it->second->incoming.push_back(id);
        }
    }
    
    // Update indexes
    atomspace->name_index[name].push_back(id);
    atomspace->type_index[type].push_back(id);
    
    // Store the atom
    atomspace->atoms[id] = std::move(atom);
    
    return id;
}

struct ggml_opencog_atom* ggml_opencog_get_atom(struct ggml_opencog_atomspace* atomspace, uint64_t id) {
    auto it = atomspace->atoms.find(id);
    return (it != atomspace->atoms.end()) ? it->second.get() : nullptr;
}

bool ggml_opencog_remove_atom(struct ggml_opencog_atomspace* atomspace, uint64_t id) {
    auto it = atomspace->atoms.find(id);
    if (it == atomspace->atoms.end()) return false;
    
    auto& atom = it->second;
    
    // Remove from incoming links of target atoms
    for (uint64_t target_id : atom->outgoing) {
        auto target_it = atomspace->atoms.find(target_id);
        if (target_it != atomspace->atoms.end()) {
            auto& incoming = target_it->second->incoming;
            incoming.erase(std::remove(incoming.begin(), incoming.end(), id), incoming.end());
        }
    }
    
    // Remove from outgoing links of source atoms
    for (uint64_t source_id : atom->incoming) {
        auto source_it = atomspace->atoms.find(source_id);
        if (source_it != atomspace->atoms.end()) {
            auto& outgoing = source_it->second->outgoing;
            outgoing.erase(std::remove(outgoing.begin(), outgoing.end(), id), outgoing.end());
        }
    }
    
    // Remove from indexes
    auto& name_vec = atomspace->name_index[atom->name];
    name_vec.erase(std::remove(name_vec.begin(), name_vec.end(), id), name_vec.end());
    
    auto& type_vec = atomspace->type_index[atom->type];
    type_vec.erase(std::remove(type_vec.begin(), type_vec.end(), id), type_vec.end());
    
    // Clean up temporal data if present
    if (atom->time_interval) {
        delete atom->time_interval;
    }
    
    // Remove the atom
    atomspace->atoms.erase(it);
    
    return true;
}

// Query functions
std::vector<uint64_t> ggml_opencog_get_atoms_by_name(struct ggml_opencog_atomspace* atomspace, const char* name) {
    auto it = atomspace->name_index.find(name);
    return (it != atomspace->name_index.end()) ? it->second : std::vector<uint64_t>();
}

std::vector<uint64_t> ggml_opencog_get_atoms_by_type(struct ggml_opencog_atomspace* atomspace, enum ggml_opencog_atom_type type) {
    auto it = atomspace->type_index.find(type);
    return (it != atomspace->type_index.end()) ? it->second : std::vector<uint64_t>();
}

// Pattern matching based on embedding similarity using cosine similarity
std::vector<uint64_t> ggml_opencog_pattern_match(struct ggml_opencog_atomspace* atomspace,
                                                  const struct ggml_tensor* pattern) {
    std::vector<std::pair<uint64_t, float>> matches_with_scores;
    
    // Get pattern data
    int pattern_dim = ggml_nelements(pattern);
    if (pattern_dim != atomspace->embedding_dim) {
        // Pattern dimension doesn't match atomspace embedding dimension
        return std::vector<uint64_t>();
    }
    
    std::vector<float> pattern_data(pattern_dim);
    // For patterns created with ggml_new_tensor_*d with immediate data,
    // pattern->data will be non-null. For patterns from backend tensors,
    // we need to use ggml_backend_tensor_get.
    if (pattern->data) {
        memcpy(pattern_data.data(), pattern->data, pattern_dim * sizeof(float));
    } else {
        ggml_backend_tensor_get(pattern, pattern_data.data(), 0, ggml_nbytes(pattern));
    }
    
    const float similarity_threshold = 0.7f;
    
    // Compare against all atoms using cosine similarity
    for (const auto& [id, atom] : atomspace->atoms) {
        if (atom->embedding_data.empty()) {
            continue;
        }
        
        // Calculate cosine similarity using the embedding data
        float similarity = cosine_similarity(pattern_data.data(), atom->embedding_data.data(), 
                                            atomspace->embedding_dim);
        
        if (similarity >= similarity_threshold) {
            matches_with_scores.push_back({id, similarity});
        }
    }
    
    // Sort by similarity (descending order)
    std::sort(matches_with_scores.begin(), matches_with_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Extract just the IDs
    std::vector<uint64_t> matches;
    matches.reserve(matches_with_scores.size());
    for (const auto& [id, score] : matches_with_scores) {
        matches.push_back(id);
    }
    
    return matches;
}

// PLN (Probabilistic Logic Networks) reasoning functions

// Deduction: A->B and B->C implies A->C
struct ggml_opencog_truth_value ggml_opencog_pln_deduction(struct ggml_opencog_truth_value premise1,
                                                           struct ggml_opencog_truth_value premise2) {
    // Advanced deduction using PLN formulas
    // Strength: s_AB * s_BC
    // Confidence: c_AB * c_BC * s_AB * s_BC
    
    struct ggml_opencog_truth_value result;
    result.strength = premise1.strength * premise2.strength;
    result.confidence = fminf(premise1.confidence, premise2.confidence) * result.strength;
    
    return result;
}

// Induction: A->B and A->C suggests B->C
struct ggml_opencog_truth_value ggml_opencog_pln_induction(struct ggml_opencog_truth_value premise1,
                                                          struct ggml_opencog_truth_value premise2) {
    // Induction is weaker than deduction
    // Simplified formula
    
    struct ggml_opencog_truth_value result;
    result.strength = (premise1.strength + premise2.strength) / 2.0f;
    result.confidence = fminf(premise1.confidence, premise2.confidence) * 0.5f;
    
    return result;
}

// Abduction: A->C and B->C suggests A->B
struct ggml_opencog_truth_value ggml_opencog_pln_abduction(struct ggml_opencog_truth_value premise1,
                                                          struct ggml_opencog_truth_value premise2) {
    // Abduction is similar to induction but works backward
    struct ggml_opencog_truth_value result;
    result.strength = (premise1.strength * premise2.strength + 
                      (1.0f - premise1.strength) * (1.0f - premise2.strength)) / 2.0f;
    result.confidence = fminf(premise1.confidence, premise2.confidence) * 0.4f; // Even weaker than induction
    
    return result;
}

// Revision: Combine two truth values for the same proposition
struct ggml_opencog_truth_value ggml_opencog_pln_revision(struct ggml_opencog_truth_value tv1,
                                                         struct ggml_opencog_truth_value tv2) {
    // Weighted average based on confidence
    float total_confidence = tv1.confidence + tv2.confidence;
    
    struct ggml_opencog_truth_value result;
    if (total_confidence > 0.0f) {
        result.strength = (tv1.strength * tv1.confidence + tv2.strength * tv2.confidence) / total_confidence;
        result.confidence = fminf(1.0f, total_confidence); // Can't exceed 1.0
    } else {
        result.strength = 0.5f;
        result.confidence = 0.0f;
    }
    
    return result;
}

// Modus Ponens: A->B and A implies B
struct ggml_opencog_truth_value ggml_opencog_pln_modus_ponens(struct ggml_opencog_truth_value implication,
                                                              struct ggml_opencog_truth_value antecedent) {
    struct ggml_opencog_truth_value result;
    result.strength = implication.strength * antecedent.strength;
    result.confidence = implication.confidence * antecedent.confidence;
    
    return result;
}

// Compute similarity between two atoms using their embeddings
float ggml_opencog_compute_similarity(struct ggml_opencog_atomspace* atomspace,
                                     uint64_t atom1_id,
                                     uint64_t atom2_id) {
    auto* atom1 = ggml_opencog_get_atom(atomspace, atom1_id);
    auto* atom2 = ggml_opencog_get_atom(atomspace, atom2_id);
    
    if (!atom1 || !atom2 || atom1->embedding_data.empty() || atom2->embedding_data.empty()) {
        return 0.0f;
    }
    
    return cosine_similarity(atom1->embedding_data.data(), atom2->embedding_data.data(), 
                           atomspace->embedding_dim);
}

// ECAN: Update attention values for an atom
void ggml_opencog_update_attention(struct ggml_opencog_atomspace* atomspace,
                                  uint64_t atom_id,
                                  float sti_delta,
                                  float lti_delta) {
    auto* atom = ggml_opencog_get_atom(atomspace, atom_id);
    if (!atom) return;
    
    // Update short-term importance (clamped to reasonable range)
    atom->sti += sti_delta;
    atom->sti = fmaxf(-100.0f, fminf(100.0f, atom->sti));
    
    // Update long-term importance (typically only increases)
    atom->lti += lti_delta;
    atom->lti = fmaxf(0.0f, fminf(100.0f, atom->lti));
    
    // Attention spreading: increase STI of connected atoms slightly
    const float spread_factor = 0.1f;
    for (uint64_t target_id : atom->outgoing) {
        auto* target = ggml_opencog_get_atom(atomspace, target_id);
        if (target) {
            target->sti += sti_delta * spread_factor;
            target->sti = fmaxf(-100.0f, fminf(100.0f, target->sti));
        }
    }
}

// Hebbian learning implementation
// "Neurons that fire together, wire together" - Donald Hebb
// When two atoms are co-activated, their embeddings are adjusted to become more similar
void ggml_opencog_hebbian_update(struct ggml_opencog_atomspace* atomspace,
                                uint64_t atom1_id,
                                uint64_t atom2_id,
                                float learning_rate) {
    auto* atom1 = ggml_opencog_get_atom(atomspace, atom1_id);
    auto* atom2 = ggml_opencog_get_atom(atomspace, atom2_id);
    
    if (!atom1 || !atom2) return;
    if (atom1->embedding_data.empty() || atom2->embedding_data.empty()) return;
    if (atom1->embedding_data.size() != atom2->embedding_data.size()) return;
    
    int dim = atom1->embedding_data.size();
    
    // Hebbian update: move embeddings closer together
    // e1' = e1 + lr * (e2 - e1) = e1 + lr * e2 - lr * e1
    // e2' = e2 + lr * (e1 - e2) = e2 + lr * e1 - lr * e2
    for (int i = 0; i < dim; i++) {
        float delta1 = learning_rate * (atom2->embedding_data[i] - atom1->embedding_data[i]);
        float delta2 = learning_rate * (atom1->embedding_data[i] - atom2->embedding_data[i]);
        
        atom1->embedding_data[i] += delta1;
        atom2->embedding_data[i] += delta2;
    }
}

// Apply Hebbian learning to all atoms connected by a link
// This strengthens the semantic relationships between linked concepts
void ggml_opencog_hebbian_update_link(struct ggml_opencog_atomspace* atomspace,
                                     uint64_t link_id,
                                     float learning_rate) {
    auto* link = ggml_opencog_get_atom(atomspace, link_id);
    if (!link) return;
    
    // Update embeddings between all pairs of atoms in the outgoing set
    const auto& outgoing = link->outgoing;
    for (size_t i = 0; i < outgoing.size(); i++) {
        for (size_t j = i + 1; j < outgoing.size(); j++) {
            ggml_opencog_hebbian_update(atomspace, outgoing[i], outgoing[j], learning_rate);
        }
        
        // Also update between link and its targets (with reduced rate)
        ggml_opencog_hebbian_update(atomspace, link_id, outgoing[i], learning_rate * 0.5f);
    }
}

// Normalize an atom's embedding to unit length
// This is important to maintain numerical stability and consistent similarity scores
void ggml_opencog_normalize_embedding(struct ggml_opencog_atomspace* atomspace,
                                     uint64_t atom_id) {
    auto* atom = ggml_opencog_get_atom(atomspace, atom_id);
    if (!atom || atom->embedding_data.empty()) return;
    
    int dim = atom->embedding_data.size();
    float norm = 0.0f;
    
    // Compute L2 norm
    for (int i = 0; i < dim; i++) {
        norm += atom->embedding_data[i] * atom->embedding_data[i];
    }
    norm = sqrtf(norm);
    
    // Avoid division by zero
    if (norm < 1e-8f) return;
    
    // Normalize to unit length
    for (int i = 0; i < dim; i++) {
        atom->embedding_data[i] /= norm;
    }
}

// Temporal reasoning implementation

// Set a time interval for an atom
void ggml_opencog_set_time_interval(struct ggml_opencog_atomspace* atomspace,
                                   uint64_t atom_id,
                                   int64_t start_time,
                                   int64_t end_time) {
    auto* atom = ggml_opencog_get_atom(atomspace, atom_id);
    if (!atom) return;
    
    if (!atom->time_interval) {
        atom->time_interval = new ggml_opencog_time_interval();
    }
    
    atom->time_interval->start_time = start_time;
    atom->time_interval->end_time = end_time;
    atom->time_interval->is_point = (start_time == end_time);
}

// Get atoms that exist at a specific time
std::vector<uint64_t> ggml_opencog_get_atoms_at_time(struct ggml_opencog_atomspace* atomspace,
                                                      int64_t time) {
    std::vector<uint64_t> result;
    
    for (const auto& pair : atomspace->atoms) {
        const auto& atom = pair.second;
        if (atom->time_interval) {
            if (time >= atom->time_interval->start_time && 
                time <= atom->time_interval->end_time) {
                result.push_back(atom->id);
            }
        }
    }
    
    return result;
}

// Get atoms that overlap with a time interval
std::vector<uint64_t> ggml_opencog_get_atoms_in_interval(struct ggml_opencog_atomspace* atomspace,
                                                         int64_t start_time,
                                                         int64_t end_time) {
    std::vector<uint64_t> result;
    
    for (const auto& pair : atomspace->atoms) {
        const auto& atom = pair.second;
        if (atom->time_interval) {
            // Check for interval overlap
            if (atom->time_interval->end_time >= start_time && 
                atom->time_interval->start_time <= end_time) {
                result.push_back(atom->id);
            }
        }
    }
    
    return result;
}

// Check if atom1 happens before atom2
bool ggml_opencog_happens_before(struct ggml_opencog_atomspace* atomspace,
                                 uint64_t atom1_id,
                                 uint64_t atom2_id) {
    auto* atom1 = ggml_opencog_get_atom(atomspace, atom1_id);
    auto* atom2 = ggml_opencog_get_atom(atomspace, atom2_id);
    
    if (!atom1 || !atom2) return false;
    if (!atom1->time_interval || !atom2->time_interval) return false;
    
    return atom1->time_interval->end_time <= atom2->time_interval->start_time;
}

// Check if atom1 happens during atom2
bool ggml_opencog_happens_during(struct ggml_opencog_atomspace* atomspace,
                                uint64_t atom1_id,
                                uint64_t atom2_id) {
    auto* atom1 = ggml_opencog_get_atom(atomspace, atom1_id);
    auto* atom2 = ggml_opencog_get_atom(atomspace, atom2_id);
    
    if (!atom1 || !atom2) return false;
    if (!atom1->time_interval || !atom2->time_interval) return false;
    
    return atom1->time_interval->start_time >= atom2->time_interval->start_time &&
           atom1->time_interval->end_time <= atom2->time_interval->end_time;
}

// Check if two atoms happen simultaneously (within tolerance)
bool ggml_opencog_happens_simultaneously(struct ggml_opencog_atomspace* atomspace,
                                        uint64_t atom1_id,
                                        uint64_t atom2_id,
                                        int64_t tolerance_ms) {
    auto* atom1 = ggml_opencog_get_atom(atomspace, atom1_id);
    auto* atom2 = ggml_opencog_get_atom(atomspace, atom2_id);
    
    if (!atom1 || !atom2) return false;
    if (!atom1->time_interval || !atom2->time_interval) return false;
    
    int64_t start_diff = std::abs(atom1->time_interval->start_time - atom2->time_interval->start_time);
    int64_t end_diff = std::abs(atom1->time_interval->end_time - atom2->time_interval->end_time);
    
    return start_diff <= tolerance_ms && end_diff <= tolerance_ms;
}

// Temporal induction: if A happens before B, and B happens before C,
// then we can infer A happens before C with reduced confidence
struct ggml_opencog_truth_value ggml_opencog_temporal_induction(
                                                struct ggml_opencog_truth_value before_link,
                                                struct ggml_opencog_truth_value after_link) {
    struct ggml_opencog_truth_value result;
    
    // Temporal transitivity: A→B and B→C implies A→C
    // Strength is the minimum of the two (weakest link)
    result.strength = fminf(before_link.strength, after_link.strength);
    
    // Confidence degrades with chain length
    result.confidence = before_link.confidence * after_link.confidence * 0.9f;
    
    return result;
}

// CogServer implementation
struct ggml_opencog_cogserver* ggml_opencog_cogserver_new(struct ggml_opencog_atomspace* atomspace) {
    auto* server = new ggml_opencog_cogserver();
    server->atomspace = atomspace;
    server->cycle_count = 0;
    server->running = false;
    return server;
}

void ggml_opencog_cogserver_free(struct ggml_opencog_cogserver* server) {
    if (!server) return;
    
    server->running = false;
    server->agents.clear();
    delete server;
}

void ggml_opencog_cogserver_add_agent(struct ggml_opencog_cogserver* server, struct ggml_opencog_mind_agent* agent) {
    server->agents.push_back(agent);
}

void ggml_opencog_cogserver_run_cycle(struct ggml_opencog_cogserver* server) {
    server->cycle_count++;
    
    // Run agents based on their frequency
    for (auto* agent : server->agents) {
        if ((server->cycle_count - agent->last_run) >= agent->frequency) {
            agent->process(server->atomspace);
            agent->last_run = server->cycle_count;
        }
    }
}

void ggml_opencog_cogserver_start(struct ggml_opencog_cogserver* server) {
    server->running = true;
}

void ggml_opencog_cogserver_stop(struct ggml_opencog_cogserver* server) {
    server->running = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared constants for inference thresholds
// ─────────────────────────────────────────────────────────────────────────────
// Minimum confidence for a derived truth value to be accepted
static const float GGML_OPENCOG_MIN_DERIVE_CONFIDENCE = 1e-6f;
// Minimum confidence for a goal to be considered proved
static const float GGML_OPENCOG_MIN_PROVED_CONFIDENCE = 0.1f;

// ─────────────────────────────────────────────────────────────────────────────
// Extended PLN rules
// ─────────────────────────────────────────────────────────────────────────────

// Conjunction (AND): P(A AND B) = s_A * s_B, conf = min(c_A, c_B)
struct ggml_opencog_truth_value ggml_opencog_pln_conjunction(
        struct ggml_opencog_truth_value tv1,
        struct ggml_opencog_truth_value tv2) {
    struct ggml_opencog_truth_value result;
    result.strength   = tv1.strength * tv2.strength;
    result.confidence = fminf(tv1.confidence, tv2.confidence);
    return result;
}

// Disjunction (OR): P(A OR B) = 1 - (1-s_A)*(1-s_B), conf = min(c_A, c_B)
struct ggml_opencog_truth_value ggml_opencog_pln_disjunction(
        struct ggml_opencog_truth_value tv1,
        struct ggml_opencog_truth_value tv2) {
    struct ggml_opencog_truth_value result;
    result.strength   = 1.0f - (1.0f - tv1.strength) * (1.0f - tv2.strength);
    result.confidence = fminf(tv1.confidence, tv2.confidence);
    return result;
}

// Negation (NOT): P(NOT A) = 1 - s_A, conf unchanged
struct ggml_opencog_truth_value ggml_opencog_pln_negation(
        struct ggml_opencog_truth_value tv) {
    struct ggml_opencog_truth_value result;
    result.strength   = 1.0f - tv.strength;
    result.confidence = tv.confidence;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward Chaining
// Apply PLN deduction transitively over InheritanceLinks:
//   A→B and B→C ⟹ A→C  (if not already present or new TV is stronger)
// Returns IDs of newly added atoms.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<uint64_t> ggml_opencog_forward_chain(
        struct ggml_opencog_atomspace* atomspace,
        int max_iterations) {
    std::vector<uint64_t> derived_atoms;

    for (int iter = 0; iter < max_iterations; ++iter) {
        bool new_derivation = false;

        // Collect all inheritance links (snapshot to avoid iterator invalidation)
        std::vector<uint64_t> inheritance_links =
            ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_INHERITANCE_LINK);

        // For each pair (A→B) and (B→C), derive (A→C)
        for (size_t i = 0; i < inheritance_links.size(); ++i) {
            auto* ab_link = ggml_opencog_get_atom(atomspace, inheritance_links[i]);
            if (!ab_link || ab_link->outgoing.size() < 2) continue;

            uint64_t a_id = ab_link->outgoing[0];
            uint64_t b_id = ab_link->outgoing[1];

            for (size_t j = 0; j < inheritance_links.size(); ++j) {
                if (i == j) continue;
                auto* bc_link = ggml_opencog_get_atom(atomspace, inheritance_links[j]);
                if (!bc_link || bc_link->outgoing.size() < 2) continue;

                // Check B→C: first element must be the same B
                if (bc_link->outgoing[0] != b_id) continue;

                uint64_t c_id = bc_link->outgoing[1];
                if (c_id == a_id) continue;  // avoid trivial A→A

                // Derive A→C via PLN deduction
                struct ggml_opencog_truth_value tv_ac =
                    ggml_opencog_pln_deduction(ab_link->tv, bc_link->tv);

                // Only add if TV is meaningful
                if (tv_ac.confidence < GGML_OPENCOG_MIN_DERIVE_CONFIDENCE) continue;

                // Build a unique name for this derived link
                auto* atom_a = ggml_opencog_get_atom(atomspace, a_id);
                auto* atom_c = ggml_opencog_get_atom(atomspace, c_id);
                if (!atom_a || !atom_c) continue;

                std::string derived_name =
                    std::string(atom_a->name) + "->" + std::string(atom_c->name)
                    + "(derived)";

                // Check for existing link between A and C
                bool already_present = false;
                std::vector<uint64_t> existing =
                    ggml_opencog_get_atoms_by_type(atomspace,
                                                   GGML_OPENCOG_INHERITANCE_LINK);
                for (uint64_t eid : existing) {
                    auto* elink = ggml_opencog_get_atom(atomspace, eid);
                    if (elink && elink->outgoing.size() >= 2 &&
                        elink->outgoing[0] == a_id &&
                        elink->outgoing[1] == c_id) {
                        // Update TV if new derivation is more confident
                        if (tv_ac.confidence > elink->tv.confidence) {
                            elink->tv = tv_ac;
                        }
                        already_present = true;
                        break;
                    }
                }

                if (!already_present) {
                    uint64_t new_id = ggml_opencog_add_atom(
                        atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                        derived_name.c_str(), tv_ac, {a_id, c_id});
                    derived_atoms.push_back(new_id);
                    new_derivation = true;
                }
            }
        }

        if (!new_derivation) break;
    }

    return derived_atoms;
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward Chaining
// Attempt to prove that goal_id is derivable from the atomspace using PLN.
// Strategy: goal is an atom that we want to show exists (with high TV).
//   1. If goal already has sufficient confidence, return true immediately.
//   2. Otherwise, look for an intermediate B such that A→B and B→goal_id.
//   3. Recursively try to prove A→B, then derive A→goal_id.
// Returns true if goal is proved. Fills derivation_path with IDs used.
// ─────────────────────────────────────────────────────────────────────────────
bool ggml_opencog_backward_chain(
        struct ggml_opencog_atomspace* atomspace,
        uint64_t goal_id,
        int max_depth,
        std::vector<uint64_t>& derivation_path) {
    if (max_depth <= 0) return false;

    auto* goal = ggml_opencog_get_atom(atomspace, goal_id);
    if (!goal) return false;

    // Base case: goal already exists with sufficient confidence
    if (goal->tv.confidence >= GGML_OPENCOG_MIN_PROVED_CONFIDENCE) {
        derivation_path.push_back(goal_id);
        return true;
    }

    // For an InheritanceLink goal (A→C), look for intermediate B
    if (goal->type == GGML_OPENCOG_INHERITANCE_LINK &&
        goal->outgoing.size() >= 2) {
        uint64_t a_id = goal->outgoing[0];
        uint64_t c_id = goal->outgoing[1];

        // Search for B: look for all atoms B such that A→B exists
        std::vector<uint64_t> links =
            ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_INHERITANCE_LINK);

        for (uint64_t link_id : links) {
            auto* link = ggml_opencog_get_atom(atomspace, link_id);
            if (!link || link->outgoing.size() < 2) continue;
            if (link->outgoing[0] != a_id) continue;  // must start from A

            uint64_t b_id = link->outgoing[1];
            if (b_id == c_id || b_id == a_id) continue;

            // Look for B→C
            for (uint64_t link2_id : links) {
                auto* link2 = ggml_opencog_get_atom(atomspace, link2_id);
                if (!link2 || link2->outgoing.size() < 2) continue;
                if (link2->outgoing[0] != b_id) continue;
                if (link2->outgoing[1] != c_id) continue;

                // Found A→B and B→C: derive A→C via deduction
                struct ggml_opencog_truth_value tv_ac =
                    ggml_opencog_pln_deduction(link->tv, link2->tv);

                if (tv_ac.confidence >= GGML_OPENCOG_MIN_PROVED_CONFIDENCE) {
                    // Update goal TV
                    goal->tv = tv_ac;
                    derivation_path.push_back(link_id);
                    derivation_path.push_back(link2_id);
                    derivation_path.push_back(goal_id);
                    return true;
                }
            }
        }

        // Recursive: try to prove sub-goals
        for (uint64_t link_id : links) {
            auto* link = ggml_opencog_get_atom(atomspace, link_id);
            if (!link || link->outgoing.size() < 2) continue;
            if (link->outgoing[0] != a_id) continue;

            uint64_t b_id = link->outgoing[1];
            if (b_id == c_id || b_id == a_id) continue;

            // Build B→C sub-goal
            auto* atom_b = ggml_opencog_get_atom(atomspace, b_id);
            auto* atom_c = ggml_opencog_get_atom(atomspace, c_id);
            if (!atom_b || !atom_c) continue;

            std::string sub_goal_name =
                std::string(atom_b->name) + "->" + std::string(atom_c->name) +
                "(subgoal)";
            struct ggml_opencog_truth_value zero_tv = {0.0f, 0.0f};
            uint64_t sub_goal_id = ggml_opencog_add_atom(
                atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                sub_goal_name.c_str(), zero_tv, {b_id, c_id});

            std::vector<uint64_t> sub_path;
            if (ggml_opencog_backward_chain(atomspace, sub_goal_id,
                                            max_depth - 1, sub_path)) {
                auto* derived_bc = ggml_opencog_get_atom(atomspace, sub_goal_id);
                if (derived_bc && derived_bc->tv.confidence >= GGML_OPENCOG_MIN_PROVED_CONFIDENCE) {
                    struct ggml_opencog_truth_value tv_ac =
                        ggml_opencog_pln_deduction(link->tv, derived_bc->tv);
                    goal->tv = tv_ac;
                    derivation_path.insert(derivation_path.end(),
                                           sub_path.begin(), sub_path.end());
                    derivation_path.push_back(goal_id);
                    return tv_ac.confidence >= GGML_OPENCOG_MIN_PROVED_CONFIDENCE;
                }
            }
            // Clean up unsuccessful sub-goal
            ggml_opencog_remove_atom(atomspace, sub_goal_id);
        }
    }

    // For concept nodes: check if they appear as conclusions of any rule
    if (goal->type == GGML_OPENCOG_CONCEPT_NODE ||
        goal->type == GGML_OPENCOG_PREDICATE_NODE) {
        // Look for evaluation links that assert a property of goal
        std::vector<uint64_t> eval_links =
            ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_EVALUATION_LINK);
        for (uint64_t eid : eval_links) {
            auto* elink = ggml_opencog_get_atom(atomspace, eid);
            if (!elink) continue;
            for (uint64_t out_id : elink->outgoing) {
                if (out_id == goal_id && elink->tv.confidence >= GGML_OPENCOG_MIN_PROVED_CONFIDENCE) {
                    derivation_path.push_back(eid);
                    derivation_path.push_back(goal_id);
                    goal->tv = elink->tv;
                    return true;
                }
            }
        }
    }

    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Variable binding pattern matching
// ─────────────────────────────────────────────────────────────────────────────

// Internal recursive helper
static bool match_recursive(
        struct ggml_opencog_atomspace* atomspace,
        uint64_t pattern_id,
        uint64_t candidate_id,
        struct ggml_opencog_binding* binding) {
    if (pattern_id == candidate_id) return true;  // trivial match

    auto* pattern   = ggml_opencog_get_atom(atomspace, pattern_id);
    auto* candidate = ggml_opencog_get_atom(atomspace, candidate_id);

    if (!pattern || !candidate) return false;

    // Variable node matches any atom; record binding
    if (pattern->type == GGML_OPENCOG_VARIABLE_NODE) {
        return binding->bind(pattern_id, candidate_id);
    }

    // Type must match for non-variable patterns
    if (pattern->type != candidate->type) return false;

    // For nodes, name must also match (non-variable)
    if (pattern->outgoing.empty() && candidate->outgoing.empty()) {
        return std::string(pattern->name) == std::string(candidate->name);
    }

    // For links, recursively match outgoing sets
    if (pattern->outgoing.size() != candidate->outgoing.size()) return false;

    for (size_t i = 0; i < pattern->outgoing.size(); ++i) {
        if (!match_recursive(atomspace, pattern->outgoing[i],
                             candidate->outgoing[i], binding)) {
            return false;
        }
    }
    return true;
}

bool ggml_opencog_match_with_binding(
        struct ggml_opencog_atomspace* atomspace,
        uint64_t pattern_id,
        uint64_t candidate_id,
        struct ggml_opencog_binding* binding) {
    return match_recursive(atomspace, pattern_id, candidate_id, binding);
}

std::vector<std::pair<uint64_t, struct ggml_opencog_binding>>
ggml_opencog_find_matching(
        struct ggml_opencog_atomspace* atomspace,
        uint64_t pattern_id) {
    std::vector<std::pair<uint64_t, struct ggml_opencog_binding>> results;

    auto* pattern = ggml_opencog_get_atom(atomspace, pattern_id);
    if (!pattern) return results;

    for (const auto& kv : atomspace->atoms) {
        struct ggml_opencog_binding binding;
        if (match_recursive(atomspace, pattern_id, kv.first, &binding)) {
            results.emplace_back(kv.first, binding);
        }
    }
    return results;
}

// Get total number of atoms in the atomspace
size_t ggml_opencog_atom_count(struct ggml_opencog_atomspace* atomspace) {
    return atomspace->atoms.size();
}