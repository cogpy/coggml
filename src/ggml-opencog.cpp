#include "ggml-opencog.h"
#include "ggml-cpu.h"
#include <cstring>
#include <cstdio>
#include <cmath>
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
    
    // Initialize embedding based on type and name
    std::vector<float> embedding_data(atomspace->embedding_dim);
    
    // Get type embedding
    std::vector<float> type_embedding(atomspace->embedding_dim);
    ggml_backend_tensor_get(atomspace->type_embeddings, type_embedding.data(), 
                           type * atomspace->embedding_dim * sizeof(float), 
                           atomspace->embedding_dim * sizeof(float));
    
    // Simple hash-based initialization for name
    std::hash<std::string> hasher;
    size_t name_hash = hasher(name);
    std::mt19937 gen(name_hash);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (int i = 0; i < atomspace->embedding_dim; i++) {
        embedding_data[i] = type_embedding[i] + dist(gen);
    }
    
    // Store embedding data directly in the atom (we'll use the atom matrix for actual tensor ops)
    atom->embedding = nullptr; // We'll implement proper tensor management later
    
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

// Simple pattern matching based on embedding similarity
std::vector<uint64_t> ggml_opencog_pattern_match(struct ggml_opencog_atomspace* atomspace,
                                                  const struct ggml_tensor* pattern) {
    std::vector<uint64_t> matches;
    
    // Get pattern data
    std::vector<float> pattern_data(ggml_nelements(pattern));
    ggml_backend_tensor_get(pattern, pattern_data.data(), 0, ggml_nbytes(pattern));
    
    const float similarity_threshold = 0.8f;
    
    // Compare against all atoms
    for (const auto& [id, atom] : atomspace->atoms) {
        // For now, use a simple name-based similarity as embedding tensor management is disabled
        // In a full implementation, this would use actual tensor embeddings
        std::string atom_name = atom->name;
        float similarity = 0.5f; // Placeholder similarity
        
        if (similarity >= similarity_threshold) {
            matches.push_back(id);
        }
    }
    
    // Sort by similarity (would need to store similarities for this)
    return matches;
}

// PLN (Probabilistic Logic Networks) reasoning functions
struct ggml_opencog_truth_value ggml_opencog_pln_deduction(struct ggml_opencog_truth_value premise1,
                                                           struct ggml_opencog_truth_value premise2) {
    // Basic deduction: if A->B and B->C then A->C
    // Strength: s1 * s2
    // Confidence: min(c1, c2) * s1 * s2
    
    struct ggml_opencog_truth_value result;
    result.strength = premise1.strength * premise2.strength;
    result.confidence = fminf(premise1.confidence, premise2.confidence) * result.strength;
    
    return result;
}

struct ggml_opencog_truth_value ggml_opencog_pln_induction(struct ggml_opencog_truth_value premise1,
                                                          struct ggml_opencog_truth_value premise2) {
    // Basic induction: if A->B and A->C, infer B->C
    // This is a simplified version
    
    struct ggml_opencog_truth_value result;
    result.strength = (premise1.strength + premise2.strength) / 2.0f;
    result.confidence = fminf(premise1.confidence, premise2.confidence) * 0.5f; // Less confident than deduction
    
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