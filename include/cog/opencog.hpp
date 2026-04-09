// cog/opencog.hpp — Bridge between cog::AtomSpace and ggml_opencog_atomspace
//
// This header is OPTIONAL and is NOT included by cog/cog.hpp because it
// requires the GGML library to be compiled (ggml-opencog.cpp).
// Include it manually when GGML is available:
//
//   #include "cog/cog.hpp"
//   #include "cog/opencog.hpp"
//
// It exposes a thin C++11 RAII wrapper around ggml_opencog_atomspace that
// speaks the same vocabulary as cog::AtomSpace, enabling seamless migration
// of cog:: code to GGML-accelerated tensor operations.
//
// Header-only adapter layer — the underlying implementation lives in
// src/ggml-opencog.cpp (linked as part of the ggml target).
//
// SPDX-License-Identifier: MIT
#ifndef COG_OPENCOG_HPP
#define COG_OPENCOG_HPP

#include "cog/core/core.hpp"
#include "ggml-opencog.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

namespace cog { namespace opencog {

// ─────────────────────────────────────────────────────────────────────────────
// TruthValue conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

inline ggml_opencog_truth_value to_ggml_tv(float strength, float confidence) {
    ggml_opencog_truth_value tv;
    tv.strength   = strength;
    tv.confidence = confidence;
    return tv;
}

inline ggml_opencog_truth_value to_ggml_tv(const cog::TruthValue& tv) {
    return to_ggml_tv(tv.strength, tv.confidence);
}

// ─────────────────────────────────────────────────────────────────────────────
// AtomType mapping: cog::AtomType  <->  ggml_opencog_atom_type
// ─────────────────────────────────────────────────────────────────────────────

inline ggml_opencog_atom_type to_ggml_type(cog::AtomType t) {
    switch (t) {
        case cog::AtomType::CONCEPT_NODE:
            return GGML_OPENCOG_CONCEPT_NODE;
        case cog::AtomType::PREDICATE_NODE:
            return GGML_OPENCOG_PREDICATE_NODE;
        case cog::AtomType::VARIABLE_NODE:
            return GGML_OPENCOG_VARIABLE_NODE;
        case cog::AtomType::NUMBER_NODE:
            return GGML_OPENCOG_NUMBER_NODE;
        case cog::AtomType::INHERITANCE_LINK:
            return GGML_OPENCOG_INHERITANCE_LINK;
        case cog::AtomType::SIMILARITY_LINK:
            return GGML_OPENCOG_SIMILARITY_LINK;
        case cog::AtomType::EVALUATION_LINK:
            return GGML_OPENCOG_EVALUATION_LINK;
        case cog::AtomType::AND_LINK:
            return GGML_OPENCOG_AND_LINK;
        case cog::AtomType::OR_LINK:
            return GGML_OPENCOG_OR_LINK;
        case cog::AtomType::NOT_LINK:
            return GGML_OPENCOG_NOT_LINK;
        case cog::AtomType::IMPLICATION_LINK:
            return GGML_OPENCOG_IMPLICATION_LINK;
        case cog::AtomType::LIST_LINK:
            return GGML_OPENCOG_LIST_LINK;
        default:
            return GGML_OPENCOG_CONCEPT_NODE;
    }
}

inline cog::AtomType from_ggml_type(ggml_opencog_atom_type t) {
    switch (t) {
        case GGML_OPENCOG_CONCEPT_NODE:     return cog::AtomType::CONCEPT_NODE;
        case GGML_OPENCOG_PREDICATE_NODE:   return cog::AtomType::PREDICATE_NODE;
        case GGML_OPENCOG_VARIABLE_NODE:    return cog::AtomType::VARIABLE_NODE;
        case GGML_OPENCOG_NUMBER_NODE:      return cog::AtomType::NUMBER_NODE;
        case GGML_OPENCOG_INHERITANCE_LINK: return cog::AtomType::INHERITANCE_LINK;
        case GGML_OPENCOG_SIMILARITY_LINK:  return cog::AtomType::SIMILARITY_LINK;
        case GGML_OPENCOG_EVALUATION_LINK:  return cog::AtomType::EVALUATION_LINK;
        case GGML_OPENCOG_AND_LINK:         return cog::AtomType::AND_LINK;
        case GGML_OPENCOG_OR_LINK:          return cog::AtomType::OR_LINK;
        case GGML_OPENCOG_NOT_LINK:         return cog::AtomType::NOT_LINK;
        case GGML_OPENCOG_IMPLICATION_LINK: return cog::AtomType::IMPLICATION_LINK;
        case GGML_OPENCOG_LIST_LINK:        return cog::AtomType::LIST_LINK;
        case GGML_OPENCOG_SEQUENTIAL_LINK:  return cog::AtomType::SEQUENTIAL_AND_LINK;
        default:                            return cog::AtomType::CONCEPT_NODE;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GGMLAtomSpace
// RAII wrapper that owns a ggml_opencog_atomspace* and exposes a cog::-style
// interface on top of it.  The underlying GGML atomspace adds tensor-accelerated
// operations (embeddings, PLN reasoning, Hebbian learning, temporal reasoning)
// that are not available in the pure header-only cog::AtomSpace.
// ─────────────────────────────────────────────────────────────────────────────
class GGMLAtomSpace {
public:
    // embedding_dim: dimensionality of atom embedding vectors (e.g. 32, 64, 128)
    explicit GGMLAtomSpace(int embedding_dim = 64)
        : as_(ggml_opencog_atomspace_new(embedding_dim))
    {
        if (!as_) throw std::runtime_error("ggml_opencog_atomspace_new failed");
    }

    ~GGMLAtomSpace() {
        if (as_) ggml_opencog_atomspace_free(as_);
    }

    // Non-copyable, movable
    GGMLAtomSpace(const GGMLAtomSpace&)            = delete;
    GGMLAtomSpace& operator=(const GGMLAtomSpace&) = delete;

    GGMLAtomSpace(GGMLAtomSpace&& o) noexcept : as_(o.as_) { o.as_ = nullptr; }
    GGMLAtomSpace& operator=(GGMLAtomSpace&& o) noexcept {
        if (this != &o) {
            if (as_) ggml_opencog_atomspace_free(as_);
            as_ = o.as_;
            o.as_ = nullptr;
        }
        return *this;
    }

    // ── Atom manipulation ────────────────────────────────────────────────────

    // Add a node (no outgoing links)
    uint64_t add_node(cog::AtomType type,
                      const std::string& name,
                      float strength = 1.0f,
                      float confidence = 0.9f) {
        auto tv = to_ggml_tv(strength, confidence);
        return ggml_opencog_add_atom(as_, to_ggml_type(type),
                                     name.c_str(), tv, {});
    }

    // Add a link with outgoing atom IDs
    uint64_t add_link(cog::AtomType type,
                      const std::string& name,
                      const std::vector<uint64_t>& outgoing,
                      float strength = 1.0f,
                      float confidence = 0.9f) {
        auto tv = to_ggml_tv(strength, confidence);
        return ggml_opencog_add_atom(as_, to_ggml_type(type),
                                     name.c_str(), tv, outgoing);
    }

    // Remove an atom by ID
    bool remove(uint64_t id) {
        return ggml_opencog_remove_atom(as_, id);
    }

    // Total number of atoms
    size_t atom_count() const {
        return ggml_opencog_atom_count(as_);
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    std::vector<uint64_t> get_by_name(const std::string& name) const {
        return ggml_opencog_get_atoms_by_name(as_, name.c_str());
    }

    std::vector<uint64_t> get_by_type(cog::AtomType type) const {
        return ggml_opencog_get_atoms_by_type(as_, to_ggml_type(type));
    }

    // Return the raw ggml_opencog_atom pointer (nullptr if not found)
    const ggml_opencog_atom* get_atom(uint64_t id) const {
        return ggml_opencog_get_atom(as_, id);
    }

    // ── Reasoning ───────────────────────────────────────────────────────────

    // Probabilistic Logic Network deduction
    ggml_opencog_truth_value pln_deduction(ggml_opencog_truth_value p1,
                                            ggml_opencog_truth_value p2) const {
        return ggml_opencog_pln_deduction(p1, p2);
    }

    // Forward chaining: derive new InheritanceLinks up to max_iterations
    std::vector<uint64_t> forward_chain(int max_iterations = 10) {
        return ggml_opencog_forward_chain(as_, max_iterations);
    }

    // Backward chaining: attempt to prove goal_id
    bool backward_chain(uint64_t goal_id, int max_depth,
                        std::vector<uint64_t>& path) {
        return ggml_opencog_backward_chain(as_, goal_id, max_depth, path);
    }

    // ── Embeddings & similarity ──────────────────────────────────────────────

    float similarity(uint64_t id1, uint64_t id2) const {
        return ggml_opencog_compute_similarity(as_, id1, id2);
    }

    void hebbian_update(uint64_t id1, uint64_t id2, float lr = 0.01f) {
        ggml_opencog_hebbian_update(as_, id1, id2, lr);
    }

    // ── Attention (ECAN) ─────────────────────────────────────────────────────

    void update_attention(uint64_t id, float sti_delta, float lti_delta = 0.0f) {
        ggml_opencog_update_attention(as_, id, sti_delta, lti_delta);
    }

    // ── Pattern matching ─────────────────────────────────────────────────────

    std::vector<std::pair<uint64_t, ggml_opencog_binding>>
    find_matching(uint64_t pattern_id) {
        return ggml_opencog_find_matching(as_, pattern_id);
    }

    // ── Temporal reasoning ───────────────────────────────────────────────────

    void set_time_interval(uint64_t id, int64_t start_ms, int64_t end_ms) {
        ggml_opencog_set_time_interval(as_, id, start_ms, end_ms);
    }

    std::vector<uint64_t> get_atoms_at_time(int64_t time_ms) const {
        return ggml_opencog_get_atoms_at_time(as_, time_ms);
    }

    // ── Raw access ───────────────────────────────────────────────────────────

    ggml_opencog_atomspace* raw() { return as_; }
    const ggml_opencog_atomspace* raw() const { return as_; }

    // ── Populate from cog::AtomSpace ─────────────────────────────────────────
    // Imports all atoms from a header-only cog::AtomSpace into this GGML
    // atomspace.  Returns a map from cog::Handle to the new GGML atom ID.
    std::unordered_map<cog::Handle, uint64_t>
    import_from(const cog::AtomSpace& src) {
        std::unordered_map<cog::Handle, uint64_t> id_map;
        // First pass: nodes only (no outgoing edges)
        src.foreach_atom([&](const cog::Atom& a) {
            if (!cog::is_link(a.type)) {
                auto tv = to_ggml_tv(a.tv.strength, a.tv.confidence);
                uint64_t gid = ggml_opencog_add_atom(as_, to_ggml_type(a.type),
                                                     a.name.c_str(), tv, {});
                id_map[a.handle] = gid;
            }
        });
        // Second pass: links with translated outgoing IDs
        src.foreach_atom([&](const cog::Atom& a) {
            if (!cog::is_link(a.type)) return;
            std::vector<uint64_t> out;
            out.reserve(a.outgoing.size());
            for (cog::Handle ch : a.outgoing) {
                auto it = id_map.find(ch);
                if (it != id_map.end()) out.push_back(it->second);
            }
            auto tv = to_ggml_tv(a.tv.strength, a.tv.confidence);
            uint64_t gid = ggml_opencog_add_atom(as_, to_ggml_type(a.type),
                                                  a.name.c_str(), tv, out);
            id_map[a.handle] = gid;
        });
        return id_map;
    }

private:
    ggml_opencog_atomspace* as_;
};

// ─────────────────────────────────────────────────────────────────────────────
// GGMLCogServer — convenience RAII wrapper around ggml_opencog_cogserver
// ─────────────────────────────────────────────────────────────────────────────
class GGMLCogServer {
public:
    explicit GGMLCogServer(GGMLAtomSpace& as)
        : srv_(ggml_opencog_cogserver_new(as.raw()))
    {
        if (!srv_) throw std::runtime_error("ggml_opencog_cogserver_new failed");
    }

    ~GGMLCogServer() {
        if (srv_) ggml_opencog_cogserver_free(srv_);
    }

    GGMLCogServer(const GGMLCogServer&)            = delete;
    GGMLCogServer& operator=(const GGMLCogServer&) = delete;

    void add_agent(ggml_opencog_mind_agent* agent) {
        ggml_opencog_cogserver_add_agent(srv_, agent);
    }

    void run_cycle() { ggml_opencog_cogserver_run_cycle(srv_); }
    void start()     { ggml_opencog_cogserver_start(srv_); }
    void stop()      { ggml_opencog_cogserver_stop(srv_); }

    ggml_opencog_cogserver* raw() { return srv_; }

private:
    ggml_opencog_cogserver* srv_;
};

}} // namespace cog::opencog

#endif // COG_OPENCOG_HPP
