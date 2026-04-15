// cog/prime/prime.hpp — AGI Architecture
// Cognitive cycle, PLN reasoning, pattern matching, memory systems
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_PRIME_HPP
#define COG_PRIME_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cassert>
#include <cmath>

namespace cog { namespace prime {

// ─────────────────────────────────────────────────────────────────────────────
// PLN Truth Values — Probabilistic Logic Networks
// ─────────────────────────────────────────────────────────────────────────────
struct TruthValue {
    float strength;    // probability estimate [0,1]
    float confidence;  // certainty weight [0,1]

    TruthValue() : strength(0.5f), confidence(0.0f) {}
    TruthValue(float s, float c) : strength(s), confidence(c) {}

    // Revision: combine two truth values
    static TruthValue revise(const TruthValue& t1, const TruthValue& t2) {
        float c1 = t1.confidence, c2 = t2.confidence;
        float total = c1 + c2;
        if (total < 1e-8f) return TruthValue(0.5f, 0.0f);
        float s = (t1.strength * c1 + t2.strength * c2) / total;
        float c = std::min(1.0f, total);
        return TruthValue(s, c);
    }

    // Conjunction (AND): min-based
    static TruthValue conjunction(const TruthValue& t1, const TruthValue& t2) {
        return TruthValue(t1.strength * t2.strength,
                          std::min(t1.confidence, t2.confidence));
    }

    // Disjunction (OR): max-based
    static TruthValue disjunction(const TruthValue& t1, const TruthValue& t2) {
        return TruthValue(1.0f - (1.0f - t1.strength) * (1.0f - t2.strength),
                          std::min(t1.confidence, t2.confidence));
    }

    // Negation
    static TruthValue negation(const TruthValue& t) {
        return TruthValue(1.0f - t.strength, t.confidence);
    }

    // Deduction: P(A→C) from P(A→B) and P(B→C)
    static TruthValue deduction(const TruthValue& ab, const TruthValue& bc,
                                float pb = 0.5f) {
        float s = ab.strength * bc.strength + (1.0f - ab.strength) * (bc.strength - pb) * 0.5f;
        float c = std::min(ab.confidence, bc.confidence) * 0.9f;
        s = std::max(0.0f, std::min(1.0f, s));
        return TruthValue(s, c);
    }

    // Induction: P(A→B) from P(A→C) and P(B→C)
    static TruthValue induction(const TruthValue& ac, const TruthValue& bc,
                                float pa = 0.5f) {
        float s = (pa > 1e-8f)
                  ? (ac.strength * bc.strength / pa)
                  : 0.0f;
        s = std::max(0.0f, std::min(1.0f, s));
        float c = std::min(ac.confidence, bc.confidence) * 0.8f;
        return TruthValue(s, c);
    }

    bool operator==(const TruthValue& o) const {
        return std::fabs(strength - o.strength) < 1e-6f &&
               std::fabs(confidence - o.confidence) < 1e-6f;
    }

    std::string to_string() const {
        std::ostringstream ss;
        ss << "TV(" << strength << ", " << confidence << ")";
        return ss.str();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Attention Values — ECAN (Economic Attention Networks)
// ─────────────────────────────────────────────────────────────────────────────
struct AttentionValue {
    float sti;   // short-term importance [-1, 1] (normalized)
    float lti;   // long-term importance  [0, 1]
    float vlti;  // very long-term importance [0, 1]

    AttentionValue() : sti(0.0f), lti(0.0f), vlti(0.0f) {}
    AttentionValue(float s, float l = 0.0f, float vl = 0.0f)
        : sti(s), lti(l), vlti(vl) {}

    // Stimulus: increase STI by amount, decay toward 0
    void stimulate(float amount, float decay = 0.95f) {
        sti = std::max(-1.0f, std::min(1.0f, sti * decay + amount));
        lti = std::max(0.0f, std::min(1.0f, lti + amount * 0.01f));
    }

    void decay(float rate = 0.99f) {
        sti *= rate;
        lti *= (rate + (1.0f - rate) * 0.1f);  // LTI decays much slower
    }

    bool is_attentional() const { return sti > 0.0f; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Atom & AtomSpace
// ─────────────────────────────────────────────────────────────────────────────
using Handle = cog::Handle;

struct Atom {
    Handle         handle;
    cog::AtomType  type;
    std::string    name;
    std::vector<Handle> outgoing;  // for links
    TruthValue     tv;
    AttentionValue av;

    Atom() : handle(cog::UNDEFINED_HANDLE), type(cog::AtomType::NOTYPE) {}
    Atom(Handle h, cog::AtomType t, const std::string& n)
        : handle(h), type(t), name(n) {}
};

class AtomSpace {
public:
    AtomSpace() : next_handle_(1) {}

    Handle add_node(cog::AtomType type, const std::string& name) {
        // Check for existing node
        auto key = std::make_pair(static_cast<uint16_t>(type), name);
        auto it = node_index_.find(name);
        if (it != node_index_.end()) {
            auto* a = get(it->second);
            if (a && a->type == type) return it->second;
        }
        Handle h = next_handle_++;
        atoms_[h] = Atom(h, type, name);
        node_index_[name] = h;
        return h;
    }

    Handle add_link(cog::AtomType type,
                    const std::vector<Handle>& outgoing,
                    const TruthValue& tv = TruthValue(1.0f, 0.9f)) {
        Handle h = next_handle_++;
        Atom a(h, type, "");
        a.outgoing = outgoing;
        a.tv = tv;
        atoms_[h] = a;
        return h;
    }

    Atom* get(Handle h) {
        auto it = atoms_.find(h);
        return (it != atoms_.end()) ? &it->second : nullptr;
    }

    const Atom* get(Handle h) const {
        auto it = atoms_.find(h);
        return (it != atoms_.end()) ? &it->second : nullptr;
    }

    Handle lookup_node(const std::string& name) const {
        auto it = node_index_.find(name);
        return (it != node_index_.end()) ? it->second : cog::UNDEFINED_HANDLE;
    }

    bool remove(Handle h) {
        auto it = atoms_.find(h);
        if (it == atoms_.end()) return false;
        auto& a = it->second;
        if (is_node(a.type)) node_index_.erase(a.name);
        atoms_.erase(it);
        return true;
    }

    size_t size() const { return atoms_.size(); }

    friend class PatternMatcher;

    // Get all atoms of a given type
    std::vector<Handle> get_by_type(cog::AtomType type) const {
        std::vector<Handle> result;
        for (auto& kv : atoms_) {
            if (kv.second.type == type) result.push_back(kv.first);
        }
        return result;
    }

    // Stimulate attention
    void stimulate(Handle h, float amount) {
        auto* a = get(h);
        if (a) a->av.stimulate(amount);
    }

    // Decay all attention values
    void decay_attention(float rate = 0.99f) {
        for (auto& kv : atoms_) kv.second.av.decay(rate);
    }

    // Get atoms sorted by STI (most important first)
    std::vector<Handle> top_by_sti(size_t n) const {
        std::vector<std::pair<float, Handle>> ranked;
        for (auto& kv : atoms_) {
            ranked.push_back({kv.second.av.sti, kv.first});
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const std::pair<float,Handle>& a, const std::pair<float,Handle>& b){
                      return a.first > b.first;
                  });
        std::vector<Handle> result;
        size_t lim = std::min(n, ranked.size());
        for (size_t i = 0; i < lim; ++i) result.push_back(ranked[i].second);
        return result;
    }

private:
    Handle next_handle_;
    std::unordered_map<Handle, Atom> atoms_;
    std::unordered_map<std::string, Handle> node_index_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Pattern Matcher — simple unification over AtomSpace
// ─────────────────────────────────────────────────────────────────────────────
struct Binding {
    std::unordered_map<Handle, Handle> var_to_atom;

    bool bind(Handle var, Handle atom) {
        auto it = var_to_atom.find(var);
        if (it != var_to_atom.end()) return it->second == atom;
        var_to_atom[var] = atom;
        return true;
    }

    Handle get(Handle var) const {
        auto it = var_to_atom.find(var);
        return (it != var_to_atom.end()) ? it->second : cog::UNDEFINED_HANDLE;
    }
};

class PatternMatcher {
public:
    explicit PatternMatcher(AtomSpace& as) : as_(as) {}

    // Match pattern atom against candidate; fill binding
    bool match(Handle pattern, Handle candidate, Binding& binding) const {
        const Atom* pat = as_.get(pattern);
        const Atom* cand = as_.get(candidate);
        if (!pat || !cand) return false;

        // Variable: bind to anything
        if (pat->type == cog::AtomType::VARIABLE_NODE) {
            return binding.bind(pattern, candidate);
        }

        // Type must match
        if (pat->type != cand->type) return false;
        if (!cog::is_link(pat->type)) {
            // Node: name must match (unless variable, handled above)
            return pat->name == cand->name;
        }

        // Link: match outgoing recursively
        if (pat->outgoing.size() != cand->outgoing.size()) return false;
        for (size_t i = 0; i < pat->outgoing.size(); ++i) {
            if (!match(pat->outgoing[i], cand->outgoing[i], binding)) {
                return false;
            }
        }
        return true;
    }

    // Find all candidates in AtomSpace matching pattern
    std::vector<Binding> find(Handle pattern) const {
        std::vector<Binding> results;
        const Atom* pat = as_.get(pattern);
        if (!pat) return results;

        for (auto& kv : as_.atoms_) {
            Binding b;
            if (match(pattern, kv.first, b)) {
                results.push_back(b);
            }
        }
        return results;
    }

private:
    AtomSpace& as_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Memory Systems
// ─────────────────────────────────────────────────────────────────────────────

// Declarative memory: fact store
class DeclarativeMemory {
public:
    DeclarativeMemory() : as_() {}

    Handle store_fact(const std::string& subject,
                      const std::string& predicate,
                      const std::string& object,
                      const TruthValue& tv = TruthValue(1.0f, 0.9f)) {
        Handle s = as_.add_node(cog::AtomType::CONCEPT_NODE, subject);
        Handle p = as_.add_node(cog::AtomType::PREDICATE_NODE, predicate);
        Handle o = as_.add_node(cog::AtomType::CONCEPT_NODE, object);
        Handle eval_h = as_.add_link(cog::AtomType::EVALUATION_LINK, {p, s, o}, tv);
        return eval_h;
    }

    std::vector<Handle> query(const std::string& subject) const {
        Handle h = as_.lookup_node(subject);
        if (h == cog::UNDEFINED_HANDLE) return {};
        return as_.get_by_type(cog::AtomType::EVALUATION_LINK);
    }

    size_t size() const { return as_.size(); }
    AtomSpace& atomspace() { return as_; }

private:
    AtomSpace as_;
};

// Episodic memory: sequence of events
struct Episode {
    uint64_t    timestamp;
    std::string context;
    std::string event;
    float       salience;

    Episode() : timestamp(0), salience(0.0f) {}
    Episode(uint64_t ts, const std::string& ctx,
            const std::string& ev, float sal = 1.0f)
        : timestamp(ts), context(ctx), event(ev), salience(sal) {}
};

class EpisodicMemory {
public:
    EpisodicMemory() : tick_(0) {}

    void record(const std::string& context,
                const std::string& event,
                float salience = 1.0f) {
        episodes_.push_back(Episode(tick_++, context, event, salience));
    }

    // Recall most recent N episodes matching context
    std::vector<Episode> recall(const std::string& context, size_t n = 5) const {
        std::vector<Episode> result;
        for (auto it = episodes_.rbegin(); it != episodes_.rend(); ++it) {
            if (it->context == context || context.empty()) {
                result.push_back(*it);
                if (result.size() >= n) break;
            }
        }
        return result;
    }

    // Recall by salience
    std::vector<Episode> recall_salient(size_t n = 5) const {
        std::vector<Episode> sorted = episodes_;
        std::sort(sorted.begin(), sorted.end(),
                  [](const Episode& a, const Episode& b){ return a.salience > b.salience; });
        if (sorted.size() > n) sorted.resize(n);
        return sorted;
    }

    size_t size() const { return episodes_.size(); }

private:
    std::vector<Episode> episodes_;
    uint64_t tick_;
};

// Procedural memory: skill/procedure store
struct Procedure {
    std::string name;
    std::string description;
    float       competence;   // [0,1]
    uint32_t    invocations;

    Procedure() : competence(0.0f), invocations(0) {}
    Procedure(const std::string& n, const std::string& d, float c = 0.5f)
        : name(n), description(d), competence(c), invocations(0) {}
};

class ProceduralMemory {
public:
    void store(const Procedure& proc) {
        procs_[proc.name] = proc;
    }

    Procedure* get(const std::string& name) {
        auto it = procs_.find(name);
        return (it != procs_.end()) ? &it->second : nullptr;
    }

    void practice(const std::string& name, float improvement = 0.01f) {
        auto* p = get(name);
        if (p) {
            p->invocations++;
            p->competence = std::min(1.0f, p->competence + improvement);
        }
    }

    size_t size() const { return procs_.size(); }

    std::vector<std::string> list() const {
        std::vector<std::string> names;
        for (auto& kv : procs_) names.push_back(kv.first);
        return names;
    }

private:
    std::unordered_map<std::string, Procedure> procs_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Cognitive Cycle — Sense-Think-Act loop
// ─────────────────────────────────────────────────────────────────────────────
enum class CyclePhase : uint8_t {
    PERCEIVE  = 0,  // Sensor input processing
    ATTEND    = 1,  // Attention allocation (ECAN)
    REMEMBER  = 2,  // Memory consolidation
    REASON    = 3,  // PLN inference
    DECIDE    = 4,  // Goal-driven action selection
    ACT       = 5,  // Motor/output execution
    REFLECT   = 6,  // Meta-cognitive self-monitoring
    PHASE_COUNT = 7
};

inline const char* phase_name(CyclePhase p) {
    static const char* names[] = {
        "Perceive", "Attend", "Remember", "Reason",
        "Decide", "Act", "Reflect"
    };
    return names[static_cast<size_t>(p)];
}

struct CognitiveState {
    uint64_t      cycle_count;
    CyclePhase    phase;
    float         arousal;    // [0,1] overall activation level
    float         valence;    // [-1,1] emotional valence
    float         coherence;  // [0,1] internal consistency
    std::string   current_goal;
    std::string   last_action;

    CognitiveState()
        : cycle_count(0), phase(CyclePhase::PERCEIVE),
          arousal(0.5f), valence(0.0f), coherence(0.9f) {}

    void advance_phase() {
        phase = static_cast<CyclePhase>(
            (static_cast<uint8_t>(phase) + 1) %
            static_cast<uint8_t>(CyclePhase::PHASE_COUNT));
        if (phase == CyclePhase::PERCEIVE) ++cycle_count;
    }

    std::string to_string() const {
        std::ostringstream ss;
        ss << "CogState[cycle=" << cycle_count
           << " phase=" << phase_name(phase)
           << " arousal=" << arousal
           << " valence=" << valence
           << " coherence=" << coherence << "]";
        return ss.str();
    }
};

using PhaseHandler = std::function<void(CognitiveState&,
                                        DeclarativeMemory&,
                                        EpisodicMemory&,
                                        ProceduralMemory&)>;

class CognitiveCycle {
public:
    CognitiveCycle()
        : decl_mem_(), epis_mem_(), proc_mem_() {
        // Register default no-op handlers
        for (int i = 0; i < static_cast<int>(CyclePhase::PHASE_COUNT); ++i) {
            handlers_[static_cast<CyclePhase>(i)] = [](CognitiveState&,
                DeclarativeMemory&, EpisodicMemory&, ProceduralMemory&){};
        }
    }

    void on_phase(CyclePhase phase, PhaseHandler handler) {
        handlers_[phase] = std::move(handler);
    }

    // Run one complete 7-phase cognitive cycle
    CognitiveState& run_cycle() {
        for (int i = 0; i < static_cast<int>(CyclePhase::PHASE_COUNT); ++i) {
            CyclePhase p = state_.phase;
            auto it = handlers_.find(p);
            if (it != handlers_.end()) {
                it->second(state_, decl_mem_, epis_mem_, proc_mem_);
            }
            state_.advance_phase();
        }
        return state_;
    }

    // Run n cycles
    void run(size_t n = 1) {
        for (size_t i = 0; i < n; ++i) run_cycle();
    }

    CognitiveState&    state()     { return state_;    }
    DeclarativeMemory& decl_mem()  { return decl_mem_; }
    EpisodicMemory&    epis_mem()  { return epis_mem_; }
    ProceduralMemory&  proc_mem()  { return proc_mem_; }

private:
    CognitiveState    state_;
    DeclarativeMemory decl_mem_;
    EpisodicMemory    epis_mem_;
    ProceduralMemory  proc_mem_;
    std::unordered_map<CyclePhase, PhaseHandler> handlers_;
};


// ─────────────────────────────────────────────────────────────────────────────
// Ontogenesis — developmental autonomy levels
// ─────────────────────────────────────────────────────────────────────────────
enum class OntogeneticLevel : uint8_t {
    SCAFFOLD     = 0,
    REACTIVE     = 1,
    DELIBERATIVE = 2,
    ENABLED      = 3,
    EMBODIED     = 4,
    TRANSCENDENT = 5
};

inline const char* ontogenetic_name(OntogeneticLevel l) {
    static const char* names[] = {
        "Scaffold", "Reactive", "Deliberative",
        "Enabled", "Embodied", "Transcendent"
    };
    return names[static_cast<size_t>(l)];
}

struct OntogeneticState {
    OntogeneticLevel level;
    float fitness;      // [0,1] performance on current level tasks
    float wisdom;       // [0,1] accumulated cross-level insight
    float metacoherence;// [0,1] consistency of self-model
    uint32_t cycles_at_level;

    OntogeneticState()
        : level(OntogeneticLevel::SCAFFOLD),
          fitness(0.0f), wisdom(0.0f), metacoherence(0.5f),
          cycles_at_level(0) {}

    // Attempt level transition if fitness threshold met
    bool try_advance(float threshold = 0.8f) {
        if (fitness >= threshold &&
            level < OntogeneticLevel::TRANSCENDENT) {
            level = static_cast<OntogeneticLevel>(
                static_cast<uint8_t>(level) + 1);
            cycles_at_level = 0;
            wisdom = std::min(1.0f, wisdom + 0.1f);
            return true;
        }
        return false;
    }

    void update(float perf_signal) {
        fitness = 0.95f * fitness + 0.05f * perf_signal;
        metacoherence = 0.99f * metacoherence + 0.01f * fitness;
        ++cycles_at_level;
    }

    std::string to_string() const {
        std::ostringstream ss;
        ss << "OntState[level=" << ontogenetic_name(level)
           << " fitness=" << fitness
           << " wisdom=" << wisdom << "]";
        return ss.str();
    }
};

}} // namespace cog::prime

#endif // COG_PRIME_HPP
