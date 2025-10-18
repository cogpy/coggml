#include "ggml-opencog.h"
#include <iostream>
#include <cassert>
#include <cmath>

bool test_atomspace_creation() {
    std::cout << "Testing AtomSpace creation... ";
    
    auto* atomspace = ggml_opencog_atomspace_new(32);
    if (!atomspace) {
        std::cout << "FAILED\n";
        return false;
    }
    
    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_atom_creation() {
    std::cout << "Testing atom creation... ";
    
    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv = {0.8f, 0.6f};
    
    uint64_t id1 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "TestConcept", tv, {});
    uint64_t id2 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_PREDICATE_NODE, "TestPredicate", tv, {});
    
    if (id1 == 0 || id2 == 0 || id1 == id2) {
        std::cout << "FAILED\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    auto* atom1 = ggml_opencog_get_atom(atomspace, id1);
    auto* atom2 = ggml_opencog_get_atom(atomspace, id2);
    
    if (!atom1 || !atom2 || 
        atom1->type != GGML_OPENCOG_CONCEPT_NODE ||
        atom2->type != GGML_OPENCOG_PREDICATE_NODE) {
        std::cout << "FAILED\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_atom_queries() {
    std::cout << "Testing atom queries... ";
    
    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv = {0.8f, 0.6f};
    
    uint64_t id1 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv, {});
    uint64_t id2 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Cat", tv, {});
    uint64_t id3 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_PREDICATE_NODE, "Likes", tv, {});
    
    // Test query by name
    auto dogs = ggml_opencog_get_atoms_by_name(atomspace, "Dog");
    if (dogs.size() != 1 || dogs[0] != id1) {
        std::cout << "FAILED (name query)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test query by type
    auto concepts = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_CONCEPT_NODE);
    if (concepts.size() != 2) {
        std::cout << "FAILED (type query)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    auto predicates = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_PREDICATE_NODE);
    if (predicates.size() != 1 || predicates[0] != id3) {
        std::cout << "FAILED (predicate query)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_links() {
    std::cout << "Testing link creation... ";
    
    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv = {0.8f, 0.6f};
    
    uint64_t dog_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv, {});
    uint64_t animal_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Animal", tv, {});
    
    uint64_t link_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                           "Dog->Animal", tv, {dog_id, animal_id});
    
    auto* link = ggml_opencog_get_atom(atomspace, link_id);
    if (!link || link->outgoing.size() != 2 || 
        link->outgoing[0] != dog_id || link->outgoing[1] != animal_id) {
        std::cout << "FAILED (outgoing links)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test incoming links
    auto* dog = ggml_opencog_get_atom(atomspace, dog_id);
    auto* animal = ggml_opencog_get_atom(atomspace, animal_id);
    
    if (!dog || !animal || 
        dog->incoming.size() != 1 || dog->incoming[0] != link_id ||
        animal->incoming.size() != 1 || animal->incoming[0] != link_id) {
        std::cout << "FAILED (incoming links)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_reasoning() {
    std::cout << "Testing PLN reasoning... ";
    
    struct ggml_opencog_truth_value tv1 = {0.9f, 0.8f};
    struct ggml_opencog_truth_value tv2 = {0.7f, 0.6f};
    
    // Test deduction
    auto deduced = ggml_opencog_pln_deduction(tv1, tv2);
    float expected_strength = 0.9f * 0.7f;
    float expected_confidence = fminf(0.8f, 0.6f) * expected_strength;
    
    if (fabsf(deduced.strength - expected_strength) > 1e-6f ||
        fabsf(deduced.confidence - expected_confidence) > 1e-6f) {
        std::cout << "FAILED (deduction)\n";
        return false;
    }
    
    // Test induction
    auto induced = ggml_opencog_pln_induction(tv1, tv2);
    float expected_ind_strength = (0.9f + 0.7f) / 2.0f;
    float expected_ind_confidence = fminf(0.8f, 0.6f) * 0.5f;
    
    if (fabsf(induced.strength - expected_ind_strength) > 1e-6f ||
        fabsf(induced.confidence - expected_ind_confidence) > 1e-6f) {
        std::cout << "FAILED (induction)\n";
        return false;
    }
    
    std::cout << "PASSED\n";
    return true;
}

bool test_cogserver() {
    std::cout << "Testing CogServer... ";
    
    auto* atomspace = ggml_opencog_atomspace_new(32);
    auto* cogserver = ggml_opencog_cogserver_new(atomspace);
    
    if (!cogserver) {
        std::cout << "FAILED (creation)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test agent addition
    int agent_calls = 0;
    auto test_agent = [](struct ggml_opencog_atomspace*) {
        // Agent function placeholder 
    };
    
    struct ggml_opencog_mind_agent agent;
    strcpy(agent.name, "TestAgent");
    agent.process = [](struct ggml_opencog_atomspace*) {}; // Simple no-op agent
    agent.frequency = 1;
    agent.last_run = 0;
    
    ggml_opencog_cogserver_add_agent(cogserver, &agent);
    
    // Test cycle execution
    ggml_opencog_cogserver_start(cogserver);
    ggml_opencog_cogserver_run_cycle(cogserver);
    ggml_opencog_cogserver_stop(cogserver);
    
    ggml_opencog_cogserver_free(cogserver);
    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

int main() {
    std::cout << "=== OpenCog GGML Tests ===\n\n";
    
    int passed = 0;
    int total = 6;
    
    if (test_atomspace_creation()) passed++;
    if (test_atom_creation()) passed++;
    if (test_atom_queries()) passed++;
    if (test_links()) passed++;
    if (test_reasoning()) passed++;
    if (test_cogserver()) passed++;
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << passed << "/" << total << " tests\n";
    
    if (passed == total) {
        std::cout << "All tests PASSED!\n";
        return 0;
    } else {
        std::cout << "Some tests FAILED!\n";
        return 1;
    }
}