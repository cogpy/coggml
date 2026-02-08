#include "ggml-opencog.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>

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
    
    // Test abduction
    auto abduced = ggml_opencog_pln_abduction(tv1, tv2);
    if (abduced.strength < 0.0f || abduced.strength > 1.0f ||
        abduced.confidence < 0.0f || abduced.confidence > 1.0f) {
        std::cout << "FAILED (abduction bounds)\n";
        return false;
    }
    
    // Test revision
    auto revised = ggml_opencog_pln_revision(tv1, tv2);
    if (revised.strength < 0.0f || revised.strength > 1.0f ||
        revised.confidence < 0.0f || revised.confidence > 1.0f) {
        std::cout << "FAILED (revision bounds)\n";
        return false;
    }
    
    // Test modus ponens
    auto consequent = ggml_opencog_pln_modus_ponens(tv1, tv2);
    if (consequent.strength < 0.0f || consequent.strength > 1.0f ||
        consequent.confidence < 0.0f || consequent.confidence > 1.0f) {
        std::cout << "FAILED (modus ponens bounds)\n";
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

bool test_embeddings() {
    std::cout << "Testing embeddings... ";
    
    auto* atomspace = ggml_opencog_atomspace_new(64);
    struct ggml_opencog_truth_value tv = {0.8f, 0.6f};
    
    uint64_t id1 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Cat", tv, {});
    uint64_t id2 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv, {});
    
    auto* atom1 = ggml_opencog_get_atom(atomspace, id1);
    auto* atom2 = ggml_opencog_get_atom(atomspace, id2);
    
    // Check that embeddings were created
    if (!atom1 || !atom2 || 
        atom1->embedding_data.empty() || atom2->embedding_data.empty()) {
        std::cout << "FAILED (embedding creation)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Check embedding dimensions
    if (atom1->embedding_data.size() != 64 || atom2->embedding_data.size() != 64) {
        std::cout << "FAILED (embedding dimensions)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_similarity() {
    std::cout << "Testing similarity computation... ";
    
    auto* atomspace = ggml_opencog_atomspace_new(64);
    struct ggml_opencog_truth_value tv = {0.8f, 0.6f};
    
    uint64_t mammal_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Mammal", tv, {});
    uint64_t dog_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv, {});
    uint64_t cat_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Cat", tv, {});
    
    // Create links to establish relationships
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, 
                         "Dog->Mammal", tv, {dog_id, mammal_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, 
                         "Cat->Mammal", tv, {cat_id, mammal_id});
    
    // Compute similarity
    float sim = ggml_opencog_compute_similarity(atomspace, dog_id, cat_id);
    
    // Similarity should be in valid range
    if (sim < -1.0f || sim > 1.0f) {
        std::cout << "FAILED (similarity bounds)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Self-similarity should be 1.0
    float self_sim = ggml_opencog_compute_similarity(atomspace, dog_id, dog_id);
    if (fabsf(self_sim - 1.0f) > 0.01f) {
        std::cout << "FAILED (self similarity)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_attention() {
    std::cout << "Testing attention allocation (ECAN)... ";
    
    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv = {0.8f, 0.6f};
    
    uint64_t id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "TestConcept", tv, {});
    auto* atom = ggml_opencog_get_atom(atomspace, id);
    
    // Initial attention should be zero
    if (atom->sti != 0.0f || atom->lti != 0.0f || atom->vlti != 0.0f) {
        std::cout << "FAILED (initial attention)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Update attention
    ggml_opencog_update_attention(atomspace, id, 10.0f, 5.0f);
    
    // Check attention was updated
    if (fabsf(atom->sti - 10.0f) > 1e-6f || fabsf(atom->lti - 5.0f) > 1e-6f) {
        std::cout << "FAILED (attention update)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test clamping - STI should be clamped to [-100, 100]
    ggml_opencog_update_attention(atomspace, id, 200.0f, 0.0f);
    if (atom->sti > 100.0f) {
        std::cout << "FAILED (STI clamping)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_hebbian_learning() {
    std::cout << "Testing Hebbian learning... ";
    
    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv = {0.8f, 0.6f};
    
    // Create two concepts
    uint64_t dog_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv, {});
    uint64_t pet_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Pet", tv, {});
    
    auto* dog = ggml_opencog_get_atom(atomspace, dog_id);
    auto* pet = ggml_opencog_get_atom(atomspace, pet_id);
    
    // Compute initial similarity
    float initial_sim = ggml_opencog_compute_similarity(atomspace, dog_id, pet_id);
    
    // Apply Hebbian learning multiple times
    for (int i = 0; i < 10; i++) {
        ggml_opencog_hebbian_update(atomspace, dog_id, pet_id, 0.1f);
    }
    
    // Compute final similarity - should be higher
    float final_sim = ggml_opencog_compute_similarity(atomspace, dog_id, pet_id);
    
    if (final_sim <= initial_sim) {
        std::cout << "FAILED (similarity did not increase: " << initial_sim << " -> " << final_sim << ")\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test normalization
    ggml_opencog_normalize_embedding(atomspace, dog_id);
    
    // Compute embedding norm (should be close to 1.0)
    float norm = 0.0f;
    for (size_t i = 0; i < dog->embedding_data.size(); i++) {
        norm += dog->embedding_data[i] * dog->embedding_data[i];
    }
    norm = sqrtf(norm);
    
    if (fabsf(norm - 1.0f) > 0.01f) {
        std::cout << "FAILED (normalization: norm = " << norm << ")\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test link-based Hebbian learning
    uint64_t cat_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Cat", tv, {});
    uint64_t mammal_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Mammal", tv, {});
    
    uint64_t link_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                            "Pets", tv, {dog_id, cat_id, mammal_id});
    
    float cat_mammal_sim_before = ggml_opencog_compute_similarity(atomspace, cat_id, mammal_id);
    
    // Apply Hebbian learning to the link
    ggml_opencog_hebbian_update_link(atomspace, link_id, 0.1f);
    
    float cat_mammal_sim_after = ggml_opencog_compute_similarity(atomspace, cat_id, mammal_id);
    
    if (cat_mammal_sim_after <= cat_mammal_sim_before) {
        std::cout << "FAILED (link learning did not increase similarity)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

int main() {
    std::cout << "=== OpenCog GGML Tests ===\n\n";
    
    int passed = 0;
    int total = 10;
    
    if (test_atomspace_creation()) passed++;
    if (test_atom_creation()) passed++;
    if (test_atom_queries()) passed++;
    if (test_links()) passed++;
    if (test_reasoning()) passed++;
    if (test_cogserver()) passed++;
    if (test_embeddings()) passed++;
    if (test_similarity()) passed++;
    if (test_attention()) passed++;
    if (test_hebbian_learning()) passed++;
    
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