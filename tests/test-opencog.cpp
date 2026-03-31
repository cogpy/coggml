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

bool test_temporal_reasoning() {
    std::cout << "Testing temporal reasoning... ";
    
    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv = {0.9f, 0.8f};
    
    // Create events
    uint64_t wake_up_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "WakeUp", tv, {});
    uint64_t breakfast_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Breakfast", tv, {});
    uint64_t work_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Work", tv, {});
    
    // Set time intervals (using milliseconds since epoch)
    // WakeUp: 7:00-7:30 (time 1000-3000)
    ggml_opencog_set_time_interval(atomspace, wake_up_id, 1000, 3000);
    
    // Breakfast: 7:30-8:00 (time 3000-5000)
    ggml_opencog_set_time_interval(atomspace, breakfast_id, 3000, 5000);
    
    // Work: 9:00-17:00 (time 7000-15000)
    ggml_opencog_set_time_interval(atomspace, work_id, 7000, 15000);
    
    // Test: WakeUp happens before Breakfast
    if (!ggml_opencog_happens_before(atomspace, wake_up_id, breakfast_id)) {
        std::cout << "FAILED (WakeUp should happen before Breakfast)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test: Breakfast happens before Work
    if (!ggml_opencog_happens_before(atomspace, breakfast_id, work_id)) {
        std::cout << "FAILED (Breakfast should happen before Work)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test: Work doesn't happen before Breakfast
    if (ggml_opencog_happens_before(atomspace, work_id, breakfast_id)) {
        std::cout << "FAILED (Work should not happen before Breakfast)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test: Query atoms at specific time
    auto atoms_at_time = ggml_opencog_get_atoms_at_time(atomspace, 4000);
    if (atoms_at_time.size() != 1 || atoms_at_time[0] != breakfast_id) {
        std::cout << "FAILED (should find Breakfast at time 4000)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test: Query atoms in interval
    auto atoms_in_interval = ggml_opencog_get_atoms_in_interval(atomspace, 2000, 8000);
    if (atoms_in_interval.size() != 3) {
        std::cout << "FAILED (should find 3 atoms in interval 2000-8000)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test: Simultaneity
    uint64_t lunch_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Lunch", tv, {});
    ggml_opencog_set_time_interval(atomspace, lunch_id, 10000, 11000);
    
    // Lunch happens during Work
    if (!ggml_opencog_happens_during(atomspace, lunch_id, work_id)) {
        std::cout << "FAILED (Lunch should happen during Work)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Test: Temporal induction
    struct ggml_opencog_truth_value tv_link1 = {0.9f, 0.8f};
    struct ggml_opencog_truth_value tv_link2 = {0.85f, 0.75f};
    
    struct ggml_opencog_truth_value result = ggml_opencog_temporal_induction(tv_link1, tv_link2);
    
    // Result strength should be minimum of inputs
    if (result.strength > 0.85f || result.strength < 0.84f) {
        std::cout << "FAILED (temporal induction strength)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    // Result confidence should be reduced
    if (result.confidence >= tv_link1.confidence * tv_link2.confidence) {
        std::cout << "FAILED (temporal induction confidence should be reduced)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    
    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}


bool test_remove_atom() {
    std::cout << "Testing atom removal... ";

    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv = {0.8f, 0.6f};

    uint64_t id1 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv, {});
    uint64_t id2 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Cat", tv, {});

    size_t count_before = ggml_opencog_atom_count(atomspace);

    bool removed = ggml_opencog_remove_atom(atomspace, id1);
    if (!removed) {
        std::cout << "FAILED (remove returned false)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    if (ggml_opencog_atom_count(atomspace) != count_before - 1) {
        std::cout << "FAILED (count not decremented)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    if (ggml_opencog_get_atom(atomspace, id1) != nullptr) {
        std::cout << "FAILED (atom still accessible after removal)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Remove non-existent atom should return false
    bool removed_fake = ggml_opencog_remove_atom(atomspace, 99999);
    if (removed_fake) {
        std::cout << "FAILED (removing non-existent atom returned true)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Remaining atom should still be accessible
    if (ggml_opencog_get_atom(atomspace, id2) == nullptr) {
        std::cout << "FAILED (surviving atom inaccessible)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_pln_direct_values() {
    std::cout << "Testing PLN direct values... ";

    // Deduction: s = s_AB * s_BC (simplified)
    {
        struct ggml_opencog_truth_value ab = {0.9f, 0.8f};
        struct ggml_opencog_truth_value bc = {0.8f, 0.7f};
        auto result = ggml_opencog_pln_deduction(ab, bc);
        if (result.strength < 0.0f || result.strength > 1.0f ||
            result.confidence < 0.0f || result.confidence > 1.0f) {
            std::cout << "FAILED (deduction bounds)\n";
            return false;
        }
        // strength should relate to product of inputs
        if (result.strength > ab.strength || result.strength > bc.strength) {
            std::cout << "FAILED (deduction should reduce strength)\n";
            return false;
        }
    }

    // Induction: strength bounded [0,1], confidence reduced
    {
        struct ggml_opencog_truth_value ac = {0.8f, 0.9f};
        struct ggml_opencog_truth_value bc = {0.7f, 0.8f};
        auto result = ggml_opencog_pln_induction(ac, bc);
        if (result.strength < 0.0f || result.strength > 1.0f) {
            std::cout << "FAILED (induction strength bounds)\n";
            return false;
        }
        if (result.confidence >= ac.confidence) {
            std::cout << "FAILED (induction should reduce confidence)\n";
            return false;
        }
    }

    // Abduction: strength bounded [0,1], confidence reduced
    {
        struct ggml_opencog_truth_value ac = {0.8f, 0.9f};
        struct ggml_opencog_truth_value bc = {0.7f, 0.8f};
        auto result = ggml_opencog_pln_abduction(ac, bc);
        if (result.strength < 0.0f || result.strength > 1.0f) {
            std::cout << "FAILED (abduction strength bounds)\n";
            return false;
        }
        if (result.confidence >= bc.confidence) {
            std::cout << "FAILED (abduction should reduce confidence)\n";
            return false;
        }
    }

    // Revision: weighted combination
    {
        struct ggml_opencog_truth_value tv1 = {0.9f, 0.8f};
        struct ggml_opencog_truth_value tv2 = {0.5f, 0.2f};
        auto result = ggml_opencog_pln_revision(tv1, tv2);
        // Result strength should be closer to tv1 (higher confidence)
        if (result.strength < 0.7f) {
            std::cout << "FAILED (revision should weight toward higher confidence)\n";
            return false;
        }
        if (result.confidence <= tv1.confidence) {
            std::cout << "FAILED (revision should increase confidence)\n";
            return false;
        }
    }

    // Modus ponens: s = s_impl * s_antecedent
    {
        struct ggml_opencog_truth_value impl = {0.9f, 0.8f};
        struct ggml_opencog_truth_value ante = {0.8f, 0.9f};
        auto result = ggml_opencog_pln_modus_ponens(impl, ante);
        float expected_s = impl.strength * ante.strength;
        if (fabsf(result.strength - expected_s) > 1e-5f) {
            std::cout << "FAILED (modus ponens strength: got " << result.strength
                      << " expected " << expected_s << ")\n";
            return false;
        }
    }

    std::cout << "PASSED\n";
    return true;
}

bool test_pln_extended_rules() {
    std::cout << "Testing PLN extended rules... ";

    // Conjunction: s = s1 * s2
    {
        struct ggml_opencog_truth_value tv1 = {0.8f, 0.9f};
        struct ggml_opencog_truth_value tv2 = {0.6f, 0.7f};
        auto result = ggml_opencog_pln_conjunction(tv1, tv2);
        float expected = 0.8f * 0.6f;
        if (fabsf(result.strength - expected) > 1e-5f) {
            std::cout << "FAILED (conjunction strength: " << result.strength
                      << " expected " << expected << ")\n";
            return false;
        }
        if (fabsf(result.confidence - fminf(tv1.confidence, tv2.confidence)) > 1e-5f) {
            std::cout << "FAILED (conjunction confidence)\n";
            return false;
        }
    }

    // Disjunction: s = 1 - (1-s1)*(1-s2)
    {
        struct ggml_opencog_truth_value tv1 = {0.8f, 0.9f};
        struct ggml_opencog_truth_value tv2 = {0.6f, 0.7f};
        auto result = ggml_opencog_pln_disjunction(tv1, tv2);
        float expected = 1.0f - (1.0f - 0.8f) * (1.0f - 0.6f);
        if (fabsf(result.strength - expected) > 1e-5f) {
            std::cout << "FAILED (disjunction strength: " << result.strength
                      << " expected " << expected << ")\n";
            return false;
        }
        // Disjunction strength should be >= each input
        if (result.strength < tv1.strength || result.strength < tv2.strength) {
            std::cout << "FAILED (disjunction should be >= each input)\n";
            return false;
        }
    }

    // Negation: s = 1 - s, confidence unchanged
    {
        struct ggml_opencog_truth_value tv = {0.7f, 0.8f};
        auto result = ggml_opencog_pln_negation(tv);
        if (fabsf(result.strength - 0.3f) > 1e-5f) {
            std::cout << "FAILED (negation strength: " << result.strength << ")\n";
            return false;
        }
        if (fabsf(result.confidence - tv.confidence) > 1e-5f) {
            std::cout << "FAILED (negation should preserve confidence)\n";
            return false;
        }
    }

    // Negation of 0 should give 1
    {
        struct ggml_opencog_truth_value tv = {0.0f, 1.0f};
        auto result = ggml_opencog_pln_negation(tv);
        if (fabsf(result.strength - 1.0f) > 1e-5f) {
            std::cout << "FAILED (negation of 0 should be 1)\n";
            return false;
        }
    }

    std::cout << "PASSED\n";
    return true;
}

bool test_variable_nodes() {
    std::cout << "Testing variable node matching... ";

    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv = {0.8f, 0.7f};

    // Create concept nodes
    uint64_t dog_id    = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv, {});
    uint64_t animal_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Animal", tv, {});

    // Create a variable node $X
    struct ggml_opencog_truth_value var_tv = {0.5f, 0.0f};
    uint64_t var_x = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_VARIABLE_NODE, "$X", var_tv, {});

    // Variable should match any concept node
    struct ggml_opencog_binding b1;
    bool match1 = ggml_opencog_match_with_binding(atomspace, var_x, dog_id, &b1);
    if (!match1) {
        std::cout << "FAILED (variable should match concept node)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    if (b1.get(var_x) != dog_id) {
        std::cout << "FAILED (binding not recorded correctly)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Exact match: concept node only matches itself
    struct ggml_opencog_binding b2;
    bool match2 = ggml_opencog_match_with_binding(atomspace, dog_id, animal_id, &b2);
    if (match2) {
        std::cout << "FAILED (Dog should not match Animal)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Exact match: same atom
    struct ggml_opencog_binding b3;
    bool match3 = ggml_opencog_match_with_binding(atomspace, dog_id, dog_id, &b3);
    if (!match3) {
        std::cout << "FAILED (atom should match itself)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Consistent binding: variable already bound should only match same atom
    struct ggml_opencog_binding b4;
    b4.bind(var_x, dog_id);
    bool match4 = ggml_opencog_match_with_binding(atomspace, var_x, animal_id, &b4);
    if (match4) {
        std::cout << "FAILED (bound variable should not match different atom)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Variable should match the same atom it's already bound to
    struct ggml_opencog_binding b5;
    b5.bind(var_x, dog_id);
    bool match5 = ggml_opencog_match_with_binding(atomspace, var_x, dog_id, &b5);
    if (!match5) {
        std::cout << "FAILED (bound variable should match its binding)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_forward_chaining() {
    std::cout << "Testing forward chaining... ";

    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv_high = {0.9f, 0.8f};

    // Create knowledge: Socrates->Human->Animal->Mortal
    uint64_t socrates = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Socrates", tv_high, {});
    uint64_t human    = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Human", tv_high, {});
    uint64_t animal   = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Animal", tv_high, {});
    uint64_t mortal   = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Mortal", tv_high, {});

    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                          "Socrates->Human", tv_high, {socrates, human});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                          "Human->Animal", tv_high, {human, animal});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                          "Animal->Mortal", tv_high, {animal, mortal});

    size_t before = ggml_opencog_atom_count(atomspace);

    // Run forward chaining
    auto derived = ggml_opencog_forward_chain(atomspace, 10);

    size_t after = ggml_opencog_atom_count(atomspace);

    // Should have derived new links (at least Socrates->Animal)
    if (derived.empty()) {
        std::cout << "FAILED (no atoms derived)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    if (after <= before) {
        std::cout << "FAILED (atomspace did not grow)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Check that Socrates->Mortal was eventually derived
    bool found_socrates_mortal = false;
    auto links = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_INHERITANCE_LINK);
    for (uint64_t lid : links) {
        auto* link = ggml_opencog_get_atom(atomspace, lid);
        if (link && link->outgoing.size() >= 2 &&
            link->outgoing[0] == socrates && link->outgoing[1] == mortal) {
            found_socrates_mortal = true;
            break;
        }
    }

    if (!found_socrates_mortal) {
        std::cout << "FAILED (Socrates->Mortal not derived)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_backward_chaining() {
    std::cout << "Testing backward chaining... ";

    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv_high = {0.9f, 0.8f};

    // Create knowledge: Socrates->Human, Human->Animal
    uint64_t socrates = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Socrates", tv_high, {});
    uint64_t human    = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Human", tv_high, {});
    uint64_t animal   = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Animal", tv_high, {});

    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                          "Socrates->Human", tv_high, {socrates, human});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                          "Human->Animal", tv_high, {human, animal});

    // Goal: prove Socrates->Animal (not directly present)
    struct ggml_opencog_truth_value zero_tv = {0.0f, 0.0f};
    uint64_t goal = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                          "Socrates->Animal(goal)", zero_tv, {socrates, animal});

    std::vector<uint64_t> path;
    bool proved = ggml_opencog_backward_chain(atomspace, goal, 5, path);

    if (!proved) {
        std::cout << "FAILED (could not prove Socrates->Animal)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    if (path.empty()) {
        std::cout << "FAILED (derivation path is empty)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Goal TV should be updated
    auto* goal_atom = ggml_opencog_get_atom(atomspace, goal);
    if (!goal_atom || goal_atom->tv.confidence < 0.1f) {
        std::cout << "FAILED (goal TV not updated)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Test with an un-provable goal
    uint64_t dog    = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv_high, {});
    uint64_t planet = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Planet", tv_high, {});
    uint64_t unprovable = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                                "Dog->Planet(goal)", zero_tv, {dog, planet});
    std::vector<uint64_t> path2;
    bool proved2 = ggml_opencog_backward_chain(atomspace, unprovable, 3, path2);
    // This should not be proved (no relevant links exist)
    if (proved2) {
        std::cout << "FAILED (unprovable goal should not be proved)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_find_matching() {
    std::cout << "Testing find_matching... ";

    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv = {0.8f, 0.7f};

    uint64_t dog    = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv, {});
    uint64_t cat    = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Cat", tv, {});
    uint64_t animal = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Animal", tv, {});

    // Exact pattern: Dog matches only itself
    auto exact_matches = ggml_opencog_find_matching(atomspace, dog);
    bool found_dog = false;
    for (auto& p : exact_matches) {
        if (p.first == dog) { found_dog = true; break; }
    }
    if (!found_dog) {
        std::cout << "FAILED (exact pattern should match itself)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Variable pattern should match all concept nodes
    struct ggml_opencog_truth_value var_tv = {0.5f, 0.0f};
    uint64_t var_x = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_VARIABLE_NODE, "$X", var_tv, {});
    auto var_matches = ggml_opencog_find_matching(atomspace, var_x);

    // Should find at least dog, cat, animal (and possibly the variable itself)
    size_t concept_matches = 0;
    for (auto& p : var_matches) {
        auto* a = ggml_opencog_get_atom(atomspace, p.first);
        if (a && a->type == GGML_OPENCOG_CONCEPT_NODE) ++concept_matches;
    }
    if (concept_matches < 3) {
        std::cout << "FAILED (variable should match all concept nodes, got "
                  << concept_matches << ")\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_atom_count() {
    std::cout << "Testing atom count... ";

    auto* atomspace = ggml_opencog_atomspace_new(32);
    if (ggml_opencog_atom_count(atomspace) != 0) {
        std::cout << "FAILED (empty atomspace should have 0 atoms)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    struct ggml_opencog_truth_value tv = {0.8f, 0.6f};
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "A", tv, {});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "B", tv, {});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_PREDICATE_NODE, "P", tv, {});

    if (ggml_opencog_atom_count(atomspace) != 3) {
        std::cout << "FAILED (expected 3, got "
                  << ggml_opencog_atom_count(atomspace) << ")\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    ggml_opencog_atomspace_free(atomspace);
    std::cout << "PASSED\n";
    return true;
}

bool test_new_atom_types() {
    std::cout << "Testing new atom types... ";

    auto* atomspace = ggml_opencog_atomspace_new(32);
    struct ggml_opencog_truth_value tv = {0.8f, 0.7f};

    // NumberNode
    uint64_t num = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_NUMBER_NODE, "42", tv, {});
    auto* num_atom = ggml_opencog_get_atom(atomspace, num);
    if (!num_atom || num_atom->type != GGML_OPENCOG_NUMBER_NODE) {
        std::cout << "FAILED (NumberNode creation)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // VariableNode
    uint64_t var = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_VARIABLE_NODE, "$V", tv, {});
    auto* var_atom = ggml_opencog_get_atom(atomspace, var);
    if (!var_atom || var_atom->type != GGML_OPENCOG_VARIABLE_NODE) {
        std::cout << "FAILED (VariableNode creation)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // AndLink, OrLink, NotLink
    uint64_t a = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "A", tv, {});
    uint64_t b = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "B", tv, {});

    uint64_t and_link = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_AND_LINK, "A AND B", tv, {a, b});
    uint64_t or_link  = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_OR_LINK,  "A OR B",  tv, {a, b});
    uint64_t not_link = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_NOT_LINK, "NOT A",   tv, {a});

    auto* and_atom = ggml_opencog_get_atom(atomspace, and_link);
    auto* or_atom  = ggml_opencog_get_atom(atomspace, or_link);
    auto* not_atom = ggml_opencog_get_atom(atomspace, not_link);

    if (!and_atom || and_atom->type != GGML_OPENCOG_AND_LINK) {
        std::cout << "FAILED (AndLink creation)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    if (!or_atom || or_atom->type != GGML_OPENCOG_OR_LINK) {
        std::cout << "FAILED (OrLink creation)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    if (!not_atom || not_atom->type != GGML_OPENCOG_NOT_LINK) {
        std::cout << "FAILED (NotLink creation)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // ImplicationLink and ListLink
    uint64_t impl = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_IMPLICATION_LINK,
                                          "A->B", tv, {a, b});
    uint64_t list = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_LIST_LINK,
                                          "list(A,B)", tv, {a, b});

    if (!ggml_opencog_get_atom(atomspace, impl) ||
        ggml_opencog_get_atom(atomspace, impl)->type != GGML_OPENCOG_IMPLICATION_LINK) {
        std::cout << "FAILED (ImplicationLink creation)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }
    if (!ggml_opencog_get_atom(atomspace, list) ||
        ggml_opencog_get_atom(atomspace, list)->type != GGML_OPENCOG_LIST_LINK) {
        std::cout << "FAILED (ListLink creation)\n";
        ggml_opencog_atomspace_free(atomspace);
        return false;
    }

    // Query by new types
    auto vars = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_VARIABLE_NODE);
    if (vars.size() != 1 || vars[0] != var) {
        std::cout << "FAILED (VariableNode query)\n";
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
    int total = 0;

    auto run = [&](bool(*fn)()) {
        ++total;
        if (fn()) ++passed;
    };

    run(test_atomspace_creation);
    run(test_atom_creation);
    run(test_atom_queries);
    run(test_links);
    run(test_reasoning);
    run(test_cogserver);
    run(test_embeddings);
    run(test_similarity);
    run(test_attention);
    run(test_hebbian_learning);
    run(test_temporal_reasoning);
    run(test_remove_atom);
    run(test_pln_direct_values);
    run(test_pln_extended_rules);
    run(test_variable_nodes);
    run(test_forward_chaining);
    run(test_backward_chaining);
    run(test_find_matching);
    run(test_atom_count);
    run(test_new_atom_types);

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