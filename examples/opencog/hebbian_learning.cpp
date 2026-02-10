#include "ggml-opencog.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

// Demonstrate Hebbian learning: "Neurons that fire together, wire together"
// This example shows how repeated co-activation of concepts strengthens their semantic relationship

void print_similarity_matrix(struct ggml_opencog_atomspace* atomspace,
                             const std::vector<uint64_t>& atom_ids,
                             const std::vector<std::string>& names) {
    std::cout << "\nSimilarity Matrix:\n";
    std::cout << std::setw(12) << "";
    for (const auto& name : names) {
        std::cout << std::setw(10) << name;
    }
    std::cout << "\n";
    
    for (size_t i = 0; i < atom_ids.size(); i++) {
        std::cout << std::setw(12) << names[i];
        for (size_t j = 0; j < atom_ids.size(); j++) {
            if (i == j) {
                std::cout << std::setw(10) << "1.000";
            } else {
                float sim = ggml_opencog_compute_similarity(atomspace, atom_ids[i], atom_ids[j]);
                std::cout << std::setw(10) << std::fixed << std::setprecision(3) << sim;
            }
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout << "=== OpenCog Hebbian Learning Demo ===\n\n";
    
    // Create AtomSpace with 64-dimensional embeddings
    auto* atomspace = ggml_opencog_atomspace_new(64);
    struct ggml_opencog_truth_value tv = {0.9f, 0.8f};
    
    // === Example 1: Learning semantic associations ===
    std::cout << "Example 1: Learning Semantic Associations\n";
    std::cout << "==========================================\n\n";
    
    // Create concepts for animals
    std::cout << "Creating animal concepts...\n";
    uint64_t dog_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv, {});
    uint64_t cat_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Cat", tv, {});
    uint64_t bird_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Bird", tv, {});
    uint64_t fish_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Fish", tv, {});
    
    std::vector<uint64_t> animal_ids = {dog_id, cat_id, bird_id, fish_id};
    std::vector<std::string> animal_names = {"Dog", "Cat", "Bird", "Fish"};
    
    std::cout << "Initial similarities (before learning):\n";
    print_similarity_matrix(atomspace, animal_ids, animal_names);
    
    // Simulate learning: Dog and Cat are both pets and mammals
    // They are frequently thought of together
    std::cout << "\nSimulating co-activation of Dog and Cat (10 iterations)...\n";
    for (int i = 0; i < 10; i++) {
        ggml_opencog_hebbian_update(atomspace, dog_id, cat_id, 0.1f);
    }
    
    std::cout << "After learning Dog-Cat association:\n";
    print_similarity_matrix(atomspace, animal_ids, animal_names);
    
    // === Example 2: Learning from relationships ===
    std::cout << "\n\nExample 2: Learning from Hierarchical Relationships\n";
    std::cout << "===================================================\n\n";
    
    // Create a taxonomic hierarchy
    uint64_t mammal_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Mammal", tv, {});
    uint64_t animal_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Animal", tv, {});
    
    // Create inheritance links
    std::cout << "Creating inheritance hierarchy:\n";
    std::cout << "  Dog -> Mammal\n";
    std::cout << "  Cat -> Mammal\n";
    std::cout << "  Mammal -> Animal\n\n";
    
    uint64_t dog_mammal_link = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                                     "Dog->Mammal", tv, {dog_id, mammal_id});
    uint64_t cat_mammal_link = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                                     "Cat->Mammal", tv, {cat_id, mammal_id});
    uint64_t mammal_animal_link = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                                        "Mammal->Animal", tv, {mammal_id, animal_id});
    
    float dog_mammal_sim_before = ggml_opencog_compute_similarity(atomspace, dog_id, mammal_id);
    std::cout << "Dog-Mammal similarity before link learning: " << std::fixed << std::setprecision(4) 
              << dog_mammal_sim_before << "\n";
    
    // Apply Hebbian learning to the links
    std::cout << "Applying Hebbian learning to inheritance links (5 iterations)...\n";
    for (int i = 0; i < 5; i++) {
        ggml_opencog_hebbian_update_link(atomspace, dog_mammal_link, 0.15f);
        ggml_opencog_hebbian_update_link(atomspace, cat_mammal_link, 0.15f);
        ggml_opencog_hebbian_update_link(atomspace, mammal_animal_link, 0.15f);
    }
    
    float dog_mammal_sim_after = ggml_opencog_compute_similarity(atomspace, dog_id, mammal_id);
    std::cout << "Dog-Mammal similarity after link learning:  " << std::fixed << std::setprecision(4) 
              << dog_mammal_sim_after << "\n";
    std::cout << "Improvement: " << std::fixed << std::setprecision(4) 
              << (dog_mammal_sim_after - dog_mammal_sim_before) << "\n";
    
    // === Example 3: Normalization for stability ===
    std::cout << "\n\nExample 3: Embedding Normalization\n";
    std::cout << "===================================\n\n";
    
    // Create a new concept and apply strong learning
    uint64_t horse_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Horse", tv, {});
    
    auto* horse = ggml_opencog_get_atom(atomspace, horse_id);
    
    // Compute initial norm
    float initial_norm = 0.0f;
    for (size_t i = 0; i < horse->embedding_data.size(); i++) {
        initial_norm += horse->embedding_data[i] * horse->embedding_data[i];
    }
    initial_norm = sqrtf(initial_norm);
    
    std::cout << "Initial embedding norm: " << std::fixed << std::setprecision(4) << initial_norm << "\n";
    
    // Apply aggressive Hebbian learning
    std::cout << "Applying aggressive learning (20 iterations with high rate)...\n";
    for (int i = 0; i < 20; i++) {
        ggml_opencog_hebbian_update(atomspace, horse_id, mammal_id, 0.3f);
    }
    
    // Compute norm after learning
    float norm_after_learning = 0.0f;
    for (size_t i = 0; i < horse->embedding_data.size(); i++) {
        norm_after_learning += horse->embedding_data[i] * horse->embedding_data[i];
    }
    norm_after_learning = sqrtf(norm_after_learning);
    
    std::cout << "Norm after learning:    " << std::fixed << std::setprecision(4) << norm_after_learning << "\n";
    
    // Normalize
    std::cout << "Applying normalization...\n";
    ggml_opencog_normalize_embedding(atomspace, horse_id);
    
    // Compute norm after normalization
    float norm_after_norm = 0.0f;
    for (size_t i = 0; i < horse->embedding_data.size(); i++) {
        norm_after_norm += horse->embedding_data[i] * horse->embedding_data[i];
    }
    norm_after_norm = sqrtf(norm_after_norm);
    
    std::cout << "Norm after normalization: " << std::fixed << std::setprecision(4) << norm_after_norm << "\n";
    std::cout << "(Should be close to 1.0 for numerical stability)\n";
    
    // === Example 4: Multi-concept learning scenario ===
    std::cout << "\n\nExample 4: Multi-Concept Learning Scenario\n";
    std::cout << "==========================================\n\n";
    
    std::cout << "Scenario: Learning about pets and their characteristics\n\n";
    
    // Create concepts
    uint64_t friendly_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Friendly", tv, {});
    uint64_t furry_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Furry", tv, {});
    uint64_t pet_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Pet", tv, {});
    
    std::vector<uint64_t> pet_concept_ids = {dog_id, cat_id, friendly_id, furry_id, pet_id};
    std::vector<std::string> pet_concept_names = {"Dog", "Cat", "Friendly", "Furry", "Pet"};
    
    std::cout << "Initial state:\n";
    print_similarity_matrix(atomspace, pet_concept_ids, pet_concept_names);
    
    // Simulate learning from experiences:
    // - Dogs are friendly and furry pets
    // - Cats are furry pets (less consistently friendly)
    std::cout << "\nLearning from experiences:\n";
    std::cout << "  Experience 1: Dog is friendly (strong association)\n";
    for (int i = 0; i < 15; i++) {
        ggml_opencog_hebbian_update(atomspace, dog_id, friendly_id, 0.1f);
    }
    
    std::cout << "  Experience 2: Dog is furry\n";
    for (int i = 0; i < 12; i++) {
        ggml_opencog_hebbian_update(atomspace, dog_id, furry_id, 0.1f);
    }
    
    std::cout << "  Experience 3: Dog is a pet\n";
    for (int i = 0; i < 15; i++) {
        ggml_opencog_hebbian_update(atomspace, dog_id, pet_id, 0.1f);
    }
    
    std::cout << "  Experience 4: Cat is furry\n";
    for (int i = 0; i < 12; i++) {
        ggml_opencog_hebbian_update(atomspace, cat_id, furry_id, 0.1f);
    }
    
    std::cout << "  Experience 5: Cat is a pet\n";
    for (int i = 0; i < 10; i++) {
        ggml_opencog_hebbian_update(atomspace, cat_id, pet_id, 0.1f);
    }
    
    std::cout << "  Experience 6: Cat can be friendly (weaker association)\n";
    for (int i = 0; i < 5; i++) {
        ggml_opencog_hebbian_update(atomspace, cat_id, friendly_id, 0.1f);
    }
    
    std::cout << "\nAfter learning:\n";
    print_similarity_matrix(atomspace, pet_concept_ids, pet_concept_names);
    
    std::cout << "\nObservations:\n";
    std::cout << "- Dog has high similarity with Friendly, Furry, and Pet\n";
    std::cout << "- Cat has high similarity with Furry and Pet\n";
    std::cout << "- Cat-Friendly similarity is lower (fewer co-activations)\n";
    std::cout << "- Friendly and Furry gained some similarity (both associated with pets)\n";
    
    // Clean up
    ggml_opencog_atomspace_free(atomspace);
    
    std::cout << "\n=== Demo Complete ===\n";
    std::cout << "\nKey Takeaways:\n";
    std::cout << "1. Hebbian learning strengthens connections between co-activated concepts\n";
    std::cout << "2. Link-based learning propagates similarity through relationships\n";
    std::cout << "3. Normalization maintains numerical stability\n";
    std::cout << "4. Learning strength reflects frequency and consistency of co-activation\n";
    
    return 0;
}
