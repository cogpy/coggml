#include "ggml-opencog.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>
#include <cmath>

// Advanced reasoning agent that performs syllogistic logic
void syllogistic_reasoning_agent(struct ggml_opencog_atomspace* atomspace) {
    std::cout << "SyllogisticReasoning Agent running...\n";
    
    // Look for syllogistic patterns: A->B, B->C => A->C
    auto inheritance_links = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_INHERITANCE_LINK);
    
    std::cout << "  Analyzing " << inheritance_links.size() << " inheritance links...\n";
    
    for (size_t i = 0; i < inheritance_links.size(); i++) {
        for (size_t j = i + 1; j < inheritance_links.size(); j++) {
            auto* link1 = ggml_opencog_get_atom(atomspace, inheritance_links[i]);
            auto* link2 = ggml_opencog_get_atom(atomspace, inheritance_links[j]);
            
            if (!link1 || !link2 || link1->outgoing.size() < 2 || link2->outgoing.size() < 2) continue;
            
            // Check if we can form a syllogism: A->B + B->C => A->C
            if (link1->outgoing[1] == link2->outgoing[0]) {
                // Found: link1 = A->B, link2 = B->C
                uint64_t A = link1->outgoing[0];
                uint64_t B = link1->outgoing[1];
                uint64_t C = link2->outgoing[1];
                
                auto* atom_A = ggml_opencog_get_atom(atomspace, A);
                auto* atom_B = ggml_opencog_get_atom(atomspace, B);
                auto* atom_C = ggml_opencog_get_atom(atomspace, C);
                
                if (atom_A && atom_B && atom_C) {
                    // Apply PLN deduction
                    auto deduced_tv = ggml_opencog_pln_deduction(link1->tv, link2->tv);
                    
                    std::cout << "  Deduction: " << atom_A->name << "->" << atom_B->name 
                              << " + " << atom_B->name << "->" << atom_C->name
                              << " => " << atom_A->name << "->" << atom_C->name
                              << " (strength: " << std::fixed << std::setprecision(3) << deduced_tv.strength
                              << ", confidence: " << deduced_tv.confidence << ")\n";
                    
                    // Check if this inference already exists
                    bool exists = false;
                    for (uint64_t existing_id : inheritance_links) {
                        auto* existing = ggml_opencog_get_atom(atomspace, existing_id);
                        if (existing && existing->outgoing.size() >= 2 && 
                            existing->outgoing[0] == A && existing->outgoing[1] == C) {
                            exists = true;
                            break;
                        }
                    }
                    
                    // Add new inference if it doesn't exist and has sufficient confidence
                    if (!exists && deduced_tv.confidence > 0.1f) {
                        std::string link_name = std::string(atom_A->name) + "->" + std::string(atom_C->name) + "(inferred)";
                        uint64_t new_link = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                                                 link_name.c_str(), deduced_tv, {A, C});
                        std::cout << "    -> Created new inference link with ID: " << new_link << "\n";
                    }
                }
            }
        }
    }
}

// Agent that performs pattern matching and similarity detection
void pattern_matching_agent(struct ggml_opencog_atomspace* atomspace) {
    std::cout << "PatternMatching Agent running...\n";
    
    auto concepts = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_CONCEPT_NODE);
    
    // Look for similar concepts and create similarity links
    for (size_t i = 0; i < concepts.size(); i++) {
        for (size_t j = i + 1; j < concepts.size(); j++) {
            auto* concept1 = ggml_opencog_get_atom(atomspace, concepts[i]);
            auto* concept2 = ggml_opencog_get_atom(atomspace, concepts[j]);
            
            if (!concept1 || !concept2) continue;
            
            // Simple similarity heuristic based on shared incoming links
            std::vector<uint64_t> shared_parents;
            for (uint64_t link1 : concept1->incoming) {
                for (uint64_t link2 : concept2->incoming) {
                    auto* l1 = ggml_opencog_get_atom(atomspace, link1);
                    auto* l2 = ggml_opencog_get_atom(atomspace, link2);
                    if (l1 && l2 && l1->type == GGML_OPENCOG_INHERITANCE_LINK && 
                        l2->type == GGML_OPENCOG_INHERITANCE_LINK &&
                        l1->outgoing.size() >= 2 && l2->outgoing.size() >= 2 &&
                        l1->outgoing[1] == l2->outgoing[1]) {
                        shared_parents.push_back(l1->outgoing[1]);
                    }
                }
            }
            
            if (!shared_parents.empty()) {
                float similarity_strength = fminf(0.9f, 0.3f + 0.2f * shared_parents.size());
                struct ggml_opencog_truth_value similarity_tv = {similarity_strength, 0.7f};
                
                std::string similarity_name = std::string("Similar(") + concept1->name + "," + concept2->name + ")";
                
                // Check if similarity link already exists
                bool exists = false;
                auto similarities = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_SIMILARITY_LINK);
                for (uint64_t sim_id : similarities) {
                    auto* sim = ggml_opencog_get_atom(atomspace, sim_id);
                    if (sim && sim->outgoing.size() >= 2 &&
                        ((sim->outgoing[0] == concepts[i] && sim->outgoing[1] == concepts[j]) ||
                         (sim->outgoing[0] == concepts[j] && sim->outgoing[1] == concepts[i]))) {
                        exists = true;
                        break;
                    }
                }
                
                if (!exists) {
                    uint64_t sim_link = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_SIMILARITY_LINK,
                                                             similarity_name.c_str(), similarity_tv, 
                                                             {concepts[i], concepts[j]});
                    std::cout << "  Found similarity: " << concept1->name << " ~ " << concept2->name
                              << " (shared " << shared_parents.size() << " parents, strength: " 
                              << std::fixed << std::setprecision(3) << similarity_strength << ")\n";
                    std::cout << "    -> Created similarity link with ID: " << sim_link << "\n";
                }
            }
        }
    }
}

int main() {
    std::cout << "=== OpenCog Advanced Reasoning Demo ===\n\n";
    
    // Initialize AtomSpace with higher-dimensional embeddings for more complex reasoning
    auto* atomspace = ggml_opencog_atomspace_new(128);
    
    std::cout << "1. Building Knowledge Base...\n";
    
    // Create a more complex taxonomy
    struct ggml_opencog_truth_value tv_certain = {0.95f, 0.9f};
    struct ggml_opencog_truth_value tv_likely = {0.8f, 0.7f};
    // struct ggml_opencog_truth_value tv_possible = {0.6f, 0.5f}; // Unused for now
    
    // Animals
    uint64_t animal_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Animal", tv_certain, {});
    uint64_t mammal_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Mammal", tv_certain, {});
    uint64_t bird_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Bird", tv_certain, {});
    uint64_t reptile_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Reptile", tv_certain, {});
    
    // Specific animals
    uint64_t dog_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dog", tv_certain, {});
    uint64_t cat_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Cat", tv_certain, {});
    uint64_t robin_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Robin", tv_certain, {});
    uint64_t snake_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Snake", tv_certain, {});
    
    // Properties
    uint64_t living_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Living", tv_certain, {});
    uint64_t mobile_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Mobile", tv_certain, {});
    uint64_t warm_blooded_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "WarmBlooded", tv_certain, {});
    uint64_t flying_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "CanFly", tv_certain, {});
    
    std::cout << "   Created " << 12 << " concept nodes\n";
    
    // Create inheritance hierarchy
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Mammal->Animal", tv_certain, {mammal_id, animal_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Bird->Animal", tv_certain, {bird_id, animal_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Reptile->Animal", tv_certain, {reptile_id, animal_id});
    
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Dog->Mammal", tv_certain, {dog_id, mammal_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Cat->Mammal", tv_certain, {cat_id, mammal_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Robin->Bird", tv_certain, {robin_id, bird_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Snake->Reptile", tv_certain, {snake_id, reptile_id});
    
    // Properties
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Animal->Living", tv_certain, {animal_id, living_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Animal->Mobile", tv_likely, {animal_id, mobile_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Mammal->WarmBlooded", tv_certain, {mammal_id, warm_blooded_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Bird->WarmBlooded", tv_certain, {bird_id, warm_blooded_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Bird->CanFly", tv_likely, {bird_id, flying_id});
    
    std::cout << "   Created " << 12 << " inheritance links\n\n";
    
    std::cout << "2. Setting up Advanced CogServer...\n";
    
    // Create CogServer with reasoning agents
    auto* cogserver = ggml_opencog_cogserver_new(atomspace);
    
    struct ggml_opencog_mind_agent syllogism_agent;
    strcpy(syllogism_agent.name, "SyllogisticReasoner");
    syllogism_agent.process = syllogistic_reasoning_agent;
    syllogism_agent.frequency = 2;
    syllogism_agent.last_run = 0;
    
    struct ggml_opencog_mind_agent pattern_agent;
    strcpy(pattern_agent.name, "PatternMatcher");
    pattern_agent.process = pattern_matching_agent;
    pattern_agent.frequency = 3;
    pattern_agent.last_run = 0;
    
    ggml_opencog_cogserver_add_agent(cogserver, &syllogism_agent);
    ggml_opencog_cogserver_add_agent(cogserver, &pattern_agent);
    
    std::cout << "3. Running Advanced Reasoning Cycles...\n";
    
    ggml_opencog_cogserver_start(cogserver);
    
    for (int cycle = 1; cycle <= 8; cycle++) {
        std::cout << "\n--- Cycle " << cycle << " ---\n";
        ggml_opencog_cogserver_run_cycle(cogserver);
    }
    
    ggml_opencog_cogserver_stop(cogserver);
    
    std::cout << "\n4. Final Knowledge Base Statistics...\n";
    
    auto final_concepts = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_CONCEPT_NODE);
    auto final_inheritance = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_INHERITANCE_LINK);
    auto final_similarities = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_SIMILARITY_LINK);
    
    std::cout << "   Concept Nodes: " << final_concepts.size() << "\n";
    std::cout << "   Inheritance Links: " << final_inheritance.size() << "\n";
    std::cout << "   Similarity Links: " << final_similarities.size() << "\n";
    std::cout << "   Total Atoms: " << (final_concepts.size() + final_inheritance.size() + final_similarities.size()) << "\n";
    
    std::cout << "\n5. Querying Derived Knowledge...\n";
    
    // Look for derived properties of specific animals
    auto* dog = ggml_opencog_get_atom(atomspace, dog_id);
    if (dog) {
        std::cout << "   Properties of " << dog->name << ":\n";
        for (uint64_t incoming_link : dog->incoming) {
            auto* link = ggml_opencog_get_atom(atomspace, incoming_link);
            if (link && link->type == GGML_OPENCOG_INHERITANCE_LINK && link->outgoing.size() >= 2) {
                auto* property = ggml_opencog_get_atom(atomspace, link->outgoing[1]);
                if (property) {
                    std::cout << "     - " << property->name 
                              << " (strength: " << std::fixed << std::setprecision(3) << link->tv.strength << ")\n";
                }
            }
        }
    }
    
    std::cout << "\n6. Cleanup...\n";
    
    ggml_opencog_cogserver_free(cogserver);
    ggml_opencog_atomspace_free(atomspace);
    
    std::cout << "\n=== Advanced Reasoning Demo Complete ===\n";
    return 0;
}