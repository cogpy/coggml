#include "ggml-opencog.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <algorithm>

// Agent that uses embedding similarity for reasoning
void similarity_based_reasoning_agent(struct ggml_opencog_atomspace* atomspace) {
    std::cout << "SimilarityBasedReasoning Agent running...\n";
    
    auto concepts = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_CONCEPT_NODE);
    
    // Find pairs of similar concepts using the new similarity function
    std::cout << "  Computing embedding-based similarities...\n";
    int count = 0;
    for (size_t i = 0; i < concepts.size() && i < 5; i++) {
        for (size_t j = i + 1; j < concepts.size() && j < 6; j++) {
            float similarity = ggml_opencog_compute_similarity(atomspace, concepts[i], concepts[j]);
            
            if (similarity > 0.5f) {
                auto* concept1 = ggml_opencog_get_atom(atomspace, concepts[i]);
                auto* concept2 = ggml_opencog_get_atom(atomspace, concepts[j]);
                
                if (concept1 && concept2) {
                    std::cout << "    " << concept1->name << " ~ " << concept2->name
                              << " : similarity = " << std::fixed << std::setprecision(3) << similarity << "\n";
                    count++;
                }
            }
        }
    }
    if (count == 0) {
        std::cout << "    No high similarities found in sample\n";
    }
}

// Agent that performs advanced PLN operations
void advanced_pln_agent(struct ggml_opencog_atomspace* atomspace) {
    std::cout << "AdvancedPLN Agent running...\n";
    
    auto inheritance_links = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_INHERITANCE_LINK);
    
    if (inheritance_links.size() >= 2) {
        // Test abduction: A->C and B->C suggests A->B
        auto* link1 = ggml_opencog_get_atom(atomspace, inheritance_links[0]);
        auto* link2 = ggml_opencog_get_atom(atomspace, inheritance_links[1]);
        
        if (link1 && link2) {
            auto abduced_tv = ggml_opencog_pln_abduction(link1->tv, link2->tv);
            std::cout << "  Abduction test: " << link1->name << " + " << link2->name
                      << " => strength: " << std::fixed << std::setprecision(3) << abduced_tv.strength
                      << ", confidence: " << abduced_tv.confidence << "\n";
            
            // Test revision: combining two truth values
            auto revised_tv = ggml_opencog_pln_revision(link1->tv, link2->tv);
            std::cout << "  Revision test: combining TVs => strength: " 
                      << std::fixed << std::setprecision(3) << revised_tv.strength
                      << ", confidence: " << revised_tv.confidence << "\n";
            
            // Test modus ponens
            struct ggml_opencog_truth_value antecedent = {0.8f, 0.7f};
            auto modus_ponens_tv = ggml_opencog_pln_modus_ponens(link1->tv, antecedent);
            std::cout << "  Modus Ponens test: " << link1->name << " + antecedent => strength: "
                      << std::fixed << std::setprecision(3) << modus_ponens_tv.strength
                      << ", confidence: " << modus_ponens_tv.confidence << "\n";
        }
    }
}

// Agent that manages attention allocation
void attention_allocation_agent(struct ggml_opencog_atomspace* atomspace) {
    std::cout << "AttentionAllocation Agent running...\n";
    
    auto concepts = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_CONCEPT_NODE);
    
    // Allocate attention based on truth value strength
    std::cout << "  Allocating attention based on importance...\n";
    for (uint64_t concept_id : concepts) {
        auto* atom = ggml_opencog_get_atom(atomspace, concept_id);
        if (atom) {
            // Atoms with higher truth value strength get more attention
            float sti_delta = (atom->tv.strength - 0.5f) * 10.0f;
            float lti_delta = atom->tv.confidence * 0.5f;
            
            ggml_opencog_update_attention(atomspace, concept_id, sti_delta, lti_delta);
        }
    }
    
    // Display top atoms by attention
    std::vector<std::pair<uint64_t, float>> atoms_by_sti;
    for (uint64_t concept_id : concepts) {
        auto* atom = ggml_opencog_get_atom(atomspace, concept_id);
        if (atom) {
            atoms_by_sti.push_back({concept_id, atom->sti});
        }
    }
    
    std::sort(atoms_by_sti.begin(), atoms_by_sti.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "  Top 5 atoms by STI (Short-Term Importance):\n";
    for (size_t i = 0; i < std::min(size_t(5), atoms_by_sti.size()); i++) {
        auto* atom = ggml_opencog_get_atom(atomspace, atoms_by_sti[i].first);
        if (atom) {
            std::cout << "    " << atom->name 
                      << " : STI = " << std::fixed << std::setprecision(2) << atom->sti
                      << ", LTI = " << atom->lti << "\n";
        }
    }
}

int main() {
    std::cout << "=== OpenCog Advanced Features Demo ===\n\n";
    
    // Initialize AtomSpace with embeddings
    auto* atomspace = ggml_opencog_atomspace_new(128);
    
    std::cout << "1. Creating Knowledge Base...\n";
    
    struct ggml_opencog_truth_value tv_high = {0.9f, 0.85f};
    struct ggml_opencog_truth_value tv_medium = {0.75f, 0.7f};
    struct ggml_opencog_truth_value tv_low = {0.6f, 0.5f};
    
    // Create a knowledge graph about AI and cognition
    uint64_t ai_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                          "ArtificialIntelligence", tv_high, {});
    uint64_t ml_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                          "MachineLearning", tv_high, {});
    uint64_t cognition_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                                  "Cognition", tv_high, {});
    uint64_t reasoning_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                                  "Reasoning", tv_high, {});
    uint64_t learning_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                                 "Learning", tv_high, {});
    uint64_t knowledge_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                                  "Knowledge", tv_medium, {});
    uint64_t neural_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                               "NeuralNetworks", tv_medium, {});
    uint64_t logic_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                             "Logic", tv_medium, {});
    
    // Create relationships
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "ML->AI", 
                         tv_high, {ml_id, ai_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Reasoning->Cognition", 
                         tv_high, {reasoning_id, cognition_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Learning->Cognition", 
                         tv_high, {learning_id, cognition_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "AI->Cognition", 
                         tv_medium, {ai_id, cognition_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Neural->ML", 
                         tv_high, {neural_id, ml_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Logic->Reasoning", 
                         tv_high, {logic_id, reasoning_id});
    ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK, "Knowledge->Cognition", 
                         tv_medium, {knowledge_id, cognition_id});
    
    std::cout << "   Created 8 concept nodes and 7 inheritance links\n\n";
    
    std::cout << "2. Testing Embedding-Based Similarity...\n";
    
    // Test similarity between concepts
    float sim_ai_ml = ggml_opencog_compute_similarity(atomspace, ai_id, ml_id);
    float sim_reasoning_logic = ggml_opencog_compute_similarity(atomspace, reasoning_id, logic_id);
    float sim_ai_reasoning = ggml_opencog_compute_similarity(atomspace, ai_id, reasoning_id);
    
    std::cout << "   AI ~ MachineLearning: " << std::fixed << std::setprecision(3) << sim_ai_ml << "\n";
    std::cout << "   Reasoning ~ Logic: " << sim_reasoning_logic << "\n";
    std::cout << "   AI ~ Reasoning: " << sim_ai_reasoning << "\n\n";
    
    std::cout << "3. Testing Advanced PLN Operations...\n";
    
    // Get some links for testing
    auto links = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_INHERITANCE_LINK);
    if (links.size() >= 2) {
        auto* link1 = ggml_opencog_get_atom(atomspace, links[0]);
        auto* link2 = ggml_opencog_get_atom(atomspace, links[1]);
        
        if (link1 && link2) {
            // Test abduction
            auto abduced = ggml_opencog_pln_abduction(link1->tv, link2->tv);
            std::cout << "   Abduction(" << link1->name << ", " << link2->name << ")\n";
            std::cout << "     => strength: " << abduced.strength 
                      << ", confidence: " << abduced.confidence << "\n";
            
            // Test revision
            auto revised = ggml_opencog_pln_revision(link1->tv, link2->tv);
            std::cout << "   Revision(" << link1->name << ", " << link2->name << ")\n";
            std::cout << "     => strength: " << revised.strength 
                      << ", confidence: " << revised.confidence << "\n";
            
            // Test modus ponens
            struct ggml_opencog_truth_value antecedent = {0.85f, 0.8f};
            auto consequent = ggml_opencog_pln_modus_ponens(link1->tv, antecedent);
            std::cout << "   ModusPonens(" << link1->name << ", antecedent)\n";
            std::cout << "     => strength: " << consequent.strength 
                      << ", confidence: " << consequent.confidence << "\n\n";
        }
    }
    
    std::cout << "4. Testing Attention Allocation (ECAN)...\n";
    
    // Allocate initial attention
    ggml_opencog_update_attention(atomspace, ai_id, 15.0f, 2.0f);
    ggml_opencog_update_attention(atomspace, cognition_id, 10.0f, 1.5f);
    ggml_opencog_update_attention(atomspace, reasoning_id, 8.0f, 1.0f);
    
    auto* ai_atom = ggml_opencog_get_atom(atomspace, ai_id);
    auto* cog_atom = ggml_opencog_get_atom(atomspace, cognition_id);
    auto* reason_atom = ggml_opencog_get_atom(atomspace, reasoning_id);
    
    std::cout << "   " << ai_atom->name << " : STI = " << ai_atom->sti << ", LTI = " << ai_atom->lti << "\n";
    std::cout << "   " << cog_atom->name << " : STI = " << cog_atom->sti << ", LTI = " << cog_atom->lti << "\n";
    std::cout << "   " << reason_atom->name << " : STI = " << reason_atom->sti << ", LTI = " << reason_atom->lti << "\n";
    
    // Check attention spreading
    auto* ml_atom = ggml_opencog_get_atom(atomspace, ml_id);
    std::cout << "   " << ml_atom->name << " (connected to AI): STI = " << ml_atom->sti << " (spread effect)\n\n";
    
    std::cout << "5. Running CogServer with Advanced Agents...\n";
    
    auto* cogserver = ggml_opencog_cogserver_new(atomspace);
    
    struct ggml_opencog_mind_agent similarity_agent;
    strcpy(similarity_agent.name, "SimilarityReasoner");
    similarity_agent.process = similarity_based_reasoning_agent;
    similarity_agent.frequency = 2;
    similarity_agent.last_run = 0;
    
    struct ggml_opencog_mind_agent pln_agent;
    strcpy(pln_agent.name, "AdvancedPLN");
    pln_agent.process = advanced_pln_agent;
    pln_agent.frequency = 3;
    pln_agent.last_run = 0;
    
    struct ggml_opencog_mind_agent attention_agent;
    strcpy(attention_agent.name, "AttentionAllocator");
    attention_agent.process = attention_allocation_agent;
    attention_agent.frequency = 4;
    attention_agent.last_run = 0;
    
    ggml_opencog_cogserver_add_agent(cogserver, &similarity_agent);
    ggml_opencog_cogserver_add_agent(cogserver, &pln_agent);
    ggml_opencog_cogserver_add_agent(cogserver, &attention_agent);
    
    ggml_opencog_cogserver_start(cogserver);
    
    for (int cycle = 1; cycle <= 6; cycle++) {
        std::cout << "\n--- Cycle " << cycle << " ---\n";
        ggml_opencog_cogserver_run_cycle(cogserver);
    }
    
    ggml_opencog_cogserver_stop(cogserver);
    
    std::cout << "\n6. Final Statistics...\n";
    
    auto final_concepts = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_CONCEPT_NODE);
    auto final_links = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_INHERITANCE_LINK);
    
    std::cout << "   Total Concepts: " << final_concepts.size() << "\n";
    std::cout << "   Total Inheritance Links: " << final_links.size() << "\n";
    
    // Show attention distribution
    std::cout << "\n   Final Attention Distribution:\n";
    std::vector<std::pair<std::string, float>> attention_scores;
    for (uint64_t id : final_concepts) {
        auto* atom = ggml_opencog_get_atom(atomspace, id);
        if (atom) {
            attention_scores.push_back({atom->name, atom->sti});
        }
    }
    std::sort(attention_scores.begin(), attention_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (size_t i = 0; i < std::min(size_t(5), attention_scores.size()); i++) {
        std::cout << "     " << std::setw(25) << std::left << attention_scores[i].first 
                  << " : STI = " << std::fixed << std::setprecision(2) << attention_scores[i].second << "\n";
    }
    
    std::cout << "\n7. Cleanup...\n";
    
    ggml_opencog_cogserver_free(cogserver);
    ggml_opencog_atomspace_free(atomspace);
    
    std::cout << "\n=== Advanced Features Demo Complete ===\n";
    return 0;
}
