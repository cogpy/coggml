#include "ggml-opencog.h"
#include <iostream>
#include <vector>
#include <cstring>

// Example MindAgent that processes concepts
void concept_processor_agent(struct ggml_opencog_atomspace* atomspace) {
    std::cout << "ConceptProcessor Agent running...\n";
    
    // Get all concept nodes
    auto concepts = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_CONCEPT_NODE);
    std::cout << "Found " << concepts.size() << " concept nodes\n";
    
    for (uint64_t concept_id : concepts) {
        auto* atom = ggml_opencog_get_atom(atomspace, concept_id);
        if (atom) {
            std::cout << "  Concept: " << atom->name 
                      << " (strength: " << atom->tv.strength 
                      << ", confidence: " << atom->tv.confidence << ")\n";
        }
    }
}

// Example MindAgent that processes relationships  
void relationship_processor_agent(struct ggml_opencog_atomspace* atomspace) {
    std::cout << "RelationshipProcessor Agent running...\n";
    
    // Get all evaluation links
    auto links = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_EVALUATION_LINK);
    std::cout << "Found " << links.size() << " evaluation links\n";
    
    for (uint64_t link_id : links) {
        auto* atom = ggml_opencog_get_atom(atomspace, link_id);
        if (atom) {
            std::cout << "  Link: " << atom->name 
                      << " connects " << atom->outgoing.size() << " atoms\n";
        }
    }
}

int main() {
    std::cout << "=== OpenCog GGML Demo ===\n\n";
    
    // Initialize AtomSpace with 64-dimensional embeddings
    auto* atomspace = ggml_opencog_atomspace_new(64);
    
    std::cout << "1. Creating atoms...\n";
    
    // Create some concept nodes
    struct ggml_opencog_truth_value tv_high = {0.9f, 0.8f};
    struct ggml_opencog_truth_value tv_medium = {0.7f, 0.6f};
    
    uint64_t human_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                             "Human", tv_high, {});
    uint64_t animal_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                              "Animal", tv_high, {});
    uint64_t socrates_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                                 "Socrates", tv_high, {});
    uint64_t mortal_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, 
                                              "Mortal", tv_high, {});
    
    // Create predicate nodes
    uint64_t isa_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_PREDICATE_NODE, 
                                           "IsA", tv_high, {});
    
    // Create inheritance links
    uint64_t human_animal_link = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                                      "Human->Animal", tv_medium, 
                                                      {human_id, animal_id});
    
    uint64_t socrates_human_link = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                                         "Socrates->Human", tv_high,
                                                         {socrates_id, human_id});
    
    uint64_t animal_mortal_link = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_INHERITANCE_LINK,
                                                        "Animal->Mortal", tv_high,
                                                        {animal_id, mortal_id});
    
    // Create evaluation link
    uint64_t socrates_isa_human = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_EVALUATION_LINK,
                                                        "Socrates IsA Human", tv_high,
                                                        {isa_id, socrates_id, human_id});
    
    std::cout << "Created atoms with IDs: " << human_id << ", " << animal_id << ", " 
              << socrates_id << ", " << mortal_id << "\n\n";
    
    std::cout << "2. Querying atoms...\n";
    auto humans = ggml_opencog_get_atoms_by_name(atomspace, "Human");
    std::cout << "Found " << humans.size() << " atoms named 'Human'\n";
    
    auto concepts = ggml_opencog_get_atoms_by_type(atomspace, GGML_OPENCOG_CONCEPT_NODE);
    std::cout << "Found " << concepts.size() << " concept nodes\n\n";
    
    std::cout << "3. Testing reasoning...\n";
    
    // Test PLN deduction
    auto* socrates_human = ggml_opencog_get_atom(atomspace, socrates_human_link);
    auto* human_animal = ggml_opencog_get_atom(atomspace, human_animal_link);
    
    if (socrates_human && human_animal) {
        auto deduced_tv = ggml_opencog_pln_deduction(socrates_human->tv, human_animal->tv);
        std::cout << "PLN Deduction: Socrates->Human + Human->Animal = "
                  << "strength: " << deduced_tv.strength 
                  << ", confidence: " << deduced_tv.confidence << "\n";
    }
    
    // Test PLN induction
    auto* animal_mortal = ggml_opencog_get_atom(atomspace, animal_mortal_link);
    if (human_animal && animal_mortal) {
        auto induced_tv = ggml_opencog_pln_induction(human_animal->tv, animal_mortal->tv);
        std::cout << "PLN Induction: Human->Animal + Animal->Mortal = "
                  << "strength: " << induced_tv.strength 
                  << ", confidence: " << induced_tv.confidence << "\n\n";
    }
    
    std::cout << "4. CogServer with MindAgents...\n";
    
    // Create CogServer
    auto* cogserver = ggml_opencog_cogserver_new(atomspace);
    
    // Create mind agents
    struct ggml_opencog_mind_agent concept_agent;
    strcpy(concept_agent.name, "ConceptProcessor");
    concept_agent.process = concept_processor_agent;
    concept_agent.frequency = 2; // Run every 2 cycles
    concept_agent.last_run = 0;
    
    struct ggml_opencog_mind_agent relationship_agent;
    strcpy(relationship_agent.name, "RelationshipProcessor");
    relationship_agent.process = relationship_processor_agent;
    relationship_agent.frequency = 3; // Run every 3 cycles
    relationship_agent.last_run = 0;
    
    // Add agents to CogServer
    ggml_opencog_cogserver_add_agent(cogserver, &concept_agent);
    ggml_opencog_cogserver_add_agent(cogserver, &relationship_agent);
    
    std::cout << "5. Running cognitive cycles...\n";
    
    // Run a few cognitive cycles
    ggml_opencog_cogserver_start(cogserver);
    
    for (int cycle = 1; cycle <= 6; cycle++) {
        std::cout << "\n--- Cycle " << cycle << " ---\n";
        ggml_opencog_cogserver_run_cycle(cogserver);
    }
    
    ggml_opencog_cogserver_stop(cogserver);
    
    std::cout << "\n6. Cleanup...\n";
    
    // Cleanup
    ggml_opencog_cogserver_free(cogserver);
    ggml_opencog_atomspace_free(atomspace);
    
    std::cout << "\n=== Demo Complete ===\n";
    return 0;
}
