#include "ggml-opencog.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <ctime>

// Demonstrate temporal reasoning and event sequence understanding

void print_event(struct ggml_opencog_atomspace* atomspace, uint64_t event_id) {
    auto* event = ggml_opencog_get_atom(atomspace, event_id);
    if (!event || !event->time_interval) return;
    
    std::cout << "  " << std::setw(20) << std::left << event->name 
              << " [" << event->time_interval->start_time 
              << " - " << event->time_interval->end_time << "]";
    
    if (event->time_interval->is_point) {
        std::cout << " (point event)";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== OpenCog Temporal Reasoning Demo ===\n\n";
    
    // Create AtomSpace
    auto* atomspace = ggml_opencog_atomspace_new(64);
    struct ggml_opencog_truth_value tv = {0.9f, 0.8f};
    
    // === Example 1: Daily routine ===
    std::cout << "Example 1: Modeling a Daily Routine\n";
    std::cout << "=====================================\n\n";
    
    // Create events for a typical day (using arbitrary time units for simplicity)
    // Let's say each hour = 1000 time units
    uint64_t wake_up_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "WakeUp", tv, {});
    uint64_t exercise_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Exercise", tv, {});
    uint64_t breakfast_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Breakfast", tv, {});
    uint64_t work_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Work", tv, {});
    uint64_t lunch_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Lunch", tv, {});
    uint64_t meeting_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Meeting", tv, {});
    uint64_t dinner_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Dinner", tv, {});
    uint64_t sleep_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Sleep", tv, {});
    
    // Set time intervals (7am = 7000, 8am = 8000, etc.)
    ggml_opencog_set_time_interval(atomspace, wake_up_id, 7000, 7500);      // 7:00-7:30
    ggml_opencog_set_time_interval(atomspace, exercise_id, 7500, 8000);     // 7:30-8:00
    ggml_opencog_set_time_interval(atomspace, breakfast_id, 8000, 8500);    // 8:00-8:30
    ggml_opencog_set_time_interval(atomspace, work_id, 9000, 17000);        // 9:00-17:00
    ggml_opencog_set_time_interval(atomspace, lunch_id, 12000, 13000);      // 12:00-13:00
    ggml_opencog_set_time_interval(atomspace, meeting_id, 14000, 15000);    // 14:00-15:00
    ggml_opencog_set_time_interval(atomspace, dinner_id, 18000, 19000);     // 18:00-19:00
    // Note: Sleep spans into next day. 31000 represents 7:00am next day (24h = 24000, so 31000 = 7:00am)
    ggml_opencog_set_time_interval(atomspace, sleep_id, 22000, 31000);      // 22:00-7:00 (next day)
    
    std::cout << "Daily events:\n";
    print_event(atomspace, wake_up_id);
    print_event(atomspace, exercise_id);
    print_event(atomspace, breakfast_id);
    print_event(atomspace, work_id);
    print_event(atomspace, lunch_id);
    print_event(atomspace, meeting_id);
    print_event(atomspace, dinner_id);
    print_event(atomspace, sleep_id);
    
    // === Example 2: Temporal queries ===
    std::cout << "\n\nExample 2: Temporal Queries\n";
    std::cout << "============================\n\n";
    
    std::cout << "Query 1: What happens at 10:00 (10000)?\n";
    auto events_at_10 = ggml_opencog_get_atoms_at_time(atomspace, 10000);
    std::cout << "  Found " << events_at_10.size() << " event(s):\n";
    for (uint64_t id : events_at_10) {
        print_event(atomspace, id);
    }
    
    std::cout << "\nQuery 2: What happens between 7:00 and 9:00 (7000-9000)?\n";
    auto morning_events = ggml_opencog_get_atoms_in_interval(atomspace, 7000, 9000);
    std::cout << "  Found " << morning_events.size() << " event(s):\n";
    for (uint64_t id : morning_events) {
        print_event(atomspace, id);
    }
    
    std::cout << "\nQuery 3: What happens during work hours?\n";
    auto work_events = ggml_opencog_get_atoms_in_interval(atomspace, 9000, 17000);
    std::cout << "  Found " << work_events.size() << " event(s):\n";
    for (uint64_t id : work_events) {
        print_event(atomspace, id);
    }
    
    // === Example 3: Temporal relationships ===
    std::cout << "\n\nExample 3: Temporal Relationships\n";
    std::cout << "==================================\n\n";
    
    std::cout << "Checking temporal ordering:\n";
    
    if (ggml_opencog_happens_before(atomspace, wake_up_id, breakfast_id)) {
        std::cout << "  ✓ WakeUp happens before Breakfast\n";
    }
    
    if (ggml_opencog_happens_before(atomspace, breakfast_id, work_id)) {
        std::cout << "  ✓ Breakfast happens before Work\n";
    }
    
    if (ggml_opencog_happens_before(atomspace, work_id, dinner_id)) {
        std::cout << "  ✓ Work happens before Dinner\n";
    }
    
    if (!ggml_opencog_happens_before(atomspace, dinner_id, breakfast_id)) {
        std::cout << "  ✓ Dinner does NOT happen before Breakfast (correct)\n";
    }
    
    std::cout << "\nChecking containment:\n";
    
    if (ggml_opencog_happens_during(atomspace, lunch_id, work_id)) {
        std::cout << "  ✓ Lunch happens during Work\n";
    }
    
    if (ggml_opencog_happens_during(atomspace, meeting_id, work_id)) {
        std::cout << "  ✓ Meeting happens during Work\n";
    }
    
    if (!ggml_opencog_happens_during(atomspace, breakfast_id, work_id)) {
        std::cout << "  ✓ Breakfast does NOT happen during Work (correct)\n";
    }
    
    // === Example 4: Event sequences and causal chains ===
    std::cout << "\n\nExample 4: Event Sequences and Causal Chains\n";
    std::cout << "=============================================\n\n";
    
    std::cout << "Modeling a project workflow:\n\n";
    
    // Create project events
    uint64_t planning_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Planning", tv, {});
    uint64_t design_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Design", tv, {});
    uint64_t implementation_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Implementation", tv, {});
    uint64_t testing_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Testing", tv, {});
    uint64_t deployment_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_CONCEPT_NODE, "Deployment", tv, {});
    
    // Set sequential time intervals
    ggml_opencog_set_time_interval(atomspace, planning_id, 1000, 2000);
    ggml_opencog_set_time_interval(atomspace, design_id, 2000, 4000);
    ggml_opencog_set_time_interval(atomspace, implementation_id, 4000, 10000);
    ggml_opencog_set_time_interval(atomspace, testing_id, 10000, 12000);
    ggml_opencog_set_time_interval(atomspace, deployment_id, 12000, 12500);
    
    std::cout << "Project phases:\n";
    print_event(atomspace, planning_id);
    print_event(atomspace, design_id);
    print_event(atomspace, implementation_id);
    print_event(atomspace, testing_id);
    print_event(atomspace, deployment_id);
    
    // Create sequential links with truth values
    struct ggml_opencog_truth_value tv_high = {0.95f, 0.9f};
    struct ggml_opencog_truth_value tv_med = {0.85f, 0.8f};
    
    uint64_t seq1 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_SEQUENTIAL_LINK,
                                         "Planning->Design", tv_high, {planning_id, design_id});
    
    uint64_t seq2 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_SEQUENTIAL_LINK,
                                         "Design->Implementation", tv_high, {design_id, implementation_id});
    
    uint64_t seq3 = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_SEQUENTIAL_LINK,
                                         "Implementation->Testing", tv_med, {implementation_id, testing_id});
    
    std::cout << "\n\nTemporal Induction (Chain of sequences):\n";
    std::cout << "  If Planning precedes Design, and Design precedes Implementation,\n";
    std::cout << "  then Planning precedes Implementation (with reduced confidence)\n\n";
    
    auto* link1 = ggml_opencog_get_atom(atomspace, seq1);
    auto* link2 = ggml_opencog_get_atom(atomspace, seq2);
    
    auto inferred_tv = ggml_opencog_temporal_induction(link1->tv, link2->tv);
    
    std::cout << "  Planning->Design:         strength=" << std::fixed << std::setprecision(3) 
              << link1->tv.strength << ", confidence=" << link1->tv.confidence << "\n";
    std::cout << "  Design->Implementation:   strength=" << link2->tv.strength 
              << ", confidence=" << link2->tv.confidence << "\n";
    std::cout << "  Inferred Planning->Impl:  strength=" << inferred_tv.strength 
              << ", confidence=" << inferred_tv.confidence << "\n";
    std::cout << "\n  Note: Confidence decreases with chain length (uncertainty propagation)\n";
    
    // === Example 5: Point events and simultaneity ===
    std::cout << "\n\nExample 5: Point Events and Simultaneity\n";
    std::cout << "=========================================\n\n";
    
    // Create point events (events at specific moments)
    uint64_t alarm_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_TIME_NODE, "AlarmRings", tv, {});
    uint64_t wake_moment_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_TIME_NODE, "WakeMoment", tv, {});
    uint64_t snooze_id = ggml_opencog_add_atom(atomspace, GGML_OPENCOG_TIME_NODE, "SnoozePress", tv, {});
    
    // Set point events (same start and end time)
    ggml_opencog_set_time_interval(atomspace, alarm_id, 7000, 7000);
    ggml_opencog_set_time_interval(atomspace, wake_moment_id, 7005, 7005);  // 5 units later
    ggml_opencog_set_time_interval(atomspace, snooze_id, 7003, 7003);       // 3 units later
    
    std::cout << "Point events:\n";
    print_event(atomspace, alarm_id);
    print_event(atomspace, snooze_id);
    print_event(atomspace, wake_moment_id);
    
    std::cout << "\nChecking simultaneity (tolerance = 10 time units):\n";
    
    if (ggml_opencog_happens_simultaneously(atomspace, alarm_id, wake_moment_id, 10)) {
        std::cout << "  ✓ AlarmRings and WakeMoment are simultaneous (within tolerance)\n";
    }
    
    if (!ggml_opencog_happens_simultaneously(atomspace, alarm_id, wake_moment_id, 3)) {
        std::cout << "  ✓ AlarmRings and WakeMoment are NOT simultaneous (tolerance too small)\n";
    }
    
    // === Example 6: Event patterns and recognition ===
    std::cout << "\n\nExample 6: Detecting Event Patterns\n";
    std::cout << "====================================\n\n";
    
    std::cout << "Looking for 'morning routine' pattern:\n";
    std::cout << "  (Events between 7:00 and 9:00 that happen in sequence)\n\n";
    
    auto morning_pattern = ggml_opencog_get_atoms_in_interval(atomspace, 7000, 9000);
    
    // Sort by start time (simple bubble sort for demo)
    for (size_t i = 0; i < morning_pattern.size(); i++) {
        for (size_t j = i + 1; j < morning_pattern.size(); j++) {
            auto* atom_i = ggml_opencog_get_atom(atomspace, morning_pattern[i]);
            auto* atom_j = ggml_opencog_get_atom(atomspace, morning_pattern[j]);
            
            if (atom_i->time_interval && atom_j->time_interval &&
                atom_i->time_interval->start_time > atom_j->time_interval->start_time) {
                std::swap(morning_pattern[i], morning_pattern[j]);
            }
        }
    }
    
    std::cout << "  Morning routine sequence:\n";
    for (size_t i = 0; i < morning_pattern.size(); i++) {
        auto* atom = ggml_opencog_get_atom(atomspace, morning_pattern[i]);
        if (atom && atom->time_interval) {
            std::cout << "    " << (i + 1) << ". " << atom->name << "\n";
        }
    }
    
    // Clean up
    ggml_opencog_atomspace_free(atomspace);
    
    std::cout << "\n\n=== Demo Complete ===\n";
    std::cout << "\nKey Takeaways:\n";
    std::cout << "1. Time intervals enable reasoning about event durations and overlaps\n";
    std::cout << "2. Temporal ordering (before/during/simultaneous) supports causal reasoning\n";
    std::cout << "3. Temporal queries allow finding events by time\n";
    std::cout << "4. Temporal induction propagates sequential relationships through chains\n";
    std::cout << "5. Point events and simultaneity detection handle instantaneous occurrences\n";
    std::cout << "6. Event patterns can be recognized through temporal structure\n";
    
    return 0;
}
