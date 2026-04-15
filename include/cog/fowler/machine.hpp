// fowler/machine.hpp — Thomas Fowler's Ternary Calculating Machine (1840)
// Header-only C++11 — Zero external dependencies
//
// A faithful software simulation of the four distinct mechanical parts
// described by Augustus DeMorgan in 1840:
//
//   1. Multiplicand Frame — rods with indices, each in {-1, 0, +1}
//   2. Multiplier Frame   — movable frame with input/output teeth
//   3. Product Frame      — accumulates results (may exceed ±1 range)
//   4. Carrying Apparatus — normalizes product to strict balanced ternary
//
// The simulation models the physical motions: lateral sliding, revolving
// frame rotation, tooth engagement, and carry propagation.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 cogpy

#ifndef COG_FOWLER_MACHINE_HPP
#define COG_FOWLER_MACHINE_HPP

#include "balanced_ternary.hpp"
#include <vector>
#include <string>
#include <functional>
#include <sstream>
#include <iomanip>
#include <cassert>

namespace cog { namespace fowler {

// ─────────────────────────────────────────────────────────────────────────────
// Rod: A single numeral column
// ─────────────────────────────────────────────────────────────────────────────
//
// "Each bearing an index, and each movable backwards and forwards."
// Forward = +1, center = 0, backward = -1.
// The rod can also hold intermediate values (±2, ±3, etc.) on the product
// frame before the carrying apparatus normalizes it.

struct Rod {
    int value;       // Current position: standard range {-1, 0, +1}, but
                     // product rods may temporarily hold larger values
    int column;      // Which numeral column this rod represents (3^column)

    Rod() : value(0), column(0) {}
    Rod(int col, int val = 0) : value(val), column(col) {}

    // Move the rod forward by delta units
    void advance(int delta) { value += delta; }

    // Reset to zero position
    void reset() { value = 0; }

    // Is this rod in the standard balanced ternary range?
    bool is_normalized() const { return value >= -1 && value <= 1; }

    // Get as a Trit (only valid if normalized)
    Trit as_trit() const {
        assert(is_normalized());
        return static_cast<Trit>(value);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Event Log: Records each mechanical motion for step-by-step replay
// ─────────────────────────────────────────────────────────────────────────────

enum class MachineEventType {
    SET_MULTIPLICAND,     // Initial setup of multiplicand frame
    SET_MULTIPLIER,       // Initial setup of multiplier frame
    SLIDE_TO_COLUMN,      // Multiplier frame slides to a new column
    REVOLVE_FRAME,        // Revolving frame engages multiplicand rod
    TOOTH_ACTION,         // Perpendicular teeth act on product rod
    ACCUMULATE_PRODUCT,   // Partial product accumulated
    CARRY_STEP,           // One carry operation on adjacent rods
    CARRY_COMPLETE,       // Carrying apparatus finished
    MULTIPLICATION_DONE,  // Final result ready
    DIVISION_STEP,        // One step of division
    DIVISION_DONE,        // Division complete
};

struct MachineEvent {
    MachineEventType type;
    int column;           // Which column is being operated on
    int value;            // Value involved in the operation
    std::string description;

    MachineEvent(MachineEventType t, int col, int val, const std::string& desc)
        : type(t), column(col), value(val), description(desc) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// MultiplicandFrame: "a collection of rods, not itself connected with any
// machinery, but only useful as indicating the manner in which the frame
// of the multiplier is to act."
// ─────────────────────────────────────────────────────────────────────────────

class MultiplicandFrame {
public:
    std::vector<Rod> rods;

    MultiplicandFrame() {}

    explicit MultiplicandFrame(size_t num_columns) {
        rods.resize(num_columns);
        for (size_t i = 0; i < num_columns; ++i) {
            rods[i] = Rod(static_cast<int>(i));
        }
    }

    // Load a balanced ternary number onto the frame
    void load(const BalancedTernary& bt) {
        size_t n = bt.num_trits();
        rods.resize(n);
        for (size_t i = 0; i < n; ++i) {
            rods[i] = Rod(static_cast<int>(i), static_cast<int>(bt.get_trit(i)));
        }
    }

    // Read the current value as a BalancedTernary
    BalancedTernary read() const {
        std::vector<Trit> trits;
        for (const auto& rod : rods) {
            trits.push_back(static_cast<Trit>(rod.value));
        }
        return BalancedTernary(trits);
    }

    // Get the trit at a specific column
    Trit get_column(size_t col) const {
        if (col >= rods.size()) return TRIT_ZERO;
        return static_cast<Trit>(rods[col].value);
    }

    // Set a specific column to zero (as the revolving frame does)
    void zero_column(size_t col) {
        if (col < rods.size()) rods[col].reset();
    }

    size_t num_columns() const { return rods.size(); }

    // ASCII visualization
    std::string visualize() const {
        std::ostringstream os;
        os << "Multiplicand [";
        for (int i = static_cast<int>(rods.size()) - 1; i >= 0; --i) {
            int v = rods[i].value;
            if (v == 1)       os << "+";
            else if (v == -1) os << "-";
            else              os << "0";
        }
        os << "] = " << read().to_int();
        return os.str();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// MultiplierFrame: "a frame movable in the direction perpendicular to the
// rods of the multiplicand and product, situated between the planes of
// the two."
// ─────────────────────────────────────────────────────────────────────────────
//
// Each rod has two teeth:
//   - Input tooth: rests in the revolving frame (above/on/below axis)
//   - Output tooth: perpendicular, acts on product frame rods
//
// The revolving frame's motion determines the direction of action.
// Rods with teeth ON the axis don't move (multiply by 0).
// Rods with teeth ABOVE the axis move in one direction (+1).
// Rods with teeth BELOW the axis move in the other direction (-1).

class MultiplierFrame {
public:
    struct MultiplierRod {
        Trit input_tooth;   // Position relative to revolving axis: {-1, 0, +1}
        int  column;        // Which column this rod represents

        MultiplierRod() : input_tooth(TRIT_ZERO), column(0) {}
        MultiplierRod(int col, Trit tooth) : input_tooth(tooth), column(col) {}

        // The output tooth action when the revolving frame moves by `direction`
        // If input_tooth is on axis (0), no motion.
        // If input_tooth matches direction, output tooth moves forward (+1).
        // If input_tooth opposes direction, output tooth moves backward (-1).
        int output_action(Trit direction) const {
            if (input_tooth == TRIT_ZERO) return 0;
            // The output is: input_tooth * direction
            return static_cast<int>(input_tooth) * static_cast<int>(direction);
        }
    };

    std::vector<MultiplierRod> rods;
    int current_position;  // Which multiplicand column we're aligned with

    MultiplierFrame() : current_position(0) {}

    explicit MultiplierFrame(size_t num_columns) : current_position(0) {
        rods.resize(num_columns);
        for (size_t i = 0; i < num_columns; ++i) {
            rods[i] = MultiplierRod(static_cast<int>(i), TRIT_ZERO);
        }
    }

    // Load a balanced ternary number as the multiplier
    void load(const BalancedTernary& bt) {
        size_t n = bt.num_trits();
        rods.resize(n);
        for (size_t i = 0; i < n; ++i) {
            rods[i] = MultiplierRod(static_cast<int>(i), bt.get_trit(i));
        }
        current_position = 0;
    }

    // Slide the frame to align with a specific multiplicand column
    void slide_to(int column) {
        current_position = column;
    }

    // Apply the revolving frame motion with the given direction.
    // Returns a vector of (product_column, delta) pairs representing
    // the action of each output tooth on the product frame.
    std::vector<std::pair<int, int>> revolve(Trit direction) const {
        std::vector<std::pair<int, int>> actions;
        for (const auto& rod : rods) {
            int action = rod.output_action(direction);
            if (action != 0) {
                // The product column affected is: multiplier rod column + current position
                int product_col = rod.column + current_position;
                actions.push_back({product_col, action});
            }
        }
        return actions;
    }

    size_t num_columns() const { return rods.size(); }

    // Read the current value as a BalancedTernary
    BalancedTernary read() const {
        std::vector<Trit> trits;
        for (const auto& rod : rods) {
            trits.push_back(rod.input_tooth);
        }
        return BalancedTernary(trits);
    }

    // ASCII visualization
    std::string visualize() const {
        std::ostringstream os;
        os << "Multiplier   [";
        for (int i = static_cast<int>(rods.size()) - 1; i >= 0; --i) {
            Trit t = rods[i].input_tooth;
            if (t == TRIT_POS)      os << "+";
            else if (t == TRIT_NEG) os << "-";
            else                    os << "0";
        }
        os << "] = " << read().to_int() << "  (pos=" << current_position << ")";
        return os.str();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// ProductFrame: "precisely resembles the frame of the multiplicand, with
// the addition of the connecting part by which the multiplier acts upon it."
// ─────────────────────────────────────────────────────────────────────────────
//
// Key difference from MultiplicandFrame: product rods can hold values
// outside {-1, 0, +1} during intermediate computation. The values are
// only normalized to strict balanced ternary after the carrying apparatus
// has been applied.

class ProductFrame {
public:
    std::vector<Rod> rods;

    ProductFrame() {}

    explicit ProductFrame(size_t num_columns) {
        rods.resize(num_columns);
        for (size_t i = 0; i < num_columns; ++i) {
            rods[i] = Rod(static_cast<int>(i));
        }
    }

    // Ensure we have enough columns
    void ensure_columns(size_t n) {
        while (rods.size() < n) {
            rods.push_back(Rod(static_cast<int>(rods.size())));
        }
    }

    // Apply an action from the multiplier's output tooth
    void apply_action(int column, int delta) {
        ensure_columns(column + 1);
        rods[column].advance(delta);
    }

    // Apply a batch of actions (from one revolve operation)
    void apply_actions(const std::vector<std::pair<int, int>>& actions) {
        for (const auto& action : actions) {
            apply_action(action.first, action.second);
        }
    }

    // Check if all rods are in normalized range
    bool is_normalized() const {
        for (const auto& rod : rods) {
            if (!rod.is_normalized()) return false;
        }
        return true;
    }

    // Read the raw (possibly unnormalized) values
    std::vector<int> raw_values() const {
        std::vector<int> vals;
        for (const auto& rod : rods) {
            vals.push_back(rod.value);
        }
        return vals;
    }

    // Read as BalancedTernary (only valid if normalized)
    BalancedTernary read() const {
        std::vector<Trit> trits;
        for (const auto& rod : rods) {
            trits.push_back(static_cast<Trit>(rod.value));
        }
        return BalancedTernary(trits);
    }

    // Read as int64_t (works even if not normalized)
    int64_t to_int() const {
        int64_t result = 0;
        int64_t power = 1;
        for (const auto& rod : rods) {
            result += static_cast<int64_t>(rod.value) * power;
            power *= 3;
        }
        return result;
    }

    // Reset all rods to zero
    void clear() {
        for (auto& rod : rods) rod.reset();
    }

    size_t num_columns() const { return rods.size(); }

    // ASCII visualization
    std::string visualize() const {
        std::ostringstream os;
        os << "Product      [";
        for (int i = static_cast<int>(rods.size()) - 1; i >= 0; --i) {
            int v = rods[i].value;
            if (v >= -1 && v <= 1) {
                if (v == 1)       os << "+";
                else if (v == -1) os << "-";
                else              os << "0";
            } else {
                // Show numeric value for unnormalized rods
                os << "(" << v << ")";
            }
        }
        os << "] = " << to_int();
        if (!is_normalized()) os << " (unnormalized)";
        return os.str();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CarryApparatus: "a simple apparatus which, like the multiplier, has a
// lateral motion, and can be brought on any pair of consecutive rods."
// ─────────────────────────────────────────────────────────────────────────────
//
// "By one motion of the hand, it advances the left of two rods by a unit,
// and throws back the right hand rod by 3 units, or vice versa."
//
// This normalizes the product frame to strict balanced ternary.
// For any rod with value > 1:  subtract 3, carry +1 to next column
// For any rod with value < -1: add 3, carry -1 to next column

class CarryApparatus {
public:
    // Perform one carry operation on a pair of consecutive rods.
    // Returns true if a carry was performed.
    static bool carry_step(ProductFrame& product, size_t right_col,
                           std::vector<MachineEvent>& log) {
        if (right_col >= product.rods.size()) return false;

        int val = product.rods[right_col].value;
        if (val >= -1 && val <= 1) return false;  // Already normalized

        product.ensure_columns(right_col + 2);

        if (val > 1) {
            // "advances the left of two rods by a unit, and throws back
            //  the right hand rod by 3 units"
            product.rods[right_col].value -= 3;
            product.rods[right_col + 1].value += 1;

            std::ostringstream desc;
            desc << "Carry at col " << right_col << ": -3 from col "
                 << right_col << ", +1 to col " << (right_col + 1);
            log.push_back(MachineEvent(MachineEventType::CARRY_STEP,
                                       right_col, val, desc.str()));
            return true;
        }

        if (val < -1) {
            // Vice versa: advance right rod by 3, throw back left by 1
            product.rods[right_col].value += 3;
            product.rods[right_col + 1].value -= 1;

            std::ostringstream desc;
            desc << "Carry at col " << right_col << ": +3 to col "
                 << right_col << ", -1 from col " << (right_col + 1);
            log.push_back(MachineEvent(MachineEventType::CARRY_STEP,
                                       right_col, val, desc.str()));
            return true;
        }

        return false;
    }

    // Normalize the entire product frame using repeated carry passes
    static void normalize(ProductFrame& product,
                          std::vector<MachineEvent>& log) {
        bool changed = true;
        int max_passes = 100;  // Safety limit

        while (changed && max_passes-- > 0) {
            changed = false;
            for (size_t i = 0; i < product.rods.size(); ++i) {
                if (carry_step(product, i, log)) {
                    changed = true;
                }
            }
        }

        log.push_back(MachineEvent(MachineEventType::CARRY_COMPLETE,
                                   -1, 0, "Carrying apparatus finished"));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// FowlerMachine: The complete calculating machine
// ─────────────────────────────────────────────────────────────────────────────
//
// Orchestrates all four parts to perform multiplication and division,
// recording every mechanical motion for step-by-step replay.

class FowlerMachine {
public:
    MultiplicandFrame multiplicand;
    MultiplierFrame   multiplier;
    ProductFrame      product;
    std::vector<MachineEvent> event_log;

    // Callback for step-by-step observation
    using StepCallback = std::function<void(const FowlerMachine&, const MachineEvent&)>;
    StepCallback on_step;

    FowlerMachine() : on_step(nullptr) {}

    // ── Setup ───────────────────────────────────────────────────────────

    void set_multiplicand(const BalancedTernary& bt) {
        multiplicand.load(bt);
        log_event(MachineEventType::SET_MULTIPLICAND, -1, bt.to_int(),
                  "Set multiplicand = " + bt.to_string() +
                  " (" + std::to_string(bt.to_int()) + ")");
    }

    void set_multiplier(const BalancedTernary& bt) {
        multiplier.load(bt);
        log_event(MachineEventType::SET_MULTIPLIER, -1, bt.to_int(),
                  "Set multiplier = " + bt.to_string() +
                  " (" + std::to_string(bt.to_int()) + ")");
    }

    // ── Multiplication ──────────────────────────────────────────────────
    //
    // "The process of multiplication is then as follows..."
    // For each column of the multiplicand:
    //   1. Slide multiplier frame to that column
    //   2. Read the multiplicand digit
    //   3. Revolve the frame to bring that rod to zero
    //   4. The perpendicular teeth act on the product frame
    //   5. Move to next column
    // Then apply the carrying apparatus.

    BalancedTernary multiply(const BalancedTernary& a, const BalancedTernary& b) {
        event_log.clear();

        set_multiplicand(a);
        set_multiplier(b);

        // Prepare product frame with enough columns
        size_t product_cols = a.num_trits() + b.num_trits();
        product = ProductFrame(product_cols);

        // Process each column of the multiplicand
        for (size_t col = 0; col < multiplicand.num_columns(); ++col) {
            Trit mcand_digit = multiplicand.get_column(col);

            // Step 1: Slide multiplier to this column
            multiplier.slide_to(static_cast<int>(col));
            log_event(MachineEventType::SLIDE_TO_COLUMN, col, 0,
                      "Slide multiplier to column " + std::to_string(col));

            if (mcand_digit == TRIT_ZERO) {
                // No action needed for zero digit
                continue;
            }

            // Step 2: Determine revolving direction
            // "The rule is, to move the revolving frame in such a way as to
            //  bring the rod of the multiplicand to its zero position"
            // If digit is +1, revolve in +1 direction (which zeros it)
            // If digit is -1, revolve in -1 direction
            Trit revolve_direction = mcand_digit;

            log_event(MachineEventType::REVOLVE_FRAME, col,
                      static_cast<int>(revolve_direction),
                      "Revolve frame (direction=" +
                      std::string(trit_to_symbol(revolve_direction)) +
                      ") to zero multiplicand col " + std::to_string(col));

            // Step 3: The perpendicular teeth act on the product frame
            auto actions = multiplier.revolve(revolve_direction);
            for (const auto& action : actions) {
                log_event(MachineEventType::TOOTH_ACTION, action.first,
                          action.second,
                          "Tooth acts on product col " +
                          std::to_string(action.first) + " by " +
                          std::to_string(action.second));
            }

            product.apply_actions(actions);

            // Step 4: Zero the multiplicand rod
            multiplicand.zero_column(col);

            log_event(MachineEventType::ACCUMULATE_PRODUCT, col, 0,
                      "Partial product accumulated: " + product.visualize());
        }

        // Step 5: Apply carrying apparatus to normalize
        CarryApparatus::normalize(product, event_log);

        // Read final result
        BalancedTernary result = product.read();
        log_event(MachineEventType::MULTIPLICATION_DONE, -1,
                  result.to_int(),
                  "Multiplication complete: " + result.to_string() +
                  " (" + std::to_string(result.to_int()) + ")");

        return result;
    }

    // ── Division ────────────────────────────────────────────────────────
    //
    // "The method of performing division is precisely the reverse of the
    //  preceding."
    // Product frame = dividend, multiplier frame = divisor,
    // multiplicand frame = quotient.
    //
    // For each column of the quotient (from MSB to LSB):
    //   1. Slide divisor to that column
    //   2. Try each trit value {+1, 0, -1} for the quotient digit
    //   3. Choose the one that brings the remainder closest to zero
    //   4. Apply the corresponding subtraction to the product frame

    BalancedTernary divide(const BalancedTernary& dividend,
                           const BalancedTernary& divisor) {
        event_log.clear();

        if (divisor.is_zero()) {
            throw std::domain_error("Division by zero");
        }

        // Load dividend into product frame
        product = ProductFrame(dividend.num_trits() + 2);
        for (size_t i = 0; i < dividend.num_trits(); ++i) {
            product.rods[i].value = static_cast<int>(dividend.get_trit(i));
        }

        // Load divisor into multiplier frame
        set_multiplier(divisor);

        // Determine quotient size
        size_t q_size = dividend.num_trits();
        multiplicand = MultiplicandFrame(q_size);

        log_event(MachineEventType::SET_MULTIPLICAND, -1,
                  dividend.to_int(),
                  "Division: dividend = " + dividend.to_string() +
                  " (" + std::to_string(dividend.to_int()) + ")");

        // Process from MSB to LSB
        // Division is the REVERSE of multiplication. In multiplication,
        // revolving with direction d ADDS (multiplier * d) to the product.
        // In division, we want to SUBTRACT (divisor * q_digit) from the
        // dividend. So the revolve direction is the NEGATION of the
        // quotient digit.
        for (int col = static_cast<int>(q_size) - 1; col >= 0; --col) {
            multiplier.slide_to(col);

            // Try each trit value and pick the best
            int64_t best_abs_rem = std::abs(product.to_int());
            Trit best_trit = TRIT_ZERO;

            for (Trit trial : {TRIT_POS, TRIT_NEG}) {
                // Revolve with NEGATED direction to subtract
                Trit neg_trial = negate_trit(trial);
                auto actions = multiplier.revolve(neg_trial);
                ProductFrame trial_product = product;
                trial_product.apply_actions(actions);
                int64_t trial_rem = std::abs(trial_product.to_int());

                if (trial_rem < best_abs_rem) {
                    best_abs_rem = trial_rem;
                    best_trit = trial;
                }
            }

            // Apply the best choice (revolve with negated direction)
            if (best_trit != TRIT_ZERO) {
                Trit neg_best = negate_trit(best_trit);
                auto actions = multiplier.revolve(neg_best);
                product.apply_actions(actions);
            }

            multiplicand.rods[col].value = static_cast<int>(best_trit);

            log_event(MachineEventType::DIVISION_STEP, col,
                      static_cast<int>(best_trit),
                      "Division: quotient[" + std::to_string(col) + "] = " +
                      std::string(trit_to_symbol(best_trit)) +
                      ", remainder = " + std::to_string(product.to_int()));
        }

        BalancedTernary quotient = multiplicand.read();
        log_event(MachineEventType::DIVISION_DONE, -1,
                  quotient.to_int(),
                  "Division complete: quotient = " + quotient.to_string() +
                  " (" + std::to_string(quotient.to_int()) + ")" +
                  ", remainder = " + std::to_string(product.to_int()));

        return quotient;
    }

    // ── State Visualization ─────────────────────────────────────────────

    std::string visualize() const {
        std::ostringstream os;
        os << "╔══════════════════════════════════════════════════════╗\n";
        os << "║  Thomas Fowler's Ternary Calculating Machine (1840) ║\n";
        os << "╠══════════════════════════════════════════════════════╣\n";
        os << "║  " << std::left << std::setw(52) << multiplicand.visualize() << "║\n";
        os << "║  " << std::left << std::setw(52) << multiplier.visualize()   << "║\n";
        os << "║  " << std::left << std::setw(52) << product.visualize()      << "║\n";
        os << "╚══════════════════════════════════════════════════════╝";
        return os.str();
    }

    // ── Event Log Access ────────────────────────────────────────────────

    const std::vector<MachineEvent>& get_log() const { return event_log; }

    std::string format_log() const {
        std::ostringstream os;
        for (size_t i = 0; i < event_log.size(); ++i) {
            os << "[" << std::setw(3) << i << "] " << event_log[i].description << "\n";
        }
        return os.str();
    }

private:
    void log_event(MachineEventType type, int col, int val,
                   const std::string& desc) {
        MachineEvent evt(type, col, val, desc);
        event_log.push_back(evt);
        if (on_step) on_step(*this, evt);
    }
};

}  // namespace fowler
}  // namespace cog

#endif  // COG_FOWLER_MACHINE_HPP
