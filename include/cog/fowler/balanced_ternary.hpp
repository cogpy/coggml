// fowler/balanced_ternary.hpp — Balanced Ternary Arithmetic for Fowler's Machine
// Header-only C++11 — Zero external dependencies
//
// Balanced ternary uses digits {-1, 0, +1} in base 3.
// Thomas Fowler (1840) used this system for his wooden calculating machine.
// Each "trit" position i represents value * 3^i.
//
// Notation: T = -1, 0 = 0, 1 = +1
// Example: 1T0 = 1*9 + (-1)*3 + 0*1 = 6
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 cogpy

#ifndef COG_FOWLER_BALANCED_TERNARY_HPP
#define COG_FOWLER_BALANCED_TERNARY_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <ostream>

namespace cog { namespace fowler {

// ─────────────────────────────────────────────────────────────────────────────
// Trit: a single balanced ternary digit {-1, 0, +1}
// ─────────────────────────────────────────────────────────────────────────────

enum Trit : int8_t {
    TRIT_NEG  = -1,  // Fowler: rod moved backward
    TRIT_ZERO =  0,  // Fowler: rod in center (zero) position
    TRIT_POS  = +1,  // Fowler: rod moved forward
};

inline const char* trit_to_symbol(Trit t) {
    switch (t) {
        case TRIT_NEG:  return "T";  // T for "minus one" (Knuth convention)
        case TRIT_ZERO: return "0";
        case TRIT_POS:  return "1";
    }
    return "?";
}

inline char trit_to_char(Trit t) {
    switch (t) {
        case TRIT_NEG:  return 'T';
        case TRIT_ZERO: return '0';
        case TRIT_POS:  return '1';
    }
    return '?';
}

inline Trit char_to_trit(char c) {
    switch (c) {
        case 'T': case 't': case '-': return TRIT_NEG;
        case '0':                      return TRIT_ZERO;
        case '1': case '+':            return TRIT_POS;
    }
    throw std::invalid_argument(std::string("Invalid trit character: ") + c);
}

inline Trit negate_trit(Trit t) {
    return static_cast<Trit>(-static_cast<int8_t>(t));
}

// ─────────────────────────────────────────────────────────────────────────────
// BalancedTernary: arbitrary-precision balanced ternary integer
// ─────────────────────────────────────────────────────────────────────────────
//
// Internal representation: trits[0] = least significant trit (3^0),
//                          trits[n-1] = most significant trit (3^(n-1))
// The value is: sum_{i=0}^{n-1} trits[i] * 3^i
//
// Invariant: no trailing zeros (trits.back() != 0 unless empty).
// Empty trits vector represents zero.

class BalancedTernary {
public:
    std::vector<Trit> trits;  // LSB first

    // ── Constructors ────────────────────────────────────────────────────

    BalancedTernary() {}

    explicit BalancedTernary(int64_t value) {
        from_int(value);
    }

    explicit BalancedTernary(const std::string& s) {
        from_string(s);
    }

    explicit BalancedTernary(const std::vector<Trit>& t) : trits(t) {
        normalize();
    }

    // ── Conversion from decimal ─────────────────────────────────────────

    void from_int(int64_t value) {
        trits.clear();
        if (value == 0) return;

        bool negative = (value < 0);
        if (negative) value = -value;

        while (value != 0) {
            int rem = static_cast<int>(value % 3);
            value /= 3;
            if (rem == 0) {
                trits.push_back(TRIT_ZERO);
            } else if (rem == 1) {
                trits.push_back(TRIT_POS);
            } else {  // rem == 2 → this is -1 with carry
                trits.push_back(TRIT_NEG);
                value += 1;  // carry
            }
        }

        if (negative) negate();
        normalize();
    }

    // ── Conversion from string (MSB first, e.g., "1T01") ───────────────

    void from_string(const std::string& s) {
        trits.clear();
        if (s.empty() || s == "0") return;

        // Parse MSB-first string into LSB-first vector
        for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
            trits.push_back(char_to_trit(s[i]));
        }
        normalize();
    }

    // ── Conversion to decimal ───────────────────────────────────────────

    int64_t to_int() const {
        int64_t result = 0;
        int64_t power = 1;
        for (size_t i = 0; i < trits.size(); ++i) {
            result += static_cast<int64_t>(trits[i]) * power;
            power *= 3;
        }
        return result;
    }

    // ── Conversion to string (MSB first) ────────────────────────────────

    std::string to_string() const {
        if (trits.empty()) return "0";
        std::string s;
        for (int i = static_cast<int>(trits.size()) - 1; i >= 0; --i) {
            s += trit_to_char(trits[i]);
        }
        return s;
    }

    // ── Number of trits (digits) ────────────────────────────────────────

    size_t num_trits() const { return trits.empty() ? 1 : trits.size(); }

    // ── Trit access (0-indexed from LSB) ────────────────────────────────

    Trit get_trit(size_t pos) const {
        if (pos >= trits.size()) return TRIT_ZERO;
        return trits[pos];
    }

    void set_trit(size_t pos, Trit t) {
        if (pos >= trits.size()) {
            trits.resize(pos + 1, TRIT_ZERO);
        }
        trits[pos] = t;
        normalize();
    }

    // ── Negation ────────────────────────────────────────────────────────

    void negate() {
        for (auto& t : trits) {
            t = negate_trit(t);
        }
    }

    BalancedTernary operator-() const {
        BalancedTernary result = *this;
        result.negate();
        return result;
    }

    // ── Comparison ──────────────────────────────────────────────────────

    bool is_zero() const { return trits.empty(); }

    bool operator==(const BalancedTernary& other) const {
        return trits == other.trits;
    }

    bool operator!=(const BalancedTernary& other) const {
        return !(*this == other);
    }

    int compare(const BalancedTernary& other) const {
        // Compare from MSB
        size_t max_len = std::max(trits.size(), other.trits.size());
        for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
            int a = (static_cast<size_t>(i) < trits.size()) ? trits[i] : 0;
            int b = (static_cast<size_t>(i) < other.trits.size()) ? other.trits[i] : 0;
            if (a < b) return -1;
            if (a > b) return +1;
        }
        return 0;
    }

    bool operator<(const BalancedTernary& o) const { return compare(o) < 0; }
    bool operator>(const BalancedTernary& o) const { return compare(o) > 0; }
    bool operator<=(const BalancedTernary& o) const { return compare(o) <= 0; }
    bool operator>=(const BalancedTernary& o) const { return compare(o) >= 0; }

    // ── Addition ────────────────────────────────────────────────────────
    //
    // Balanced ternary addition: digit-by-digit with carry.
    // Sum of two trits + carry ∈ {-3, -2, -1, 0, 1, 2, 3}
    // We decompose into (new_carry, new_digit) where digit ∈ {-1, 0, 1}

    BalancedTernary operator+(const BalancedTernary& other) const {
        BalancedTernary result;
        size_t max_len = std::max(trits.size(), other.trits.size());
        result.trits.resize(max_len + 1, TRIT_ZERO);

        int carry = 0;
        for (size_t i = 0; i <= max_len; ++i) {
            int a = (i < trits.size()) ? static_cast<int>(trits[i]) : 0;
            int b = (i < other.trits.size()) ? static_cast<int>(other.trits[i]) : 0;
            int sum = a + b + carry;

            // Decompose sum into balanced ternary digit + carry
            carry = 0;
            while (sum > 1)  { sum -= 3; carry += 1; }
            while (sum < -1) { sum += 3; carry -= 1; }

            result.trits[i] = static_cast<Trit>(sum);
        }

        result.normalize();
        return result;
    }

    BalancedTernary& operator+=(const BalancedTernary& other) {
        *this = *this + other;
        return *this;
    }

    // ── Subtraction ─────────────────────────────────────────────────────

    BalancedTernary operator-(const BalancedTernary& other) const {
        return *this + (-other);
    }

    BalancedTernary& operator-=(const BalancedTernary& other) {
        *this = *this - other;
        return *this;
    }

    // ── Multiplication by a single trit ─────────────────────────────────
    //
    // This is the fundamental operation of Fowler's machine:
    // multiplying the entire multiplier by a single digit of the multiplicand.
    //   trit = +1: result is unchanged (pass)
    //   trit =  0: result is zero (skip)
    //   trit = -1: result is negated

    BalancedTernary multiply_by_trit(Trit t) const {
        if (t == TRIT_ZERO) return BalancedTernary();
        if (t == TRIT_POS)  return *this;
        // t == TRIT_NEG
        return -(*this);
    }

    // ── Shift left (multiply by 3^n) ────────────────────────────────────
    //
    // In Fowler's machine, this corresponds to the lateral sliding motion
    // of the multiplier frame to the next column position.

    BalancedTernary shift_left(size_t n) const {
        if (is_zero() || n == 0) return *this;
        BalancedTernary result;
        result.trits.resize(trits.size() + n, TRIT_ZERO);
        for (size_t i = 0; i < trits.size(); ++i) {
            result.trits[i + n] = trits[i];
        }
        return result;
    }

    // ── Full multiplication ─────────────────────────────────────────────
    //
    // Fowler's algorithm: for each trit of the multiplicand (from LSB),
    // multiply the entire multiplier by that trit, shift by position,
    // and accumulate onto the product frame.

    BalancedTernary operator*(const BalancedTernary& other) const {
        BalancedTernary product;
        for (size_t i = 0; i < trits.size(); ++i) {
            if (trits[i] == TRIT_ZERO) continue;
            BalancedTernary partial = other.multiply_by_trit(trits[i]);
            product += partial.shift_left(i);
        }
        return product;
    }

    BalancedTernary& operator*=(const BalancedTernary& other) {
        *this = *this * other;
        return *this;
    }

    // ── Division (returns quotient, remainder via output parameter) ──────
    //
    // Fowler's division is the reverse of multiplication.
    // We implement restoring division in balanced ternary.

    BalancedTernary divmod(const BalancedTernary& divisor,
                           BalancedTernary& remainder) const {
        if (divisor.is_zero()) {
            throw std::domain_error("Division by zero");
        }

        if (is_zero()) {
            remainder = BalancedTernary();
            return BalancedTernary();
        }

        // Work with absolute values, track sign
        BalancedTernary dividend_abs = *this;
        BalancedTernary divisor_abs = divisor;
        bool neg_dividend = (to_int() < 0);
        bool neg_divisor  = (divisor.to_int() < 0);
        if (neg_dividend) dividend_abs.negate();
        if (neg_divisor)  divisor_abs.negate();

        // Determine quotient size
        size_t n = dividend_abs.trits.size();
        size_t m = divisor_abs.trits.size();

        if (n < m) {
            remainder = *this;
            return BalancedTernary();
        }

        // Trial division: for each position from MSB to LSB
        BalancedTernary quotient;
        BalancedTernary rem;

        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            // Shift remainder left and add next trit
            rem = rem.shift_left(1);
            rem.trits[0] = dividend_abs.trits[i];
            rem.normalize();

            // Try each trit value {+1, 0, -1} for this quotient position
            Trit best = TRIT_ZERO;
            BalancedTernary trial_pos = divisor_abs;
            BalancedTernary trial_neg = -divisor_abs;

            // Check if rem - divisor_abs is closer to zero
            BalancedTernary r_pos = rem - divisor_abs;
            BalancedTernary r_neg = rem + divisor_abs;

            int64_t abs_rem = std::abs(rem.to_int());
            int64_t abs_pos = std::abs(r_pos.to_int());
            int64_t abs_neg = std::abs(r_neg.to_int());

            if (abs_pos <= abs_rem && abs_pos <= abs_neg) {
                best = TRIT_POS;
                rem = r_pos;
            } else if (abs_neg < abs_rem && abs_neg < abs_pos) {
                best = TRIT_NEG;
                rem = r_neg;
            }
            // else best stays TRIT_ZERO, rem unchanged

            quotient.set_trit(i, best);
        }

        // Apply signs
        if (neg_dividend != neg_divisor) quotient.negate();
        if (neg_dividend) rem.negate();

        quotient.normalize();
        rem.normalize();
        remainder = rem;
        return quotient;
    }

    BalancedTernary operator/(const BalancedTernary& other) const {
        BalancedTernary rem;
        return divmod(other, rem);
    }

    BalancedTernary operator%(const BalancedTernary& other) const {
        BalancedTernary rem;
        divmod(other, rem);
        return rem;
    }

    // ── Absolute value ──────────────────────────────────────────────────

    BalancedTernary abs() const {
        if (to_int() < 0) return -(*this);
        return *this;
    }

    // ── Stream output ───────────────────────────────────────────────────

    friend std::ostream& operator<<(std::ostream& os, const BalancedTernary& bt) {
        os << bt.to_string();
        return os;
    }

private:
    // Remove trailing zeros
    void normalize() {
        while (!trits.empty() && trits.back() == TRIT_ZERO) {
            trits.pop_back();
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Carry normalization (Fowler's carrying apparatus)
// ─────────────────────────────────────────────────────────────────────────────
//
// After multiplication, the product frame may contain digits outside {-1,0,+1}.
// The carrying apparatus normalizes by: for any digit d where |d| > 1,
// add floor((d+1)/3) to the next column and set current to d - 3*carry.
// In Fowler's terms: "advances the left of two rods by a unit, and throws
// back the right hand rod by 3 units, or vice versa."

inline std::vector<int> carry_normalize(const std::vector<int>& raw_trits) {
    std::vector<int> result = raw_trits;
    // Ensure we have room for carries
    result.push_back(0);

    for (size_t i = 0; i < result.size() - 1; ++i) {
        while (result[i] > 1) {
            result[i] -= 3;
            result[i + 1] += 1;
        }
        while (result[i] < -1) {
            result[i] += 3;
            result[i + 1] -= 1;
        }
    }

    // Handle final position
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }

    // If the last trit still exceeds range, extend
    while (!result.empty() && (result.back() > 1 || result.back() < -1)) {
        int last = result.back();
        int carry = 0;
        while (last > 1)  { last -= 3; carry += 1; }
        while (last < -1) { last += 3; carry -= 1; }
        result.back() = last;
        if (carry != 0) result.push_back(carry);
    }

    // Remove trailing zeros
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }

    return result;
}

// Convert normalized int vector to BalancedTernary
inline BalancedTernary from_raw_trits(const std::vector<int>& raw) {
    auto normalized = carry_normalize(raw);
    std::vector<Trit> trits;
    for (int v : normalized) {
        trits.push_back(static_cast<Trit>(v));
    }
    return BalancedTernary(trits);
}

}  // namespace fowler
}  // namespace cog

#endif  // COG_FOWLER_BALANCED_TERNARY_HPP
