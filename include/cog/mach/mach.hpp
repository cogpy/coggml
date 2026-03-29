// cog/mach/mach.hpp — Mach Microkernel Cognitive
// Q16.16 fixed-point, IPC, VM, kernel AtomSpace
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_MACH_HPP
#define COG_MACH_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace cog { namespace mach {

// ─────────────────────────────────────────────────────────────────────────────
// Q16.16 Fixed-Point Arithmetic
// 16-bit integer part, 16-bit fractional part
// Range: [-32768, 32767.99998]
// ─────────────────────────────────────────────────────────────────────────────
struct Fixed {
    int32_t raw;  // Q16.16

    static const int FRAC_BITS = 16;
    static const int32_t ONE = 1 << FRAC_BITS;

    Fixed() : raw(0) {}
    explicit Fixed(int32_t r) : raw(r) {}

    static Fixed from_int(int v) {
        return Fixed(static_cast<int32_t>(v) << FRAC_BITS);
    }

    static Fixed from_float(float f) {
        return Fixed(static_cast<int32_t>(f * (float)ONE));
    }

    float to_float() const {
        return (float)raw / (float)ONE;
    }

    int to_int() const {
        return raw >> FRAC_BITS;
    }

    Fixed operator+(const Fixed& o) const { return Fixed(raw + o.raw); }
    Fixed operator-(const Fixed& o) const { return Fixed(raw - o.raw); }
    Fixed operator-()               const { return Fixed(-raw); }

    Fixed operator*(const Fixed& o) const {
        return Fixed((int32_t)(((int64_t)raw * o.raw) >> FRAC_BITS));
    }

    Fixed operator/(const Fixed& o) const {
        assert(o.raw != 0);
        return Fixed((int32_t)(((int64_t)raw << FRAC_BITS) / o.raw));
    }

    bool operator==(const Fixed& o) const { return raw == o.raw; }
    bool operator!=(const Fixed& o) const { return raw != o.raw; }
    bool operator< (const Fixed& o) const { return raw < o.raw;  }
    bool operator<=(const Fixed& o) const { return raw <= o.raw; }
    bool operator> (const Fixed& o) const { return raw > o.raw;  }
    bool operator>=(const Fixed& o) const { return raw >= o.raw; }

    Fixed& operator+=(const Fixed& o) { raw += o.raw; return *this; }
    Fixed& operator-=(const Fixed& o) { raw -= o.raw; return *this; }
    Fixed& operator*=(const Fixed& o) { *this = *this * o; return *this; }

    // Absolute value
    Fixed abs() const { return Fixed(raw < 0 ? -raw : raw); }

    // Integer floor
    Fixed floor() const { return Fixed(raw & ~(ONE - 1)); }

    // Reciprocal approximation via Newton–Raphson
    Fixed recip() const {
        assert(raw != 0);
        return Fixed::from_float(1.0f / to_float());
    }
};

inline Fixed fixed_sqrt(Fixed x) {
    if (x.raw <= 0) return Fixed(0);
    return Fixed::from_float(std::sqrt(x.to_float()));
}

inline Fixed fixed_tanh(Fixed x) {
    return Fixed::from_float(std::tanh(x.to_float()));
}

inline Fixed fixed_exp(Fixed x) {
    return Fixed::from_float(std::exp(x.to_float()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixed-Point Tensor (1D or 2D, Q16.16)
// ─────────────────────────────────────────────────────────────────────────────
struct FixedTensor {
    std::vector<Fixed> data;
    size_t rows, cols;

    FixedTensor() : rows(0), cols(0) {}
    FixedTensor(size_t r, size_t c) : data(r*c), rows(r), cols(c) {}

    Fixed& at(size_t r, size_t c) { return data[r * cols + c]; }
    const Fixed& at(size_t r, size_t c) const { return data[r * cols + c]; }

    void fill(Fixed v) { std::fill(data.begin(), data.end(), v); }

    // Matrix–vector multiply (rows × cols) × (cols,) → (rows,)
    std::vector<Fixed> matvec(const std::vector<Fixed>& v) const {
        assert(v.size() == cols);
        std::vector<Fixed> out(rows);
        for (size_t r = 0; r < rows; ++r) {
            Fixed sum;
            for (size_t c = 0; c < cols; ++c) {
                sum += at(r, c) * v[c];
            }
            out[r] = sum;
        }
        return out;
    }

    // Element-wise add
    FixedTensor operator+(const FixedTensor& o) const {
        assert(rows == o.rows && cols == o.cols);
        FixedTensor result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + o.data[i];
        }
        return result;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// IPC — Inter-Process Communication (Mach ports)
// ─────────────────────────────────────────────────────────────────────────────
using MachPort = uint32_t;
static const MachPort MACH_PORT_NULL  = 0;
static const MachPort MACH_PORT_DEAD  = ~0u;

enum class MachMsgType : uint32_t {
    SEND         = 1,
    RECEIVE      = 2,
    SEND_ONCE    = 4,
    PORT_RIGHTS  = 8,
    KERNEL_REPLY = 16
};

struct MachMsg {
    uint32_t  msgh_bits;
    uint32_t  msgh_size;
    MachPort  msgh_remote_port;
    MachPort  msgh_local_port;
    uint32_t  msgh_id;
    std::vector<uint8_t> body;

    MachMsg() : msgh_bits(0), msgh_size(0),
                msgh_remote_port(MACH_PORT_NULL),
                msgh_local_port(MACH_PORT_NULL),
                msgh_id(0) {}

    MachMsg(MachPort remote, MachPort local, uint32_t id,
            const std::vector<uint8_t>& b = {})
        : msgh_bits(static_cast<uint32_t>(MachMsgType::SEND)),
          msgh_size(static_cast<uint32_t>(sizeof(MachMsg) + b.size())),
          msgh_remote_port(remote),
          msgh_local_port(local),
          msgh_id(id),
          body(b) {}
};

enum class KernReturn : int32_t {
    SUCCESS        =  0,
    INVALID_ADDR   = -1,
    INVALID_TASK   = -2,
    PORT_BUSY      = -3,
    PORT_DEAD      = -4,
    NO_SPACE       = -5,
    INVALID_RIGHT  = -6,
    FAILURE        = -7
};

class MachKernel {
public:
    using Handler = std::function<KernReturn(const MachMsg&, MachMsg&)>;

    MachKernel() : next_port_(1) {}

    MachPort alloc_port() { return next_port_++; }

    void register_handler(MachPort port, Handler h) {
        handlers_[port] = std::move(h);
    }

    KernReturn msg_send(const MachMsg& msg, MachMsg& reply) {
        auto it = handlers_.find(msg.msgh_remote_port);
        if (it == handlers_.end()) return KernReturn::INVALID_ADDR;
        return it->second(msg, reply);
    }

    size_t port_count() const { return next_port_ - 1; }

private:
    MachPort next_port_;
    std::unordered_map<MachPort, Handler> handlers_;
};

// ─────────────────────────────────────────────────────────────────────────────
// VM — Virtual Memory regions
// ─────────────────────────────────────────────────────────────────────────────
struct VMRegion {
    uint64_t         base;
    uint64_t         size;
    uint32_t         protection;  // PROT_READ=1, PROT_WRITE=2, PROT_EXEC=4
    std::vector<uint8_t> backing;

    static const uint32_t PROT_NONE  = 0;
    static const uint32_t PROT_READ  = 1;
    static const uint32_t PROT_WRITE = 2;
    static const uint32_t PROT_EXEC  = 4;
    static const uint32_t PROT_RW    = PROT_READ | PROT_WRITE;
    static const uint32_t PROT_RWX   = PROT_READ | PROT_WRITE | PROT_EXEC;

    VMRegion() : base(0), size(0), protection(PROT_RW) {}
    VMRegion(uint64_t b, uint64_t s, uint32_t p = PROT_RW)
        : base(b), size(s), protection(p), backing(s, 0) {}

    bool contains(uint64_t addr) const {
        return addr >= base && addr < base + size;
    }

    bool readable()   const { return (protection & PROT_READ) != 0; }
    bool writable()   const { return (protection & PROT_WRITE) != 0; }
    bool executable() const { return (protection & PROT_EXEC) != 0; }
};

class VMMap {
public:
    VMMap() : next_base_(0x1000) {}

    uint64_t allocate(size_t size, uint32_t prot = VMRegion::PROT_RW) {
        uint64_t base = next_base_;
        next_base_ += (uint64_t)((size + 0xFFF) & ~0xFFF);  // page-align
        regions_.emplace_back(base, size, prot);
        return base;
    }

    VMRegion* find(uint64_t addr) {
        for (auto& r : regions_) {
            if (r.contains(addr)) return &r;
        }
        return nullptr;
    }

    bool write(uint64_t addr, const uint8_t* data, size_t len) {
        VMRegion* r = find(addr);
        if (!r || !r->writable()) return false;
        size_t off = (size_t)(addr - r->base);
        if (off + len > r->backing.size()) return false;
        std::memcpy(r->backing.data() + off, data, len);
        return true;
    }

    bool read(uint64_t addr, uint8_t* buf, size_t len) const {
        for (const auto& r : regions_) {
            if (r.contains(addr)) {
                if (!r.readable()) return false;
                size_t off = (size_t)(addr - r.base);
                if (off + len > r.backing.size()) return false;
                std::memcpy(buf, r.backing.data() + off, len);
                return true;
            }
        }
        return false;
    }

    size_t region_count() const { return regions_.size(); }

private:
    std::vector<VMRegion> regions_;
    uint64_t next_base_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel AtomSpace — lightweight AtomSpace embedded in Mach kernel
// ─────────────────────────────────────────────────────────────────────────────
class KernelAtomSpace {
public:
    struct KAtom {
        cog::Handle handle;
        cog::AtomType type;
        std::string name;
        Fixed tv_strength;   // Q16.16 truth value strength
        Fixed tv_confidence; // Q16.16 truth value confidence
    };

    KernelAtomSpace() : next_handle_(1) {}

    cog::Handle add(cog::AtomType type, const std::string& name) {
        cog::Handle h = next_handle_++;
        KAtom a;
        a.handle = h;
        a.type   = type;
        a.name   = name;
        a.tv_strength   = Fixed::from_float(1.0f);
        a.tv_confidence = Fixed::from_float(0.9f);
        atoms_[h] = a;
        names_[name] = h;
        return h;
    }

    KAtom* get(cog::Handle h) {
        auto it = atoms_.find(h);
        return (it != atoms_.end()) ? &it->second : nullptr;
    }

    cog::Handle lookup(const std::string& name) const {
        auto it = names_.find(name);
        return (it != names_.end()) ? it->second : cog::UNDEFINED_HANDLE;
    }

    void set_tv(cog::Handle h, float strength, float confidence) {
        auto* a = get(h);
        if (a) {
            a->tv_strength   = Fixed::from_float(strength);
            a->tv_confidence = Fixed::from_float(confidence);
        }
    }

    size_t size() const { return atoms_.size(); }

private:
    uint32_t next_handle_;
    std::unordered_map<cog::Handle, KAtom> atoms_;
    std::unordered_map<std::string, cog::Handle> names_;
};

}} // namespace cog::mach

#endif // COG_MACH_HPP
