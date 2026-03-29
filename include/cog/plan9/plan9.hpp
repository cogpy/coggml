// cog/plan9/plan9.hpp — Plan 9 Cognitive OS
// 9P2000 protocol, CogFS, MachSpace
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_PLAN9_HPP
#define COG_PLAN9_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <sstream>

namespace cog { namespace plan9 {

// ─────────────────────────────────────────────────────────────────────────────
// 9P2000 Message Types
// ─────────────────────────────────────────────────────────────────────────────
enum class MsgType : uint8_t {
    Tversion = 100, Rversion = 101,
    Tauth    = 102, Rauth    = 103,
    Tattach  = 104, Rattach  = 105,
    Terror   = 106, Rerror   = 107,
    Tflush   = 108, Rflush   = 109,
    Twalk    = 110, Rwalk    = 111,
    Topen    = 112, Ropen    = 113,
    Tcreate  = 114, Rcreate  = 115,
    Tread    = 116, Rread    = 117,
    Twrite   = 118, Rwrite   = 119,
    Tclunk   = 120, Rclunk   = 121,
    Tremove  = 122, Rremove  = 123,
    Tstat    = 124, Rstat    = 125,
    Twstat   = 126, Rwstat   = 127
};

// Qid — unique file identifier
struct Qid {
    uint64_t path;   // unique path identifier
    uint32_t vers;   // version (modification count)
    uint8_t  type;   // QTDIR, QTFILE, etc.

    static const uint8_t QTDIR  = 0x80;
    static const uint8_t QTFILE = 0x00;

    Qid() : path(0), vers(0), type(QTFILE) {}
    Qid(uint64_t p, uint32_t v, uint8_t t) : path(p), vers(v), type(t) {}

    bool is_dir() const { return (type & QTDIR) != 0; }
};

// Stat — file metadata
struct Stat {
    uint16_t    type;
    uint32_t    dev;
    Qid         qid;
    uint32_t    mode;
    uint32_t    atime;
    uint32_t    mtime;
    uint64_t    length;
    std::string name;
    std::string uid;
    std::string gid;
    std::string muid;

    Stat() : type(0), dev(0), mode(0644), atime(0), mtime(0), length(0) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// 9P2000 Message — variable-length TLV
// ─────────────────────────────────────────────────────────────────────────────
struct Message {
    uint32_t  size;
    MsgType   type;
    uint16_t  tag;
    uint32_t  fid;
    uint32_t  afid;
    uint16_t  msize;
    std::string version;
    std::string aname;
    std::string uname;
    Qid       qid;
    Stat      stat;
    uint8_t   mode;
    uint64_t  offset;
    uint32_t  count;
    std::vector<uint8_t> data;
    std::string ename;  // error string
    std::vector<std::string> wnames;
    std::vector<Qid> wqids;

    Message() : size(0), type(MsgType::Tversion), tag(0xFFFF),
                fid(0), afid(~0u), msize(8192), mode(0),
                offset(0), count(0) {}

    static Message version_request(uint32_t msize = 8192,
                                   const std::string& ver = "9P2000") {
        Message m;
        m.type = MsgType::Tversion;
        m.msize = static_cast<uint16_t>(msize);
        m.version = ver;
        return m;
    }

    static Message error(const std::string& e) {
        Message m;
        m.type = MsgType::Rerror;
        m.ename = e;
        return m;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CogFS — Cognitive Filesystem Node
// Maps AtomSpace handles to 9P file tree
// ─────────────────────────────────────────────────────────────────────────────
struct CogFSNode {
    std::string            name;
    Qid                    qid;
    bool                   is_dir;
    cog::Handle            atom;
    std::string            content;
    std::vector<std::shared_ptr<CogFSNode>> children;

    CogFSNode() : is_dir(false), atom(cog::UNDEFINED_HANDLE) {}
    CogFSNode(const std::string& n, Qid q, bool d)
        : name(n), qid(q), is_dir(d), atom(cog::UNDEFINED_HANDLE) {}

    CogFSNode* find_child(const std::string& n) const {
        for (auto& c : children) {
            if (c->name == n) return c.get();
        }
        return nullptr;
    }

    void add_child(std::shared_ptr<CogFSNode> child) {
        children.push_back(std::move(child));
    }

    std::string read_dir() const {
        std::ostringstream ss;
        for (auto& c : children) {
            ss << (c->is_dir ? "d " : "f ") << c->name << "\n";
        }
        return ss.str();
    }
};

class CogFS {
public:
    CogFS() : next_qid_(1) {
        root_ = std::make_shared<CogFSNode>("/", make_qid(true), true);
    }

    CogFSNode* root() { return root_.get(); }

    CogFSNode* mkdir(const std::string& path) {
        auto node = std::make_shared<CogFSNode>(
            basename(path), make_qid(true), true);
        attach(path, node);
        return node.get();
    }

    CogFSNode* mkfile(const std::string& path,
                      const std::string& content = "") {
        auto node = std::make_shared<CogFSNode>(
            basename(path), make_qid(false), false);
        node->content = content;
        attach(path, node);
        return node.get();
    }

    CogFSNode* lookup(const std::string& path) const {
        if (path == "/" || path.empty()) return root_.get();
        std::vector<std::string> parts = split(path, '/');
        CogFSNode* cur = root_.get();
        for (auto& p : parts) {
            if (p.empty()) continue;
            cur = cur->find_child(p);
            if (!cur) return nullptr;
        }
        return cur;
    }

    size_t node_count() const { return next_qid_ - 1; }

private:
    std::shared_ptr<CogFSNode> root_;
    uint64_t next_qid_;

    Qid make_qid(bool dir) {
        return Qid(next_qid_++, 0, dir ? Qid::QTDIR : Qid::QTFILE);
    }

    std::string basename(const std::string& path) {
        size_t p = path.rfind('/');
        return (p == std::string::npos) ? path : path.substr(p + 1);
    }

    std::string dirname(const std::string& path) {
        size_t p = path.rfind('/');
        if (p == std::string::npos) return "/";
        if (p == 0) return "/";
        return path.substr(0, p);
    }

    std::vector<std::string> split(const std::string& s, char delim) const {
        std::vector<std::string> parts;
        std::istringstream ss(s);
        std::string token;
        while (std::getline(ss, token, delim)) parts.push_back(token);
        return parts;
    }

    void attach(const std::string& path, std::shared_ptr<CogFSNode> node) {
        std::string dir = dirname(path);
        CogFSNode* parent = lookup(dir);
        if (!parent) {
            mkdir(dir);
            parent = lookup(dir);
        }
        if (parent) parent->add_child(std::move(node));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// MachSpace — Namespace / process boundary abstraction
// ─────────────────────────────────────────────────────────────────────────────
struct MachSpace {
    std::string name;
    uint32_t    pid;
    uint32_t    parent_pid;
    CogFS       fs;
    std::unordered_map<std::string, std::string> env;

    MachSpace() : pid(1), parent_pid(0) {}
    explicit MachSpace(const std::string& n, uint32_t p = 1)
        : name(n), pid(p), parent_pid(0) {}

    void setenv(const std::string& k, const std::string& v) {
        env[k] = v;
    }

    std::string getenv(const std::string& k) const {
        auto it = env.find(k);
        return (it != env.end()) ? it->second : "";
    }

    bool has_mount(const std::string& path) const {
        return fs.lookup(path) != nullptr;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// P9Server — minimal 9P2000 server skeleton
// ─────────────────────────────────────────────────────────────────────────────
class P9Server {
public:
    using Handler = std::function<Message(const Message&)>;

    P9Server() { register_defaults(); }

    void mount(MachSpace& space) { space_ = &space; }

    Message dispatch(const Message& req) {
        auto it = handlers_.find(req.type);
        if (it != handlers_.end()) return it->second(req);
        return Message::error("unknown message type");
    }

    void on(MsgType t, Handler h) { handlers_[t] = std::move(h); }

private:
    MachSpace* space_ = nullptr;
    std::unordered_map<MsgType, Handler> handlers_;

    void register_defaults() {
        handlers_[MsgType::Tversion] = [](const Message& req) {
            Message r;
            r.type = MsgType::Rversion;
            r.tag  = req.tag;
            r.msize = req.msize;
            r.version = "9P2000";
            return r;
        };
        handlers_[MsgType::Tflush] = [](const Message& req) {
            Message r;
            r.type = MsgType::Rflush;
            r.tag  = req.tag;
            return r;
        };
    }
};

}} // namespace cog::plan9

#endif // COG_PLAN9_HPP
