// cog/webvm/webvm.hpp — Web AtomSpace VM
// S-expression parser, Scheme REPL, JSON serialization
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_WEBVM_HPP
#define COG_WEBVM_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <stdexcept>

namespace cog { namespace webvm {

// ─────────────────────────────────────────────────────────────────────────────
// S-Expression Parser
// ─────────────────────────────────────────────────────────────────────────────
struct SExpr {
    enum class Type { ATOM, LIST, NUMBER, STRING };

    Type                         type;
    std::string                  atom;   // for ATOM or STRING
    double                       number; // for NUMBER
    std::vector<std::shared_ptr<SExpr>> list; // for LIST

    SExpr() : type(Type::ATOM), number(0.0) {}

    static std::shared_ptr<SExpr> make_atom(const std::string& s) {
        auto e = std::make_shared<SExpr>();
        e->type = Type::ATOM;
        e->atom = s;
        return e;
    }

    static std::shared_ptr<SExpr> make_string(const std::string& s) {
        auto e = std::make_shared<SExpr>();
        e->type = Type::STRING;
        e->atom = s;
        return e;
    }

    static std::shared_ptr<SExpr> make_number(double n) {
        auto e = std::make_shared<SExpr>();
        e->type = Type::NUMBER;
        e->number = n;
        return e;
    }

    static std::shared_ptr<SExpr> make_list(std::vector<std::shared_ptr<SExpr>> items) {
        auto e = std::make_shared<SExpr>();
        e->type = Type::LIST;
        e->list = std::move(items);
        return e;
    }

    bool is_atom()   const { return type == Type::ATOM;   }
    bool is_list()   const { return type == Type::LIST;   }
    bool is_number() const { return type == Type::NUMBER; }
    bool is_string() const { return type == Type::STRING; }

    std::string to_string() const {
        switch (type) {
            case Type::ATOM:   return atom;
            case Type::STRING: return "\"" + atom + "\"";
            case Type::NUMBER: {
                std::ostringstream ss;
                ss << number;
                return ss.str();
            }
            case Type::LIST: {
                std::ostringstream ss;
                ss << "(";
                for (size_t i = 0; i < list.size(); ++i) {
                    if (i) ss << " ";
                    ss << (list[i] ? list[i]->to_string() : "nil");
                }
                ss << ")";
                return ss.str();
            }
        }
        return "nil";
    }
};

using SExprPtr = std::shared_ptr<SExpr>;
using ParseError = std::runtime_error;

class SParser {
public:
    SParser() {}

    // Parse all expressions from input
    std::vector<SExprPtr> parse_all(const std::string& src) {
        src_ = src;
        pos_ = 0;
        std::vector<SExprPtr> results;
        skip_ws();
        while (pos_ < src_.size()) {
            results.push_back(parse_expr());
            skip_ws();
        }
        return results;
    }

    // Parse a single expression
    SExprPtr parse(const std::string& src) {
        src_ = src;
        pos_ = 0;
        skip_ws();
        return parse_expr();
    }

private:
    std::string src_;
    size_t      pos_;

    void skip_ws() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else if (c == ';') {
                // Line comment
                while (pos_ < src_.size() && src_[pos_] != '\n') ++pos_;
            } else {
                break;
            }
        }
    }

    SExprPtr parse_expr() {
        skip_ws();
        if (pos_ >= src_.size()) return SExpr::make_atom("nil");

        char c = src_[pos_];
        if (c == '(')  return parse_list();
        if (c == '"')  return parse_string();
        if (c == '\'') return parse_quote();
        return parse_atom_or_number();
    }

    SExprPtr parse_list() {
        assert(src_[pos_] == '(');
        ++pos_;  // consume '('
        std::vector<SExprPtr> items;
        skip_ws();
        while (pos_ < src_.size() && src_[pos_] != ')') {
            items.push_back(parse_expr());
            skip_ws();
        }
        if (pos_ < src_.size()) ++pos_;  // consume ')'
        return SExpr::make_list(std::move(items));
    }

    SExprPtr parse_string() {
        assert(src_[pos_] == '"');
        ++pos_;
        std::string s;
        while (pos_ < src_.size() && src_[pos_] != '"') {
            if (src_[pos_] == '\\' && pos_ + 1 < src_.size()) {
                ++pos_;
                switch (src_[pos_]) {
                    case 'n': s += '\n'; break;
                    case 't': s += '\t'; break;
                    case '"': s += '"';  break;
                    case '\\': s += '\\'; break;
                    default: s += src_[pos_]; break;
                }
            } else {
                s += src_[pos_];
            }
            ++pos_;
        }
        if (pos_ < src_.size()) ++pos_;  // consume closing '"'
        return SExpr::make_string(s);
    }

    SExprPtr parse_quote() {
        assert(src_[pos_] == '\'');
        ++pos_;
        auto inner = parse_expr();
        return SExpr::make_list({SExpr::make_atom("quote"), inner});
    }

    SExprPtr parse_atom_or_number() {
        size_t start = pos_;
        static const std::string delimiters = " \t\n\r()\"";
        while (pos_ < src_.size() &&
               delimiters.find(src_[pos_]) == std::string::npos) {
            ++pos_;
        }
        std::string tok = src_.substr(start, pos_ - start);
        if (tok.empty()) return SExpr::make_atom("nil");
        // Try to parse as number
        char* end = nullptr;
        double n = std::strtod(tok.c_str(), &end);
        if (end == tok.c_str() + tok.size()) {
            return SExpr::make_number(n);
        }
        return SExpr::make_atom(tok);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Scheme REPL — minimal Scheme interpreter for AtomSpace scripting
// ─────────────────────────────────────────────────────────────────────────────
struct SchemeEnv;
using SchemeEnvPtr = std::shared_ptr<SchemeEnv>;
using SchemeFunc = std::function<SExprPtr(const std::vector<SExprPtr>&, SchemeEnvPtr)>;

struct SchemeEnv {
    std::unordered_map<std::string, SExprPtr>  bindings;
    std::unordered_map<std::string, SchemeFunc> builtins;
    SchemeEnvPtr parent;

    SchemeEnv() {}
    explicit SchemeEnv(SchemeEnvPtr p) : parent(std::move(p)) {}

    void define(const std::string& name, SExprPtr val) {
        bindings[name] = std::move(val);
    }

    void define_builtin(const std::string& name, SchemeFunc fn) {
        builtins[name] = std::move(fn);
    }

    SExprPtr lookup(const std::string& name) const {
        auto it = bindings.find(name);
        if (it != bindings.end()) return it->second;
        if (parent) return parent->lookup(name);
        return nullptr;
    }

    SchemeFunc* lookup_builtin(const std::string& name) {
        auto it = builtins.find(name);
        if (it != builtins.end()) return &it->second;
        if (parent) return parent->lookup_builtin(name);
        return nullptr;
    }
};

class SchemeREPL {
public:
    SchemeREPL() {
        env_ = std::make_shared<SchemeEnv>();
        register_core_builtins();
    }

    // Evaluate source string, return result as string
    std::string eval_str(const std::string& src) {
        SParser parser;
        try {
            auto exprs = parser.parse_all(src);
            SExprPtr result = SExpr::make_atom("nil");
            for (auto& e : exprs) {
                result = eval(e, env_);
            }
            return result ? result->to_string() : "nil";
        } catch (const std::exception& ex) {
            return std::string("ERROR: ") + ex.what();
        }
    }

    // Expose a value to the Scheme environment
    void define(const std::string& name, SExprPtr val) {
        env_->define(name, std::move(val));
    }

    // Register a custom built-in function
    void define_builtin(const std::string& name, SchemeFunc fn) {
        env_->define_builtin(name, std::move(fn));
    }

    SchemeEnvPtr env() { return env_; }

private:
    SchemeEnvPtr env_;

    SExprPtr eval(SExprPtr expr, SchemeEnvPtr env) {
        if (!expr) return SExpr::make_atom("nil");
        switch (expr->type) {
            case SExpr::Type::NUMBER:
            case SExpr::Type::STRING:
                return expr;
            case SExpr::Type::ATOM: {
                if (expr->atom == "nil"  || expr->atom == "#f") return expr;
                if (expr->atom == "#t") return expr;
                auto val = env->lookup(expr->atom);
                if (val) return val;
                return expr;  // unbound symbol returns itself
            }
            case SExpr::Type::LIST: {
                if (expr->list.empty()) return SExpr::make_atom("nil");
                auto head = expr->list[0];
                // Special forms
                if (head->type == SExpr::Type::ATOM) {
                    const std::string& op = head->atom;
                    if (op == "quote")  return eval_quote(expr);
                    if (op == "define") return eval_define(expr, env);
                    if (op == "lambda") return eval_lambda(expr, env);
                    if (op == "if")     return eval_if(expr, env);
                    if (op == "cond")   return eval_cond(expr, env);
                    if (op == "let")    return eval_let(expr, env);
                    if (op == "begin")  return eval_begin(expr, env);
                    if (op == "set!")   return eval_set(expr, env);
                    // Look up builtin
                    auto* bf = env->lookup_builtin(op);
                    if (bf) {
                        std::vector<SExprPtr> args;
                        for (size_t i = 1; i < expr->list.size(); ++i) {
                            args.push_back(eval(expr->list[i], env));
                        }
                        return (*bf)(args, env);
                    }
                }
                // Function application
                auto fn = eval(head, env);
                if (fn && fn->type == SExpr::Type::LIST &&
                    !fn->list.empty() &&
                    fn->list[0]->atom == "__lambda__") {
                    return apply_lambda(fn, expr, env);
                }
                // Unknown — return as list
                return expr;
            }
        }
        return SExpr::make_atom("nil");
    }

    SExprPtr eval_quote(SExprPtr expr) {
        if (expr->list.size() < 2) return SExpr::make_atom("nil");
        return expr->list[1];
    }

    SExprPtr eval_define(SExprPtr expr, SchemeEnvPtr env) {
        if (expr->list.size() < 3) return SExpr::make_atom("nil");
        auto name_expr = expr->list[1];
        if (name_expr->type == SExpr::Type::ATOM) {
            auto val = eval(expr->list[2], env);
            env->define(name_expr->atom, val);
            return val;
        }
        return SExpr::make_atom("nil");
    }

    SExprPtr eval_lambda(SExprPtr expr, SchemeEnvPtr capture_env) {
        // (lambda (params...) body...)
        // Pack lambda as tagged list: (__lambda__ params body env_handle)
        auto lam = std::make_shared<SExpr>();
        lam->type = SExpr::Type::LIST;
        lam->list.push_back(SExpr::make_atom("__lambda__"));
        if (expr->list.size() > 1) lam->list.push_back(expr->list[1]);
        if (expr->list.size() > 2) lam->list.push_back(expr->list[2]);
        // Store capture env pointer as atom (simplified — no closures over mutable state)
        (void)capture_env;
        return lam;
    }

    SExprPtr apply_lambda(SExprPtr fn, SExprPtr call_expr, SchemeEnvPtr env) {
        if (fn->list.size() < 3) return SExpr::make_atom("nil");
        auto params = fn->list[1];
        auto body   = fn->list[2];
        auto local  = std::make_shared<SchemeEnv>(env);
        if (params && params->type == SExpr::Type::LIST) {
            for (size_t i = 0; i < params->list.size(); ++i) {
                std::string pname = params->list[i]->atom;
                SExprPtr arg = SExpr::make_atom("nil");
                if (i + 1 < call_expr->list.size()) {
                    arg = eval(call_expr->list[i + 1], env);
                }
                local->define(pname, arg);
            }
        }
        return eval(body, local);
    }

    SExprPtr eval_if(SExprPtr expr, SchemeEnvPtr env) {
        if (expr->list.size() < 3) return SExpr::make_atom("nil");
        auto cond_val = eval(expr->list[1], env);
        bool truthy = cond_val &&
                      !(cond_val->type == SExpr::Type::ATOM &&
                        cond_val->atom == "#f") &&
                      !(cond_val->type == SExpr::Type::ATOM &&
                        cond_val->atom == "nil") &&
                      !(cond_val->type == SExpr::Type::NUMBER &&
                        cond_val->number == 0.0);
        if (truthy) return eval(expr->list[2], env);
        if (expr->list.size() > 3) return eval(expr->list[3], env);
        return SExpr::make_atom("nil");
    }

    SExprPtr eval_cond(SExprPtr expr, SchemeEnvPtr env) {
        for (size_t i = 1; i < expr->list.size(); ++i) {
            auto clause = expr->list[i];
            if (!clause || clause->type != SExpr::Type::LIST) continue;
            if (clause->list.empty()) continue;
            auto test = clause->list[0];
            bool truthy = (test->atom == "else");
            if (!truthy) {
                auto val = eval(test, env);
                truthy = !(val->type == SExpr::Type::ATOM &&
                           (val->atom == "#f" || val->atom == "nil"));
            }
            if (truthy && clause->list.size() > 1) {
                return eval(clause->list[1], env);
            }
        }
        return SExpr::make_atom("nil");
    }

    SExprPtr eval_let(SExprPtr expr, SchemeEnvPtr env) {
        if (expr->list.size() < 3) return SExpr::make_atom("nil");
        auto local = std::make_shared<SchemeEnv>(env);
        auto bindings = expr->list[1];
        if (bindings && bindings->type == SExpr::Type::LIST) {
            for (auto& binding : bindings->list) {
                if (!binding || binding->type != SExpr::Type::LIST ||
                    binding->list.size() < 2) continue;
                std::string nm = binding->list[0]->atom;
                auto val = eval(binding->list[1], env);
                local->define(nm, val);
            }
        }
        return eval(expr->list[2], local);
    }

    SExprPtr eval_begin(SExprPtr expr, SchemeEnvPtr env) {
        SExprPtr result = SExpr::make_atom("nil");
        for (size_t i = 1; i < expr->list.size(); ++i) {
            result = eval(expr->list[i], env);
        }
        return result;
    }

    SExprPtr eval_set(SExprPtr expr, SchemeEnvPtr env) {
        if (expr->list.size() < 3) return SExpr::make_atom("nil");
        std::string name = expr->list[1]->atom;
        auto val = eval(expr->list[2], env);
        env->define(name, val);
        return val;
    }

    void register_core_builtins() {
        // Arithmetic
        env_->define_builtin("+", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            double sum = 0.0;
            for (auto& a : args) if (a->is_number()) sum += a->number;
            return SExpr::make_number(sum);
        });
        env_->define_builtin("-", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.empty()) return SExpr::make_number(0.0);
            double val = args[0]->is_number() ? args[0]->number : 0.0;
            for (size_t i = 1; i < args.size(); ++i) {
                if (args[i]->is_number()) val -= args[i]->number;
            }
            return SExpr::make_number(val);
        });
        env_->define_builtin("*", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            double prod = 1.0;
            for (auto& a : args) if (a->is_number()) prod *= a->number;
            return SExpr::make_number(prod);
        });
        env_->define_builtin("/", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.empty()) return SExpr::make_number(1.0);
            double val = args[0]->is_number() ? args[0]->number : 1.0;
            for (size_t i = 1; i < args.size(); ++i) {
                double d = args[i]->is_number() ? args[i]->number : 0.0;
                if (d != 0.0) val /= d;
            }
            return SExpr::make_number(val);
        });
        // Comparison
        env_->define_builtin("=", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.size() < 2) return SExpr::make_atom("#t");
            bool eq = args[0]->is_number() && args[1]->is_number() &&
                      args[0]->number == args[1]->number;
            return SExpr::make_atom(eq ? "#t" : "#f");
        });
        env_->define_builtin("<", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.size() < 2) return SExpr::make_atom("#f");
            bool lt = args[0]->is_number() && args[1]->is_number() &&
                      args[0]->number < args[1]->number;
            return SExpr::make_atom(lt ? "#t" : "#f");
        });
        env_->define_builtin(">", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.size() < 2) return SExpr::make_atom("#f");
            bool gt = args[0]->is_number() && args[1]->is_number() &&
                      args[0]->number > args[1]->number;
            return SExpr::make_atom(gt ? "#t" : "#f");
        });
        // List operations
        env_->define_builtin("car", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.empty() || !args[0]->is_list() || args[0]->list.empty())
                return SExpr::make_atom("nil");
            return args[0]->list[0];
        });
        env_->define_builtin("cdr", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.empty() || !args[0]->is_list() || args[0]->list.size() < 2)
                return SExpr::make_list({});
            std::vector<SExprPtr> rest(args[0]->list.begin() + 1,
                                       args[0]->list.end());
            return SExpr::make_list(rest);
        });
        env_->define_builtin("cons", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.size() < 2) return SExpr::make_atom("nil");
            std::vector<SExprPtr> items = {args[0]};
            if (args[1]->is_list()) {
                for (auto& x : args[1]->list) items.push_back(x);
            } else {
                items.push_back(args[1]);
            }
            return SExpr::make_list(items);
        });
        env_->define_builtin("list", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            return SExpr::make_list(args);
        });
        env_->define_builtin("null?", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.empty()) return SExpr::make_atom("#t");
            bool nil = (args[0]->is_atom() && args[0]->atom == "nil") ||
                       (args[0]->is_list() && args[0]->list.empty());
            return SExpr::make_atom(nil ? "#t" : "#f");
        });
        // String/display
        env_->define_builtin("display", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (!args.empty()) return args[0];
            return SExpr::make_atom("nil");
        });
        env_->define_builtin("newline", [](const std::vector<SExprPtr>&, SchemeEnvPtr) {
            return SExpr::make_string("\n");
        });
        env_->define_builtin("not", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.empty()) return SExpr::make_atom("#t");
            bool f = args[0]->is_atom() && (args[0]->atom == "#f" || args[0]->atom == "nil");
            return SExpr::make_atom(f ? "#t" : "#f");
        });
        env_->define_builtin("and", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.empty()) return SExpr::make_atom("#t");
            SExprPtr last = SExpr::make_atom("#t");
            for (auto& a : args) {
                if (a->is_atom() && (a->atom == "#f" || a->atom == "nil"))
                    return SExpr::make_atom("#f");
                last = a;
            }
            return last;
        });
        env_->define_builtin("or", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            for (auto& a : args) {
                if (!(a->is_atom() && (a->atom == "#f" || a->atom == "nil")))
                    return a;
            }
            return SExpr::make_atom("#f");
        });
        env_->define_builtin("number->string", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.empty() || !args[0]->is_number())
                return SExpr::make_string("");
            std::ostringstream ss; ss << args[0]->number;
            return SExpr::make_string(ss.str());
        });
        env_->define_builtin("string->number", [](const std::vector<SExprPtr>& args, SchemeEnvPtr) {
            if (args.empty() || !args[0]->is_string())
                return SExpr::make_atom("#f");
            char* end = nullptr;
            double n = std::strtod(args[0]->atom.c_str(), &end);
            if (end == args[0]->atom.c_str()) return SExpr::make_atom("#f");
            return SExpr::make_number(n);
        });
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// JSON Serializer — lightweight AtomSpace ↔ JSON
// ─────────────────────────────────────────────────────────────────────────────
class JSONSerializer {
public:
    // Escape string for JSON
    static std::string escape(const std::string& s) {
        std::ostringstream ss;
        for (char c : s) {
            switch (c) {
                case '"':  ss << "\\\""; break;
                case '\\': ss << "\\\\"; break;
                case '\n': ss << "\\n";  break;
                case '\r': ss << "\\r";  break;
                case '\t': ss << "\\t";  break;
                default:   ss << c; break;
            }
        }
        return ss.str();
    }

    // Serialize SExpr to JSON
    static std::string sexpr_to_json(const SExpr& e) {
        switch (e.type) {
            case SExpr::Type::NUMBER: {
                std::ostringstream ss; ss << e.number; return ss.str();
            }
            case SExpr::Type::STRING:
                return "\"" + escape(e.atom) + "\"";
            case SExpr::Type::ATOM:
                if (e.atom == "#t") return "true";
                if (e.atom == "#f") return "false";
                if (e.atom == "nil") return "null";
                return "\"" + escape(e.atom) + "\"";
            case SExpr::Type::LIST: {
                std::ostringstream ss;
                ss << "[";
                for (size_t i = 0; i < e.list.size(); ++i) {
                    if (i) ss << ",";
                    ss << (e.list[i] ? sexpr_to_json(*e.list[i]) : "null");
                }
                ss << "]";
                return ss.str();
            }
        }
        return "null";
    }

    // Build JSON object from key-value pairs
    static std::string make_object(
            const std::vector<std::pair<std::string, std::string>>& kv) {
        std::ostringstream ss;
        ss << "{";
        for (size_t i = 0; i < kv.size(); ++i) {
            if (i) ss << ",";
            ss << "\"" << escape(kv[i].first) << "\":" << kv[i].second;
        }
        ss << "}";
        return ss.str();
    }

    // Build JSON array from values
    static std::string make_array(const std::vector<std::string>& items) {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < items.size(); ++i) {
            if (i) ss << ",";
            ss << items[i];
        }
        ss << "]";
        return ss.str();
    }

    // Quote a string value
    static std::string str(const std::string& s) {
        return "\"" + escape(s) + "\"";
    }

    // Numeric value
    static std::string num(double n) {
        std::ostringstream ss; ss << n; return ss.str();
    }

    // Boolean value
    static std::string boolean(bool b) {
        return b ? "true" : "false";
    }
};

}} // namespace cog::webvm

#endif // COG_WEBVM_HPP
