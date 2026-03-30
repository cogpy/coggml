// test/test_all.cpp — Unified test suite for all CogPy modules
// Compile: g++ -std=c++11 -I../include -o test_all test/test_all.cpp && ./test_all
// CMake:   mkdir build && cd build && cmake .. && make && ctest

#include "cog/cog.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

// ─── Test Infrastructure ──────────────────────────────────────────────────────

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, ...) \
    do { \
        std::cout << "  [" #name "] "; \
        bool _ok = true; \
        do { __VA_ARGS__ } while (0); \
        if (_ok) { std::cout << "PASS\n"; tests_passed++; } \
        else     { std::cout << "FAIL\n"; tests_failed++;  } \
    } while (0)

#define REQUIRE(expr) \
    if (!(expr)) { \
        std::cout << "FAIL (line " << __LINE__ << ": " << #expr << ")\n"; \
        _ok = false; \
        break; \
    }

#define REQUIRE_NEAR(a, b, tol) \
    REQUIRE(std::fabs((double)(a) - (double)(b)) < (tol))

static void section(const char* name) {
    std::cout << "\n[" << name << "]\n";
}

// ─── cog::core ────────────────────────────────────────────────────────────────

void test_core() {
    section("cog::core");

    TEST(atom_type_names, {
        REQUIRE(cog::atom_type_name(cog::AtomType::CONCEPT_NODE) == std::string("ConceptNode"));
        REQUIRE(cog::atom_type_name(cog::AtomType::INHERITANCE_LINK) == std::string("InheritanceLink"));
    });

    TEST(is_node_is_link, {
        REQUIRE( cog::is_node(cog::AtomType::CONCEPT_NODE));
        REQUIRE(!cog::is_link(cog::AtomType::CONCEPT_NODE));
        REQUIRE( cog::is_link(cog::AtomType::INHERITANCE_LINK));
        REQUIRE(!cog::is_node(cog::AtomType::INHERITANCE_LINK));
    });

    TEST(handle_types, {
        cog::Handle h1 = 42;
        cog::Handle h2 = cog::UNDEFINED_HANDLE;
        REQUIRE(h1 == 42);
        REQUIRE(h2 == 0);
    });
}

// ─── cog::plan9 ───────────────────────────────────────────────────────────────

void test_plan9() {
    section("cog::plan9");

    TEST(qid_types, {
        cog::plan9::Qid dir_qid(1, 0, cog::plan9::Qid::QTDIR);
        cog::plan9::Qid file_qid(2, 0, cog::plan9::Qid::QTFILE);
        REQUIRE(dir_qid.is_dir());
        REQUIRE(!file_qid.is_dir());
    });

    TEST(cogfs_mkdir_mkfile, {
        cog::plan9::CogFS fs;
        fs.mkdir("/atoms");
        fs.mkfile("/atoms/cat", "ConceptNode: cat");
        auto* dir  = fs.lookup("/atoms");
        auto* file = fs.lookup("/atoms/cat");
        REQUIRE(dir != nullptr);
        REQUIRE(file != nullptr);
        REQUIRE(dir->is_dir);
        REQUIRE(!file->is_dir);
        REQUIRE(file->content == "ConceptNode: cat");
    });

    TEST(cogfs_root, {
        cog::plan9::CogFS fs;
        auto* root = fs.root();
        REQUIRE(root != nullptr);
        REQUIRE(root->is_dir);
        REQUIRE(root->name == "/");
    });

    TEST(machspace_env, {
        cog::plan9::MachSpace space("test", 1);
        space.setenv("COG_HOME", "/cog");
        REQUIRE(space.getenv("COG_HOME") == "/cog");
        REQUIRE(space.getenv("MISSING") == "");
    });

    TEST(p9server_version, {
        cog::plan9::P9Server server;
        cog::plan9::Message req = cog::plan9::Message::version_request(8192);
        cog::plan9::Message reply = server.dispatch(req);
        REQUIRE(reply.type == cog::plan9::MsgType::Rversion);
        REQUIRE(reply.version == "9P2000");
        REQUIRE(reply.msize == 8192);
    });
}

// ─── cog::pilot ───────────────────────────────────────────────────────────────

void test_pilot() {
    section("cog::pilot");

    TEST(a000081_sequence, {
        REQUIRE(cog::pilot::a000081(0) == 0);
        REQUIRE(cog::pilot::a000081(1) == 1);
        REQUIRE(cog::pilot::a000081(2) == 1);
        REQUIRE(cog::pilot::a000081(3) == 2);
        REQUIRE(cog::pilot::a000081(4) == 4);
        REQUIRE(cog::pilot::a000081(5) == 9);
    });

    TEST(bseries_gamma, {
        double g0 = cog::pilot::BSeries::gamma(0);
        double g1 = cog::pilot::BSeries::gamma(1);
        REQUIRE_NEAR(g0, 1.0, 1e-6);
        REQUIRE(g1 > 0.0);
    });

    TEST(jsurface_spectral_radius, {
        double sr = cog::pilot::JSurface::spectral_radius(4);
        REQUIRE(sr > 0.5 && sr < 1.0);
    });

    TEST(jsurface_project, {
        cog::pilot::JSurface surf(4);
        std::vector<double> signal = {1.0, 0.5, 0.2, 0.1};
        auto proj = surf.project(signal);
        REQUIRE(proj.size() == 4);
    });

    TEST(psystem_add_membrane, {
        cog::pilot::PSystem ps;
        auto* m = ps.add_membrane("inner");
        REQUIRE(m != nullptr);
        REQUIRE(ps.membrane_count() >= 2);
    });

    TEST(psystem_rules, {
        cog::pilot::PSystem ps;
        auto* m = ps.add_membrane("reactor");
        m->add_object("a", 3);
        m->add_rule(cog::pilot::PRule("a", "b"));
        ps.evolve_step();
        REQUIRE(m->get_count("a") == 2);
        REQUIRE(m->get_count("b") == 1);
    });

    TEST(esn_initialize, {
        cog::pilot::DualPoolESN esn(64, 8, 8);
        esn.initialize(42);
        REQUIRE(esn.initialized);
        REQUIRE(esn.n_fast == 32);
        REQUIRE(esn.n_slow == 32);
    });

    TEST(esn_forward_step, {
        cog::pilot::DualPoolESN esn(64, 8, 8);
        esn.initialize(42);
        std::vector<float> input(8, 0.1f);
        auto output = esn.step(input);
        REQUIRE(output.size() == 8);
        REQUIRE(esn.step_count == 1);
    });

    TEST(esn_echo_property, {
        // Signal should decay after input stops
        cog::pilot::DualPoolESN esn(64, 4, 4);
        esn.initialize(7);
        std::vector<float> driven(4, 1.0f);
        std::vector<float> zeros(4, 0.0f);
        for (int i = 0; i < 20; ++i) esn.step(driven);
        auto st1 = esn.state();
        for (int i = 0; i < 50; ++i) esn.step(zeros);
        auto st2 = esn.state();
        // State should change from initial after driving
        float norm1 = 0.0f, norm2 = 0.0f;
        for (float v : st1) norm1 += v*v;
        for (float v : st2) norm2 += v*v;
        REQUIRE(norm1 > 0.0f);
        // After silence, state should decay (norm2 < norm1 due to leak)
        REQUIRE(norm2 < norm1 + 1e-6f);
    });

    TEST(cogpilot_forward, {
        cog::pilot::CogPilot pilot(64, 8, 8, 4);
        pilot.initialize(42);
        std::vector<float> input(8, 0.5f);
        auto out = pilot.forward(input);
        REQUIRE(out.size() == 8);
        REQUIRE(pilot.echobeats_step == 1);
    });

    TEST(cogpilot_echobeats_thread, {
        REQUIRE(cog::pilot::CogPilot::echobeats_thread(0) == 0);
        REQUIRE(cog::pilot::CogPilot::echobeats_thread(1) == 1);
        REQUIRE(cog::pilot::CogPilot::echobeats_thread(4) == 0);
        REQUIRE(cog::pilot::CogPilot::echobeats_thread(11) == 3);
    });
}

// ─── cog::mach ────────────────────────────────────────────────────────────────

void test_mach() {
    section("cog::mach");

    TEST(fixed_from_int, {
        auto x = cog::mach::Fixed::from_int(5);
        REQUIRE_NEAR(x.to_float(), 5.0f, 1e-4f);
        REQUIRE(x.to_int() == 5);
    });

    TEST(fixed_from_float, {
        auto x = cog::mach::Fixed::from_float(3.14f);
        REQUIRE_NEAR(x.to_float(), 3.14f, 0.001f);
    });

    TEST(fixed_arithmetic, {
        auto a = cog::mach::Fixed::from_float(2.5f);
        auto b = cog::mach::Fixed::from_float(1.5f);
        REQUIRE_NEAR((a + b).to_float(), 4.0f, 1e-3f);
        REQUIRE_NEAR((a - b).to_float(), 1.0f, 1e-3f);
        REQUIRE_NEAR((a * b).to_float(), 3.75f, 1e-3f);
        REQUIRE_NEAR((a / b).to_float(), 2.5f/1.5f, 0.001f);
    });

    TEST(fixed_comparison, {
        auto a = cog::mach::Fixed::from_float(1.0f);
        auto b = cog::mach::Fixed::from_float(2.0f);
        REQUIRE(a < b);
        REQUIRE(b > a);
        REQUIRE(a == a);
        REQUIRE(a != b);
    });

    TEST(fixed_tensor_matvec, {
        cog::mach::FixedTensor mat(2, 2);
        mat.at(0,0) = cog::mach::Fixed::from_float(1.0f);
        mat.at(0,1) = cog::mach::Fixed::from_float(0.0f);
        mat.at(1,0) = cog::mach::Fixed::from_float(0.0f);
        mat.at(1,1) = cog::mach::Fixed::from_float(1.0f);  // Identity
        std::vector<cog::mach::Fixed> v = {
            cog::mach::Fixed::from_float(3.0f),
            cog::mach::Fixed::from_float(4.0f)
        };
        auto out = mat.matvec(v);
        REQUIRE(out.size() == 2);
        REQUIRE_NEAR(out[0].to_float(), 3.0f, 0.001f);
        REQUIRE_NEAR(out[1].to_float(), 4.0f, 0.001f);
    });

    TEST(mach_kernel_port, {
        cog::mach::MachKernel k;
        auto port = k.alloc_port();
        REQUIRE(port != cog::mach::MACH_PORT_NULL);
    });

    TEST(mach_kernel_ipc, {
        cog::mach::MachKernel k;
        auto port = k.alloc_port();
        k.register_handler(port, [](const cog::mach::MachMsg& req,
                                    cog::mach::MachMsg& rep) {
            rep.msgh_id = req.msgh_id + 1;
            return cog::mach::KernReturn::SUCCESS;
        });
        cog::mach::MachMsg msg(port, cog::mach::MACH_PORT_NULL, 42);
        cog::mach::MachMsg reply;
        auto rc = k.msg_send(msg, reply);
        REQUIRE(rc == cog::mach::KernReturn::SUCCESS);
        REQUIRE(reply.msgh_id == 43);
    });

    TEST(vm_map_allocate, {
        cog::mach::VMMap vm;
        uint64_t addr = vm.allocate(4096);
        REQUIRE(addr != 0);
        auto* r = vm.find(addr);
        REQUIRE(r != nullptr);
        REQUIRE(r->readable());
        REQUIRE(r->writable());
    });

    TEST(vm_map_read_write, {
        cog::mach::VMMap vm;
        uint64_t addr = vm.allocate(256);
        uint8_t data[4] = {1, 2, 3, 4};
        REQUIRE(vm.write(addr, data, 4));
        uint8_t buf[4] = {};
        REQUIRE(vm.read(addr, buf, 4));
        REQUIRE(buf[0] == 1 && buf[1] == 2 && buf[2] == 3 && buf[3] == 4);
    });

    TEST(kernel_atomspace, {
        cog::mach::KernelAtomSpace kas;
        auto h = kas.add(cog::AtomType::CONCEPT_NODE, "kernel-concept");
        REQUIRE(h != cog::UNDEFINED_HANDLE);
        auto* a = kas.get(h);
        REQUIRE(a != nullptr);
        REQUIRE(a->name == "kernel-concept");
        REQUIRE_NEAR(a->tv_strength.to_float(), 1.0f, 0.01f);
        REQUIRE(kas.size() == 1);
    });
}

// ─── cog::lux ─────────────────────────────────────────────────────────────────

void test_lux() {
    section("cog::lux");

    TEST(add_nodes, {
        cog::lux::LuxGraph g;
        auto cat    = g.add_node(cog::lux::NodeType::CONCEPT, "cat");
        auto animal = g.add_node(cog::lux::NodeType::CONCEPT, "animal");
        REQUIRE(g.node_count() == 2);
        REQUIRE(cat != cog::lux::INVALID_NODE);
        REQUIRE(animal != cog::lux::INVALID_NODE);
    });

    TEST(add_edges, {
        cog::lux::LuxGraph g;
        auto cat    = g.add_node(cog::lux::NodeType::CONCEPT, "cat");
        auto animal = g.add_node(cog::lux::NodeType::CONCEPT, "animal");
        auto e      = g.add_edge(cog::lux::EdgeType::INHERITANCE, cat, animal, 1.0f);
        REQUIRE(g.edge_count() == 1);
        REQUIRE(e != cog::lux::INVALID_EDGE);
        REQUIRE(g.out_edges(cat).size() == 1);
        REQUIRE(g.in_edges(animal).size() == 1);
    });

    TEST(find_node, {
        cog::lux::LuxGraph g;
        g.add_node(cog::lux::NodeType::CONCEPT, "dog");
        REQUIRE(g.find_node("dog") != cog::lux::INVALID_NODE);
        REQUIRE(g.find_node("cat") == cog::lux::INVALID_NODE);
    });

    TEST(bfs, {
        cog::lux::LuxGraph g;
        auto a = g.add_node(cog::lux::NodeType::CONCEPT, "a");
        auto b = g.add_node(cog::lux::NodeType::CONCEPT, "b");
        auto c = g.add_node(cog::lux::NodeType::CONCEPT, "c");
        g.add_edge(cog::lux::EdgeType::INHERITANCE, a, b);
        g.add_edge(cog::lux::EdgeType::INHERITANCE, b, c);
        auto visited = g.bfs(a);
        REQUIRE(visited.size() == 3);
        REQUIRE(visited[0] == a);
    });

    TEST(dfs, {
        cog::lux::LuxGraph g;
        auto a = g.add_node(cog::lux::NodeType::CONCEPT, "x");
        auto b = g.add_node(cog::lux::NodeType::CONCEPT, "y");
        auto c = g.add_node(cog::lux::NodeType::CONCEPT, "z");
        g.add_edge(cog::lux::EdgeType::INHERITANCE, a, b);
        g.add_edge(cog::lux::EdgeType::INHERITANCE, b, c);
        auto visited = g.dfs(a);
        REQUIRE(visited.size() == 3);
        REQUIRE(visited[0] == a);
    });

    TEST(pagerank, {
        cog::lux::LuxGraph g;
        auto a = g.add_node(cog::lux::NodeType::CONCEPT, "pa");
        auto b = g.add_node(cog::lux::NodeType::CONCEPT, "pb");
        auto c = g.add_node(cog::lux::NodeType::CONCEPT, "pc");
        g.add_edge(cog::lux::EdgeType::INHERITANCE, a, b);
        g.add_edge(cog::lux::EdgeType::INHERITANCE, a, c);
        g.add_edge(cog::lux::EdgeType::INHERITANCE, b, c);
        auto pr = g.pagerank();
        REQUIRE(pr.find(c) != pr.end());
        // c has two incoming links, should have highest rank
        REQUIRE(pr[c] > pr[a]);
    });

    TEST(dot_export, {
        cog::lux::LuxGraph g;
        auto a = g.add_node(cog::lux::NodeType::CONCEPT, "alpha");
        auto b = g.add_node(cog::lux::NodeType::CONCEPT, "beta");
        g.add_edge(cog::lux::EdgeType::SIMILARITY, a, b);
        std::string dot = g.to_dot();
        REQUIRE(dot.find("digraph") != std::string::npos);
        REQUIRE(dot.find("alpha") != std::string::npos);
        REQUIRE(dot.find("Similarity") != std::string::npos);
    });

    TEST(node_attrs, {
        cog::lux::LuxGraph g;
        auto n = g.add_node(cog::lux::NodeType::CONCEPT, "tagged");
        auto* node = g.node(n);
        REQUIRE(node != nullptr);
        node->set_attr("color", "blue");
        REQUIRE(node->get_attr("color") == "blue");
        REQUIRE(node->get_attr("missing", "default") == "default");
    });
}

// ─── cog::glow ────────────────────────────────────────────────────────────────

void test_glow() {
    section("cog::glow");

    TEST(graph_add_nodes_values, {
        cog::glow::GlowGraph g;
        auto v  = g.add_value(cog::glow::IRType(cog::gml::DType::F32, {4}), "x");
        auto n  = g.add_node(cog::glow::OpType::RELU, "relu");
        REQUIRE(g.node_count() == 1);
        REQUIRE(g.value_count() == 1);
        REQUIRE(v != cog::glow::INVALID_VALUE);
        REQUIRE(n != cog::glow::INVALID_NODE);
    });

    TEST(op_type_names, {
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::ADD)) == "Add");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::RELU)) == "Relu");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::MATMUL)) == "MatMul");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::SOFTMAX)) == "Softmax");
    });

    TEST(irtype_to_string, {
        cog::glow::IRType t(cog::gml::DType::F32, {2, 4});
        std::string s = t.to_string();
        REQUIRE(s.find("f32") != std::string::npos);
        REQUIRE(s.find("2") != std::string::npos);
        REQUIRE(s.find("4") != std::string::npos);
    });

    TEST(compiler_add_graph, {
        cog::glow::GlowCompiler comp;
        auto x  = comp.input({4}, cog::gml::DType::F32, "x");
        auto rx = comp.op1(cog::glow::OpType::RELU, x, "relu_x");
        comp.mark_output(rx);
        REQUIRE(comp.graph().node_count() >= 2);
        REQUIRE(comp.graph().outputs().size() == 1);
    });

    TEST(interpreter_add, {
        cog::glow::GlowCompiler comp;
        auto a  = comp.input({3}, cog::gml::DType::F32, "a");
        auto b  = comp.input({3}, cog::gml::DType::F32, "b");
        auto c  = comp.op2(cog::glow::OpType::ADD, a, b, "c");
        comp.mark_output(c);

        cog::glow::GlowInterpreter interp;
        cog::glow::GlowInterpreter::TensorMap inputs;
        inputs[a] = {1.0f, 2.0f, 3.0f};
        inputs[b] = {4.0f, 5.0f, 6.0f};
        auto results = interp.run(comp.graph(), inputs);
        REQUIRE(!results.empty());
        auto it = results.find(c);
        REQUIRE(it != results.end());
        REQUIRE_NEAR(it->second[0], 5.0f, 1e-5f);
        REQUIRE_NEAR(it->second[1], 7.0f, 1e-5f);
        REQUIRE_NEAR(it->second[2], 9.0f, 1e-5f);
    });

    TEST(interpreter_relu, {
        cog::glow::GlowCompiler comp;
        auto x  = comp.input({4}, cog::gml::DType::F32, "x");
        auto y  = comp.op1(cog::glow::OpType::RELU, x, "y");
        comp.mark_output(y);

        cog::glow::GlowInterpreter interp;
        cog::glow::GlowInterpreter::TensorMap inputs;
        inputs[x] = {-2.0f, -0.5f, 0.0f, 3.0f};
        auto results = interp.run(comp.graph(), inputs);
        auto it = results.find(y);
        REQUIRE(it != results.end());
        REQUIRE_NEAR(it->second[0], 0.0f, 1e-5f);
        REQUIRE_NEAR(it->second[1], 0.0f, 1e-5f);
        REQUIRE_NEAR(it->second[2], 0.0f, 1e-5f);
        REQUIRE_NEAR(it->second[3], 3.0f, 1e-5f);
    });

    TEST(interpreter_softmax, {
        cog::glow::GlowCompiler comp;
        auto x  = comp.input({3}, cog::gml::DType::F32, "x");
        auto y  = comp.op1(cog::glow::OpType::SOFTMAX, x, "y");
        comp.mark_output(y);

        cog::glow::GlowInterpreter interp;
        cog::glow::GlowInterpreter::TensorMap inputs;
        inputs[x] = {1.0f, 2.0f, 3.0f};
        auto results = interp.run(comp.graph(), inputs);
        auto it = results.find(y);
        REQUIRE(it != results.end());
        REQUIRE(it->second.size() == 3);
        float sum = 0.0f;
        for (float v : it->second) sum += v;
        REQUIRE_NEAR(sum, 1.0f, 1e-4f);
        // Highest input should give highest probability
        REQUIRE(it->second[2] > it->second[1]);
        REQUIRE(it->second[1] > it->second[0]);
    });

    TEST(topological_sort, {
        cog::glow::GlowCompiler comp;
        auto x  = comp.input({4}, cog::gml::DType::F32, "x");
        auto h1 = comp.op1(cog::glow::OpType::RELU, x, "h1");
        auto h2 = comp.op1(cog::glow::OpType::TANH, h1, "h2");
        comp.mark_output(h2);
        auto order = comp.graph().topological_order();
        REQUIRE(order.size() >= 3);
    });

    TEST(dot_export, {
        cog::glow::GlowCompiler comp;
        auto x = comp.input({4}, cog::gml::DType::F32, "x");
        auto y = comp.op1(cog::glow::OpType::RELU, x, "y");
        comp.mark_output(y);
        std::string dot = comp.graph().to_dot("TestGraph");
        REQUIRE(dot.find("digraph TestGraph") != std::string::npos);
    });
}

// ─── cog::gml ─────────────────────────────────────────────────────────────────

void test_gml() {
    section("cog::gml");

    TEST(dtype_names, {
        REQUIRE(std::string(cog::gml::dtype_name(cog::gml::DType::F32)) == "f32");
        REQUIRE(std::string(cog::gml::dtype_name(cog::gml::DType::Q4_0)) == "q4_0");
        REQUIRE(std::string(cog::gml::dtype_name(cog::gml::DType::Q8_0)) == "q8_0");
    });

    TEST(dtype_sizes, {
        REQUIRE(cog::gml::dtype_size(cog::gml::DType::F32) == 4);
        REQUIRE(cog::gml::dtype_size(cog::gml::DType::F16) == 2);
        REQUIRE(cog::gml::dtype_size(cog::gml::DType::I32) == 4);
    });

    TEST(f16_roundtrip, {
        float vals[] = {0.0f, 1.0f, -1.0f, 3.14f, 0.001f, 100.0f};
        for (float v : vals) {
            uint16_t f16 = cog::gml::f32_to_f16(v);
            float    f32 = cog::gml::f16_to_f32(f16);
            REQUIRE_NEAR(f32, v, std::fabs(v) * 0.01f + 0.001f);
        }
    });
}

// ─── cog::prime ───────────────────────────────────────────────────────────────

void test_prime() {
    section("cog::prime");

    TEST(truth_value_revision, {
        cog::prime::TruthValue t1(0.8f, 0.9f);
        cog::prime::TruthValue t2(0.6f, 0.7f);
        auto rev = cog::prime::TruthValue::revise(t1, t2);
        REQUIRE(rev.strength > 0.0f && rev.strength < 1.0f);
        REQUIRE(rev.confidence > 0.0f);
    });

    TEST(truth_value_conjunction, {
        cog::prime::TruthValue t1(0.9f, 0.8f);
        cog::prime::TruthValue t2(0.7f, 0.6f);
        auto conj = cog::prime::TruthValue::conjunction(t1, t2);
        REQUIRE_NEAR(conj.strength, 0.9f * 0.7f, 1e-5f);
    });

    TEST(truth_value_negation, {
        cog::prime::TruthValue t(0.3f, 0.9f);
        auto neg = cog::prime::TruthValue::negation(t);
        REQUIRE_NEAR(neg.strength, 0.7f, 1e-5f);
        REQUIRE_NEAR(neg.confidence, 0.9f, 1e-5f);
    });

    TEST(truth_value_deduction, {
        cog::prime::TruthValue ab(0.9f, 0.8f);
        cog::prime::TruthValue bc(0.8f, 0.7f);
        auto ac = cog::prime::TruthValue::deduction(ab, bc);
        REQUIRE(ac.strength > 0.0f && ac.strength <= 1.0f);
    });

    TEST(attention_value_stimulate, {
        cog::prime::AttentionValue av;
        av.stimulate(0.5f);
        REQUIRE(av.sti > 0.0f);
        REQUIRE(av.lti >= 0.0f);
    });

    TEST(attention_value_decay, {
        cog::prime::AttentionValue av(0.8f, 0.5f);
        av.decay(0.9f);
        REQUIRE(av.sti < 0.8f);
    });

    TEST(atomspace_add_node, {
        cog::prime::AtomSpace as;
        auto h = as.add_node(cog::AtomType::CONCEPT_NODE, "sky");
        REQUIRE(h != cog::UNDEFINED_HANDLE);
        REQUIRE(as.size() == 1);
    });

    TEST(atomspace_lookup, {
        cog::prime::AtomSpace as;
        as.add_node(cog::AtomType::CONCEPT_NODE, "water");
        auto h = as.lookup_node("water");
        REQUIRE(h != cog::UNDEFINED_HANDLE);
        REQUIRE(as.lookup_node("fire") == cog::UNDEFINED_HANDLE);
    });

    TEST(atomspace_add_link, {
        cog::prime::AtomSpace as;
        auto cat    = as.add_node(cog::AtomType::CONCEPT_NODE, "cat");
        auto animal = as.add_node(cog::AtomType::CONCEPT_NODE, "animal");
        auto link   = as.add_link(cog::AtomType::INHERITANCE_LINK, {cat, animal});
        REQUIRE(link != cog::UNDEFINED_HANDLE);
        REQUIRE(as.size() == 3);
    });

    TEST(atomspace_get_by_type, {
        cog::prime::AtomSpace as;
        as.add_node(cog::AtomType::CONCEPT_NODE, "n1");
        as.add_node(cog::AtomType::CONCEPT_NODE, "n2");
        as.add_node(cog::AtomType::PREDICATE_NODE, "pred");
        auto concepts = as.get_by_type(cog::AtomType::CONCEPT_NODE);
        REQUIRE(concepts.size() == 2);
    });

    TEST(atomspace_attention, {
        cog::prime::AtomSpace as;
        auto h = as.add_node(cog::AtomType::CONCEPT_NODE, "important");
        as.stimulate(h, 0.7f);
        auto* a = as.get(h);
        REQUIRE(a != nullptr);
        REQUIRE(a->av.sti > 0.0f);
        auto top = as.top_by_sti(1);
        REQUIRE(!top.empty());
        REQUIRE(top[0] == h);
    });

    TEST(declarative_memory, {
        cog::prime::DeclarativeMemory mem;
        auto h = mem.store_fact("Alice", "knows", "Bob");
        REQUIRE(h != cog::UNDEFINED_HANDLE);
        REQUIRE(mem.size() > 0);
    });

    TEST(episodic_memory, {
        cog::prime::EpisodicMemory mem;
        mem.record("office", "Alice arrived", 0.8f);
        mem.record("office", "Meeting started", 0.9f);
        mem.record("home",   "Dinner", 0.5f);
        auto recalled = mem.recall("office");
        REQUIRE(recalled.size() == 2);
        REQUIRE(mem.size() == 3);
    });

    TEST(procedural_memory, {
        cog::prime::ProceduralMemory mem;
        cog::prime::Procedure p("walk", "Locomotion skill", 0.3f);
        mem.store(p);
        REQUIRE(mem.size() == 1);
        mem.practice("walk");
        auto* pw = mem.get("walk");
        REQUIRE(pw != nullptr);
        REQUIRE(pw->invocations == 1);
        REQUIRE(pw->competence > 0.3f);
    });

    TEST(cognitive_cycle_run, {
        cog::prime::CognitiveCycle cycle;
        int perceive_count = 0;
        cycle.on_phase(cog::prime::CyclePhase::PERCEIVE,
            [&perceive_count](cog::prime::CognitiveState& st,
                              cog::prime::DeclarativeMemory&,
                              cog::prime::EpisodicMemory&,
                              cog::prime::ProceduralMemory&) {
                ++perceive_count;
                st.arousal = 0.6f;
            });
        cycle.run(3);
        REQUIRE(cycle.state().cycle_count == 3);
        REQUIRE(perceive_count == 3);
        REQUIRE_NEAR(cycle.state().arousal, 0.6f, 1e-5f);
    });

    TEST(ontogenetic_state, {
        cog::prime::OntogeneticState os;
        REQUIRE(os.level == cog::prime::OntogeneticLevel::SCAFFOLD);
        // Update multiple times to raise fitness above threshold
        for (int i = 0; i < 100; ++i) os.update(1.0f);
        bool advanced = os.try_advance(0.8f);
        REQUIRE(advanced);
        REQUIRE(os.level == cog::prime::OntogeneticLevel::REACTIVE);
    });
}

// ─── cog::webvm ───────────────────────────────────────────────────────────────

void test_webvm() {
    section("cog::webvm");

    TEST(sexpr_parser_atom, {
        cog::webvm::SParser p;
        auto e = p.parse("hello");
        REQUIRE(e != nullptr);
        REQUIRE(e->is_atom());
        REQUIRE(e->atom == "hello");
    });

    TEST(sexpr_parser_number, {
        cog::webvm::SParser p;
        auto e = p.parse("42.5");
        REQUIRE(e != nullptr);
        REQUIRE(e->is_number());
        REQUIRE_NEAR(e->number, 42.5, 1e-9);
    });

    TEST(sexpr_parser_string, {
        cog::webvm::SParser p;
        auto e = p.parse("\"hello world\"");
        REQUIRE(e != nullptr);
        REQUIRE(e->is_string());
        REQUIRE(e->atom == "hello world");
    });

    TEST(sexpr_parser_list, {
        cog::webvm::SParser p;
        auto e = p.parse("(a b c)");
        REQUIRE(e != nullptr);
        REQUIRE(e->is_list());
        REQUIRE(e->list.size() == 3);
        REQUIRE(e->list[0]->atom == "a");
    });

    TEST(sexpr_nested_list, {
        cog::webvm::SParser p;
        auto e = p.parse("(define x (+ 1 2))");
        REQUIRE(e != nullptr);
        REQUIRE(e->is_list());
        REQUIRE(e->list.size() == 3);
        REQUIRE(e->list[0]->atom == "define");
    });

    TEST(sexpr_to_string, {
        cog::webvm::SParser p;
        auto e = p.parse("(+ 1 2)");
        std::string s = e->to_string();
        REQUIRE(s.find("(") != std::string::npos);
        REQUIRE(s.find("+") != std::string::npos);
    });

    TEST(scheme_arithmetic, {
        cog::webvm::SchemeREPL repl;
        REQUIRE(repl.eval_str("(+ 1 2)") == "3");
        REQUIRE(repl.eval_str("(- 10 3)") == "7");
        REQUIRE(repl.eval_str("(* 4 5)") == "20");
        REQUIRE(repl.eval_str("(/ 10 2)") == "5");
    });

    TEST(scheme_define_and_use, {
        cog::webvm::SchemeREPL repl;
        repl.eval_str("(define x 42)");
        std::string result = repl.eval_str("x");
        REQUIRE(result == "42");
    });

    TEST(scheme_if, {
        cog::webvm::SchemeREPL repl;
        REQUIRE(repl.eval_str("(if #t 1 2)") == "1");
        REQUIRE(repl.eval_str("(if #f 1 2)") == "2");
    });

    TEST(scheme_list_ops, {
        cog::webvm::SchemeREPL repl;
        REQUIRE(repl.eval_str("(car (list 1 2 3))") == "1");
        REQUIRE(repl.eval_str("(null? (list))") == "#t");
        REQUIRE(repl.eval_str("(null? (list 1))") == "#f");
    });

    TEST(scheme_begin, {
        cog::webvm::SchemeREPL repl;
        std::string r = repl.eval_str("(begin (define y 10) (* y y))");
        REQUIRE(r == "100");
    });

    TEST(json_escape, {
        std::string s = cog::webvm::JSONSerializer::escape("hello \"world\"\n");
        REQUIRE(s.find("\\\"") != std::string::npos);
        REQUIRE(s.find("\\n") != std::string::npos);
    });

    TEST(json_sexpr, {
        cog::webvm::SParser p;
        auto e = p.parse("(a 1 \"hi\")");
        std::string j = cog::webvm::JSONSerializer::sexpr_to_json(*e);
        REQUIRE(j.find("[") != std::string::npos);
        REQUIRE(j.find("1") != std::string::npos);
    });

    TEST(json_object, {
        std::vector<std::pair<std::string, std::string>> kv = {
            {"name", cog::webvm::JSONSerializer::str("Alice")},
            {"age",  cog::webvm::JSONSerializer::num(30)}
        };
        std::string json = cog::webvm::JSONSerializer::make_object(kv);
        REQUIRE(json.find("\"name\"") != std::string::npos);
        REQUIRE(json.find("Alice") != std::string::npos);
        REQUIRE(json.find("30") != std::string::npos);
    });

    TEST(json_boolean, {
        REQUIRE(cog::webvm::JSONSerializer::boolean(true)  == "true");
        REQUIRE(cog::webvm::JSONSerializer::boolean(false) == "false");
    });
}

// ─── cog::fowler ──────────────────────────────────────────────────────────────

void test_fowler() {
    section("cog::fowler");

    // ── BalancedTernary: construction and conversion ──────────────────────────

    TEST(bt_zero, {
        cog::fowler::BalancedTernary z;
        REQUIRE(z.is_zero());
        REQUIRE(z.to_int() == 0);
        REQUIRE(z.to_string() == "0");
    });

    TEST(bt_from_int_positive, {
        cog::fowler::BalancedTernary bt(7);
        REQUIRE(bt.to_int() == 7);
        // 7 = 9 - 3 + 1 = 1*9 + T*3 + 1*1 → "1T1"
        REQUIRE(bt.to_string() == "1T1");
    });

    TEST(bt_from_int_negative, {
        cog::fowler::BalancedTernary bt(-7);
        REQUIRE(bt.to_int() == -7);
        // -7 is negation of 7="1T1" → "T1T"
        REQUIRE(bt.to_string() == "T1T");
    });

    TEST(bt_from_string, {
        cog::fowler::BalancedTernary bt("1T01");
        // 1T01 (MSB first): 1*27 + (-1)*9 + 0*3 + 1*1 = 27-9+0+1 = 19
        REQUIRE(bt.to_int() == 19);
    });

    TEST(bt_roundtrip, {
        for (int64_t v : {-100, -13, -1, 0, 1, 13, 100, 364}) {
            cog::fowler::BalancedTernary bt(v);
            REQUIRE(bt.to_int() == v);
        }
    });

    // ── BalancedTernary: arithmetic ───────────────────────────────────────────

    TEST(bt_negation, {
        cog::fowler::BalancedTernary a(5);
        cog::fowler::BalancedTernary b = -a;
        REQUIRE(b.to_int() == -5);
        REQUIRE((-b).to_int() == 5);
    });

    TEST(bt_addition, {
        REQUIRE((cog::fowler::BalancedTernary(3) + cog::fowler::BalancedTernary(4)).to_int() == 7);
        REQUIRE((cog::fowler::BalancedTernary(-3) + cog::fowler::BalancedTernary(3)).is_zero());
        REQUIRE((cog::fowler::BalancedTernary(13) + cog::fowler::BalancedTernary(-5)).to_int() == 8);
    });

    TEST(bt_subtraction, {
        REQUIRE((cog::fowler::BalancedTernary(10) - cog::fowler::BalancedTernary(3)).to_int() == 7);
        REQUIRE((cog::fowler::BalancedTernary(0) - cog::fowler::BalancedTernary(1)).to_int() == -1);
    });

    TEST(bt_multiply_by_trit, {
        cog::fowler::BalancedTernary bt(5);
        REQUIRE(bt.multiply_by_trit(cog::fowler::TRIT_POS).to_int() == 5);
        REQUIRE(bt.multiply_by_trit(cog::fowler::TRIT_ZERO).is_zero());
        REQUIRE(bt.multiply_by_trit(cog::fowler::TRIT_NEG).to_int() == -5);
    });

    TEST(bt_multiplication, {
        REQUIRE((cog::fowler::BalancedTernary(3) * cog::fowler::BalancedTernary(4)).to_int() == 12);
        REQUIRE((cog::fowler::BalancedTernary(-3) * cog::fowler::BalancedTernary(4)).to_int() == -12);
        REQUIRE((cog::fowler::BalancedTernary(7) * cog::fowler::BalancedTernary(7)).to_int() == 49);
        REQUIRE((cog::fowler::BalancedTernary(0) * cog::fowler::BalancedTernary(99)).is_zero());
    });

    TEST(bt_shift_left, {
        cog::fowler::BalancedTernary bt(2);
        // shift_left(1) = 2 * 3^1 = 6
        REQUIRE(bt.shift_left(1).to_int() == 6);
        // shift_left(3) = 2 * 3^3 = 54
        REQUIRE(bt.shift_left(3).to_int() == 54);
        // shift_left(0) = unchanged
        REQUIRE(bt.shift_left(0).to_int() == 2);
    });

    TEST(bt_division, {
        cog::fowler::BalancedTernary rem;
        cog::fowler::BalancedTernary q = cog::fowler::BalancedTernary(12).divmod(
            cog::fowler::BalancedTernary(3), rem);
        REQUIRE(q.to_int() == 4);
        REQUIRE(rem.is_zero());
    });

    TEST(bt_division_with_remainder, {
        cog::fowler::BalancedTernary rem;
        cog::fowler::BalancedTernary q = cog::fowler::BalancedTernary(13).divmod(
            cog::fowler::BalancedTernary(5), rem);
        // 13 = 5*2 + 3  or  13 = 5*3 + (-2); choose closest remainder → q=3, r=-2
        REQUIRE(q.to_int() * 5 + rem.to_int() == 13);
    });

    TEST(bt_comparison, {
        cog::fowler::BalancedTernary a(5), b(3), c(5);
        REQUIRE(a > b);
        REQUIRE(b < a);
        REQUIRE(a == c);
        REQUIRE(a != b);
        REQUIRE(a >= c);
        REQUIRE(b <= a);
    });

    // ── FowlerMachine: multiply ───────────────────────────────────────────────

    TEST(machine_multiply_basic, {
        cog::fowler::FowlerMachine m;
        auto result = m.multiply(cog::fowler::BalancedTernary(3),
                                 cog::fowler::BalancedTernary(4));
        REQUIRE(result.to_int() == 12);
        REQUIRE(!m.get_log().empty());
    });

    TEST(machine_multiply_negatives, {
        cog::fowler::FowlerMachine m;
        auto r1 = m.multiply(cog::fowler::BalancedTernary(-3),
                              cog::fowler::BalancedTernary(4));
        REQUIRE(r1.to_int() == -12);

        auto r2 = m.multiply(cog::fowler::BalancedTernary(-5),
                              cog::fowler::BalancedTernary(-7));
        REQUIRE(r2.to_int() == 35);
    });

    TEST(machine_multiply_by_zero, {
        cog::fowler::FowlerMachine m;
        auto result = m.multiply(cog::fowler::BalancedTernary(42),
                                 cog::fowler::BalancedTernary(0));
        REQUIRE(result.is_zero());
    });

    TEST(machine_multiply_larger, {
        cog::fowler::FowlerMachine m;
        // 13 × 7 = 91
        auto result = m.multiply(cog::fowler::BalancedTernary(13),
                                 cog::fowler::BalancedTernary(7));
        REQUIRE(result.to_int() == 91);
    });

    TEST(machine_event_log, {
        cog::fowler::FowlerMachine m;
        m.multiply(cog::fowler::BalancedTernary(5), cog::fowler::BalancedTernary(3));
        const auto& log = m.get_log();
        // Log must start with SET_MULTIPLICAND and SET_MULTIPLIER events
        REQUIRE(log.size() >= 2);
        REQUIRE(log[0].type == cog::fowler::MachineEventType::SET_MULTIPLICAND);
        REQUIRE(log[1].type == cog::fowler::MachineEventType::SET_MULTIPLIER);
        // Last event must be MULTIPLICATION_DONE
        REQUIRE(log.back().type == cog::fowler::MachineEventType::MULTIPLICATION_DONE);
    });

    // ── FowlerMachine: divide ─────────────────────────────────────────────────

    TEST(machine_divide_exact, {
        cog::fowler::FowlerMachine m;
        auto result = m.divide(cog::fowler::BalancedTernary(12),
                               cog::fowler::BalancedTernary(3));
        REQUIRE(result.to_int() == 4);
    });

    TEST(machine_divide_negative, {
        cog::fowler::FowlerMachine m;
        auto result = m.divide(cog::fowler::BalancedTernary(-12),
                               cog::fowler::BalancedTernary(4));
        REQUIRE(result.to_int() == -3);
    });

    // ── carry_normalize ───────────────────────────────────────────────────────

    TEST(carry_normalize, {
        // Raw: [2, 0, 0] = 2*1 = 2 (unnormalized: digit > 1)
        // After normalization: 2 = 3*1 - 1 → digit=-1, carry=1 → [-1, 1]
        std::vector<int> raw = {2, 0, 0};
        auto norm = cog::fowler::carry_normalize(raw);
        int64_t val = 0;
        int64_t pw = 1;
        for (int v : norm) { val += v * pw; pw *= 3; }
        REQUIRE(val == 2);
        for (int v : norm) { REQUIRE(v >= -1); REQUIRE(v <= 1); }
    });

    TEST(carry_normalize_negative, {
        // Raw: [-2, 0] → should normalize to [1, -1] (1 - 3 = -2)
        std::vector<int> raw = {-2, 0};
        auto norm = cog::fowler::carry_normalize(raw);
        int64_t val = 0;
        int64_t pw = 1;
        for (int v : norm) { val += v * pw; pw *= 3; }
        REQUIRE(val == -2);
        for (int v : norm) { REQUIRE(v >= -1); REQUIRE(v <= 1); }
    });
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== CogPy Unified Test Suite ===\n";

    test_core();
    test_plan9();
    test_pilot();
    test_mach();
    test_lux();
    test_glow();
    test_gml();
    test_prime();
    test_webvm();
    test_fowler();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return (tests_failed == 0) ? 0 : 1;
}
