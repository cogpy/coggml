#include "ggml-opencog.h"
#include <iostream>
int main() {
    std::cout << "OpenCog GGML Demo\n";
    auto* atomspace = ggml_opencog_atomspace_new(64);
    std::cout << "AtomSpace created!\n";
    ggml_opencog_atomspace_free(atomspace);
    return 0;
}
