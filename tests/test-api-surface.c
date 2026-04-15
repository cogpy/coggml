#include "../ggml.h"
#include <stdio.h>
#include <string.h>

// A simple testing framework
#define TEST_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", #condition, __FILE__, __LINE__); \
            return 1; \
        } \
    } while (0)

int test_ggml_init_and_free() {
    struct ggml_init_params params = { 16 * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL);
    ggml_free(ctx);
    return 0;
}

int test_ggml_new_tensor() {
    struct ggml_init_params params = { 16 * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
    TEST_ASSERT(t != NULL);
    TEST_ASSERT(ggml_nelements(t) == 100);
    TEST_ASSERT(ggml_nbytes(t) == 100 * sizeof(float));
    ggml_free(ctx);
    return 0;
}

int test_ggml_add() {
    struct ggml_init_params params = { 16 * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_f32(a, 2.0f);
    ggml_set_f32(b, 3.0f);

    struct ggml_tensor * c = ggml_add(ctx, a, b);
    struct ggml_cgraph graph = ggml_build_forward(c);
    ggml_graph_compute_with_ctx(ctx, &graph, 1);

    TEST_ASSERT(ggml_get_f32_1d(c, 0) == 5.0f);

    ggml_free(ctx);
    return 0;
}

int main(void) {
    int result = 0;
    result |= test_ggml_init_and_free();
    result |= test_ggml_new_tensor();
    result |= test_ggml_add();

    if (result == 0) {
        printf("All API tests passed!\n");
    }

    return result;
}
