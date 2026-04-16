#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#include <HAP_farf.h>

static inline void hex_dump_int8_line(char * pref, const int8_t * x, int n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    int written = snprintf(p, (size_t)(p_end - p), "%s: ", pref);
    if (written < 0 || (size_t) written >= (size_t) (p_end - p)) {
        p = p_end - 1;
        *p = '\0';
    } else {
        p += written;
    }
    for (int i = 0; i < n && p < p_end; i++) {
        size_t rem = (size_t) (p_end - p);
        written = snprintf(p, rem, "%d, ", x[i]);
        if (written < 0 || (size_t) written >= rem) {
            p = p_end - 1;
            *p = '\0';
            break;
        }
        p += written;
    }
    FARF(HIGH, "%s\n", str);
}

static inline void hex_dump_uint8_line(char * pref, const uint8_t * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    int written = snprintf(p, (size_t)(p_end - p), "%s: ", pref);
    if (written < 0 || (size_t) written >= (size_t) (p_end - p)) {
        p = p_end - 1;
        *p = '\0';
    } else {
        p += written;
    }
    for (int i = 0; i < n && p < p_end; i++) {
        size_t rem = (size_t) (p_end - p);
        written = snprintf(p, rem, "%d, ", x[i]);
        if (written < 0 || (size_t) written >= rem) {
            p = p_end - 1;
            *p = '\0';
            break;
        }
        p += written;
    }
    FARF(HIGH, "%s\n", str);
}

static inline void hex_dump_uint32_line(char * pref, const uint32_t * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    int written = snprintf(p, (size_t)(p_end - p), "%s: ", pref);
    if (written < 0 || (size_t) written >= (size_t) (p_end - p)) {
        p = p_end - 1;
        *p = '\0';
    } else {
        p += written;
    }
    for (int i = 0; i < n && p < p_end; i++) {
        size_t rem = (size_t) (p_end - p);
        written = snprintf(p, rem, "%u, ", (unsigned int) x[i]);
        if (written < 0 || (size_t) written >= rem) {
            p = p_end - 1;
            *p = '\0';
            break;
        }
        p += written;
    }
    FARF(HIGH, "%s\n", str);
}

static inline void hex_dump_int32_line(char * pref, const int32_t * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    int written = snprintf(p, (size_t)(p_end - p), "%s: ", pref);
    if (written < 0 || (size_t) written >= (size_t) (p_end - p)) {
        p = p_end - 1;
        *p = '\0';
    } else {
        p += written;
    }
    for (int i = 0; i < n && p < p_end; i++) {
        size_t rem = (size_t) (p_end - p);
        written = snprintf(p, rem, "%d, ", (int) x[i]);
        if (written < 0 || (size_t) written >= rem) {
            p = p_end - 1;
            *p = '\0';
            break;
        }
        p += written;
    }
    FARF(HIGH, "%s\n", str);
}

static inline void hex_dump_f16_line(char * pref, const __fp16 * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    int written = snprintf(p, (size_t)(p_end - p), "%s: ", pref);
    if (written < 0 || (size_t) written >= (size_t) (p_end - p)) {
        p = p_end - 1;
        *p = '\0';
    } else {
        p += written;
    }
    for (int i = 0; i < n && p < p_end; i++) {
        size_t rem = (size_t) (p_end - p);
        written = snprintf(p, rem, "%.6f, ", (float) x[i]);
        if (written < 0 || (size_t) written >= rem) {
            p = p_end - 1;
            *p = '\0';
            break;
        }
        p += written;
    }
    FARF(HIGH, "%s\n", str);
}

static inline void hex_dump_f32_line(char * pref, const float * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    int written = snprintf(p, (size_t)(p_end - p), "%s: ", pref);
    if (written < 0 || (size_t) written >= (size_t) (p_end - p)) {
        p = p_end - 1;
        *p = '\0';
    } else {
        p += written;
    }
    for (int i = 0; i < n && p < p_end; i++) {
        size_t rem = (size_t) (p_end - p);
        written = snprintf(p, rem, "%.6f, ", x[i]);
        if (written < 0 || (size_t) written >= rem) {
            p = p_end - 1;
            *p = '\0';
            break;
        }
        p += written;
    }
    FARF(HIGH, "%s\n", str);
}

static inline void hex_dump_f32(char * pref, const float * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    for (; i < n0; i++) {
        hex_dump_f32_line(pref, x + (16 * i), 16);
    }
    if (n1) {
        hex_dump_f32_line(pref, x + (16 * i), n1);
    }
}

static inline void hex_dump_f16(char * pref, const __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    for (; i < n0; i++) {
        hex_dump_f16_line(pref, x + (16 * i), 16);
    }
    if (n1) {
        hex_dump_f16_line(pref, x + (16 * i), n1);
    }
}

#endif /* HEX_DUMP_H */
