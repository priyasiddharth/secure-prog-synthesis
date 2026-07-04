/*
 * Driver for one kernel invocation under a Valgrind instrument.
 *
 *   harness <kernel> <secret_file> <count|taint>
 *
 * The secret is read from a file so the code path is identical regardless of the
 * secret's contents. Public inputs are fixed constants generated here (never
 * depend on the secret). In `count` mode callgrind instrumentation is gated to
 * the kernel call; in `taint` mode the secret bytes are marked UNDEFINED so
 * memcheck reports any control-flow/address dependence on them.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <valgrind/callgrind.h>
#include <valgrind/memcheck.h>

/* kernels.c */
void k_matmul(const float *W, const float *x, float *y, size_t n);
void k_relu(const float *in, float *out, size_t n);
void k_select_branch(const uint8_t *mask, const float *a, const float *b, float *out, size_t n);
void k_select_ct(const uint8_t *mask, const uint32_t *a, const uint32_t *b, uint32_t *out, size_t n);
int  k_memeq_earlyexit(const uint8_t *secret, const uint8_t *guess, size_t n);

#define MM_N 256      /* matmul dimension (kept small: valgrind is slow) */
#define EL_N 4096     /* elementwise length */

volatile uint64_t g_sink;   /* prevent dead-code elimination of outputs */

static uint8_t *read_file(const char *path, size_t *out_sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(2); }
    fclose(f);
    *out_sz = (size_t)sz;
    return buf;
}

static void expect(size_t got, size_t want) {
    if (got != want) { fprintf(stderr, "secret size %zu != expected %zu\n", got, want); exit(2); }
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <kernel> <secret_file> <count|taint>\n", argv[0]); return 2; }
    const char *kern = argv[1];
    size_t sz;
    uint8_t *secret = read_file(argv[2], &sz);
    int taint = (strcmp(argv[3], "taint") == 0);

    /* mark_secret marks exactly the secret bytes UNDEFINED (taint mode only) */
    #define BEGIN() do { if (taint) { VALGRIND_PRINTF("taint region begin\n"); \
                          VALGRIND_MAKE_MEM_UNDEFINED(secret, sz); } \
                         else CALLGRIND_START_INSTRUMENTATION; } while (0)
    #define END(outp, outn) do { if (taint) { VALGRIND_MAKE_MEM_DEFINED(outp, outn); \
                          VALGRIND_MAKE_MEM_DEFINED(secret, sz); VALGRIND_PRINTF("taint region end\n"); } \
                         else CALLGRIND_STOP_INSTRUMENTATION; } while (0)

    if (!strcmp(kern, "matmul")) {
        expect(sz, (size_t)MM_N * MM_N * sizeof(float));
        const float *W = (const float *)secret;
        float *x = malloc(MM_N * sizeof(float)), *y = malloc(MM_N * sizeof(float));
        for (int i = 0; i < MM_N; i++) x[i] = 1.0f;
        BEGIN();
        k_matmul(W, x, y, MM_N);
        END(y, MM_N * sizeof(float));
        g_sink += (uint64_t)y[0];
    } else if (!strcmp(kern, "relu")) {
        expect(sz, (size_t)EL_N * sizeof(float));
        const float *in = (const float *)secret;
        float *out = malloc(EL_N * sizeof(float));
        BEGIN();
        k_relu(in, out, EL_N);
        END(out, EL_N * sizeof(float));
        g_sink += (uint64_t)out[0];
    } else if (!strcmp(kern, "select_branch")) {
        expect(sz, (size_t)EL_N);
        float *a = malloc(EL_N * sizeof(float)), *b = malloc(EL_N * sizeof(float)), *out = malloc(EL_N * sizeof(float));
        for (int i = 0; i < EL_N; i++) { a[i] = 2.0f; b[i] = 3.0f; }
        BEGIN();
        k_select_branch(secret, a, b, out, EL_N);
        END(out, EL_N * sizeof(float));
        g_sink += (uint64_t)out[0];
    } else if (!strcmp(kern, "select_ct")) {
        expect(sz, (size_t)EL_N);
        uint32_t *a = malloc(EL_N * 4), *b = malloc(EL_N * 4), *out = malloc(EL_N * 4);
        for (int i = 0; i < EL_N; i++) { a[i] = 0x11111111u; b[i] = 0x22222222u; }
        BEGIN();
        k_select_ct(secret, a, b, out, EL_N);
        END(out, EL_N * 4);
        g_sink += out[0];
    } else if (!strcmp(kern, "memeq")) {
        expect(sz, (size_t)EL_N);
        uint8_t *guess = malloc(EL_N);
        for (int i = 0; i < EL_N; i++) guess[i] = 0xAA;
        BEGIN();
        int r = k_memeq_earlyexit(secret, guess, EL_N);
        END(&r, sizeof(int));
        g_sink += (uint64_t)r;
    } else {
        fprintf(stderr, "unknown kernel %s\n", kern);
        return 2;
    }
    return 0;
}
