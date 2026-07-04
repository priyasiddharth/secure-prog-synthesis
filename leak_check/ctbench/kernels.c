/*
 * Kernels for the toolchain leakage testbed. Each consumes a SECRET buffer and
 * produces a sink. Two secret classes A/B (see run_ct.py) differ only in the
 * protected values; a data-oblivious kernel must execute identically on both.
 *
 * These are written to expose how the SAME source lowers differently under
 * different (compiler, flags): a source-level branch may survive as a real
 * data-dependent branch, or be vectorized into a branchless blend/cmov.
 *
 * Kept in a separate TU and marked noinline so the optimizer can't fold a kernel
 * into the harness/const-propagate the secret away.
 */
#include <stddef.h>
#include <stdint.h>

#define NOINLINE __attribute__((noinline))

/* Control: dense matvec y = x . W. Oblivious at every opt level (same FLOPs
 * regardless of values). n = DIM, W is n*n, x is n, y is n. */
NOINLINE void k_matmul(const float *W, const float *x, float *y, size_t n) {
    for (size_t j = 0; j < n; j++) {
        float acc = 0.f;
        for (size_t i = 0; i < n; i++)
            acc += x[i] * W[i * n + j];
        y[j] = acc;
    }
}

/* relu written as a branch on sign. Secret = sign pattern of `in`.
 * Question: does -O3 lower the ?: to a branchless max(x,0)? */
NOINLINE void k_relu(const float *in, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (in[i] > 0.f)      /* data-dependent branch in the source */
            out[i] = in[i];
        else
            out[i] = 0.f;
    }
}

/* Naive select via if. Secret = the mask. Question: does vectorization turn the
 * per-element branch into a blend (removing the control-flow leak)? */
NOINLINE void k_select_branch(const uint8_t *mask, const float *a,
                              const float *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (mask[i])
            out[i] = a[i];
        else
            out[i] = b[i];
    }
}

/* Constant-time select: out = b ^ ((a ^ b) & (-mask)). No branch in the source.
 * Question: does any compiler/flag REINTRODUCE a branch (the killer case)? */
NOINLINE void k_select_ct(const uint8_t *mask, const uint32_t *a,
                          const uint32_t *b, uint32_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t m = (uint32_t)0 - (uint32_t)(mask[i] & 1); /* 0x00.. or 0xff.. */
        out[i] = b[i] ^ ((a[i] ^ b[i]) & m);
    }
}

/* Byte compare with early return on first mismatch. Secret = position of first
 * mismatch. An irreducible source-level timing leak: must leak at every build. */
NOINLINE int k_memeq_earlyexit(const uint8_t *secret, const uint8_t *guess,
                               size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (secret[i] != guess[i])
            return 0;          /* early exit -> time depends on match length */
    }
    return 1;
}
