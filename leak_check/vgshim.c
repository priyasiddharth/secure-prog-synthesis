/*
 * vgshim: a tiny bridge to Valgrind "client requests" so a Python process can,
 * from inside itself, (a) mark a secret buffer as UNDEFINED for memcheck taint
 * tracking, and (b) gate callgrind instrumentation so we count only the forward
 * pass rather than all of CPython startup/compile.
 *
 * All of these macros compile to special no-op instruction sequences that do
 * nothing unless the process is running under Valgrind -- so loading this .so in
 * a normal run is harmless.
 *
 * Build:  g++ -shared -fPIC -O0 vgshim.c -o vgshim.so
 * (requires the valgrind package, which ships the headers under
 *  /usr/include/valgrind/)
 */
#include <valgrind/valgrind.h>
#include <valgrind/memcheck.h>
#include <valgrind/callgrind.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- memcheck: secret-taint marking ------------------------------------ */
void vg_make_undefined(void *p, size_t n) { (void)VALGRIND_MAKE_MEM_UNDEFINED(p, n); }
void vg_make_defined(void *p, size_t n)   { (void)VALGRIND_MAKE_MEM_DEFINED(p, n); }

/* Emit a named marker into memcheck's error stream so we can attribute any
 * "depends on uninitialised value" report to the measured region. */
void vg_marker(const char *msg) { VALGRIND_PRINTF("%s\n", msg); }

/* ---- callgrind: gate instrumentation to the measured region ------------ */
void cg_start(void) { CALLGRIND_START_INSTRUMENTATION; CALLGRIND_ZERO_STATS; }
void cg_stop(void)  { CALLGRIND_STOP_INSTRUMENTATION; }
void cg_dump(void)  { CALLGRIND_DUMP_STATS; }

/* 1 if under Valgrind, 0 otherwise. Lets the harness fail loudly if the
 * instruments aren't actually active. */
int vg_running(void) { return (int)RUNNING_ON_VALGRIND; }

#ifdef __cplusplus
}
#endif
