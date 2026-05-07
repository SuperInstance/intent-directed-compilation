#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

#define N 10000000

typedef struct { double value, lower, upper, stakes; } AoSConstraint;
typedef enum { P_INT8, P_INT16, P_INT32, P_DUAL } PrecClass;

PrecClass classify(double stakes, double range) {
    if (stakes > 0.75) return P_DUAL;
    if (stakes > 0.5 || range > 32000) return P_INT32;
    if (stakes > 0.25 || range > 127) return P_INT16;
    return P_INT8;
}

int main() {
    printf("=== TWO-PASS PIPELINE (sequential classify + bulk copy) ===\n\n");
    
    // Generate with AV mix distribution (75% INT8, 15% INT16, 8% INT32, 2% DUAL)
    AoSConstraint* input = aligned_alloc(64, N * sizeof(AoSConstraint));
    uint8_t* prec_class = aligned_alloc(64, N);  // classification byte array
    srand(42);
    for (int i = 0; i < N; i++) {
        int r = rand() % 100;
        double stakes;
        double lo, hi, v;
        if (r < 75) {
            // INT8-safe: small range, low stakes
            stakes = 0.05; lo = rand()%100; hi = lo + rand()%20 + 1; v = lo + rand()%(int)(hi-lo+1);
        } else if (r < 90) {
            // INT16: medium range
            stakes = 0.3; lo = rand()%3000-1500; hi = lo + rand()%500+1; v = lo + rand()%(int)(hi-lo+1);
        } else if (r < 98) {
            // INT32: large range or high stakes
            stakes = 0.6; lo = rand()%100000-50000; hi = lo + rand()%30000+1; v = lo + rand()%(int)(hi-lo+1);
        } else {
            // DUAL: critical
            stakes = 0.9; lo = rand()%100000-50000; hi = lo + rand()%30000+1; v = lo + rand()%(int)(hi-lo+1);
        }
        input[i] = (AoSConstraint){(double)v, (double)lo, (double)hi, stakes};
    }
    
    // Allocate SoA buffers (sized for worst case = N each)
    int8_t*  sv8 = aligned_alloc(64, N); int8_t*  slo8 = aligned_alloc(64, N); int8_t*  shi8 = aligned_alloc(64, N); int* si8 = malloc(N*sizeof(int));
    int16_t* sv16 = aligned_alloc(64, N*2); int16_t* slo16 = aligned_alloc(64, N*2); int16_t* shi16 = aligned_alloc(64, N*2); int* si16 = malloc(N*sizeof(int));
    int32_t* sv32 = aligned_alloc(64, N*4); int32_t* slo32 = aligned_alloc(64, N*4); int32_t* shi32 = aligned_alloc(64, N*4); int* si32 = malloc(N*sizeof(int));
    int32_t* svd = aligned_alloc(64, N*4); int32_t* slod = aligned_alloc(64, N*4); int32_t* shid = aligned_alloc(64, N*4); int* sid = malloc(N*sizeof(int));
    uint8_t* results = calloc(N, 1);
    uint8_t* baseline = calloc(N, 1);
    
    volatile int sink = 0;
    uint64_t t0;
    
    // BASELINE: scalar INT32
    t0 = rdtsc();
    for (int i = 0; i < N; i++) {
        baseline[i] = !(input[i].value >= input[i].lower && input[i].value <= input[i].upper);
    }
    uint64_t t_base = rdtsc() - t0;
    
    // === PASS 1: Classify all constraints (sequential write to prec_class[]) ===
    t0 = rdtsc();
    int counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < N; i++) {
        double range = input[i].upper - input[i].lower;
        PrecClass pc = classify(input[i].stakes, range);
        prec_class[i] = (uint8_t)pc;
        counts[pc]++;
    }
    uint64_t t_pass1 = rdtsc() - t0;
    
    // === PASS 2: Counting sort into SoA buffers ===
    int pos[4] = {0, 0, 0, 0};
    t0 = rdtsc();
    for (int i = 0; i < N; i++) {
        int idx;
        switch (prec_class[i]) {
            case P_INT8:
                idx = pos[P_INT8]++;
                sv8[idx] = (int8_t)input[i].value;
                slo8[idx] = (int8_t)input[i].lower;
                shi8[idx] = (int8_t)input[i].upper;
                si8[idx] = i;
                break;
            case P_INT16:
                idx = pos[P_INT16]++;
                sv16[idx] = (int16_t)input[i].value;
                slo16[idx] = (int16_t)input[i].lower;
                shi16[idx] = (int16_t)input[i].upper;
                si16[idx] = i;
                break;
            case P_INT32:
                idx = pos[P_INT32]++;
                sv32[idx] = (int32_t)input[i].value;
                slo32[idx] = (int32_t)input[i].lower;
                shi32[idx] = (int32_t)input[i].upper;
                si32[idx] = i;
                break;
            case P_DUAL:
                idx = pos[P_DUAL]++;
                svd[idx] = (int32_t)input[i].value;
                slod[idx] = (int32_t)input[i].lower;
                shid[idx] = (int32_t)input[i].upper;
                sid[idx] = i;
                break;
        }
    }
    uint64_t t_pass2 = rdtsc() - t0;
    
    // === PHASE 3: AVX-512 check (same as before) ===
    t0 = rdtsc();
    int n8 = pos[P_INT8], n16 = pos[P_INT16], n32 = pos[P_INT32], nd = pos[P_DUAL];
    
    for (int i = 0; i <= n8-64; i += 64) {
        __m512i vv = _mm512_loadu_si512((void*)(sv8+i));
        __m512i ll = _mm512_loadu_si512((void*)(slo8+i));
        __m512i hh = _mm512_loadu_si512((void*)(shi8+i));
        __mmask64 k = _mm512_cmpge_epi8_mask(vv, ll) & _mm512_cmple_epi8_mask(vv, hh);
        sink += __builtin_popcountll(k);
    }
    for (int i = 0; i <= n16-32; i += 32) {
        __m512i vv = _mm512_loadu_si512((void*)(sv16+i));
        __m512i ll = _mm512_loadu_si512((void*)(slo16+i));
        __m512i hh = _mm512_loadu_si512((void*)(shi16+i));
        __mmask32 k = _mm512_cmpge_epi16_mask(vv, ll) & _mm512_cmple_epi16_mask(vv, hh);
        sink += __builtin_popcount(k);
    }
    for (int i = 0; i <= n32-16; i += 16) {
        __m512i vv = _mm512_loadu_si512((void*)(sv32+i));
        __m512i ll = _mm512_loadu_si512((void*)(slo32+i));
        __m512i hh = _mm512_loadu_si512((void*)(shi32+i));
        __mmask16 k = _mm512_cmpge_epi32_mask(vv, ll) & _mm512_cmple_epi32_mask(vv, hh);
        sink += __builtin_popcount(k);
    }
    __m512i sign = _mm512_set1_epi32(0x80000000);
    for (int i = 0; i <= nd-16; i += 16) {
        __m512i vv = _mm512_loadu_si512((void*)(svd+i));
        __m512i ll = _mm512_loadu_si512((void*)(slod+i));
        __m512i hh = _mm512_loadu_si512((void*)(shid+i));
        __mmask16 ka = _mm512_cmpge_epi32_mask(vv, ll) & _mm512_cmple_epi32_mask(vv, hh);
        __mmask16 kb = _mm512_cmpge_epu32_mask(_mm512_xor_si512(vv,sign), _mm512_xor_si512(ll,sign)) &
                       _mm512_cmple_epu32_mask(_mm512_xor_si512(vv,sign), _mm512_xor_si512(hh,sign));
        sink += __builtin_popcount(ka & kb);
    }
    uint64_t t_check = rdtsc() - t0;
    
    printf("%-30s %12s %12s %10s\n", "Phase", "Cycles", "cyc/constr", "% total");
    printf("=============================================================\n");
    uint64_t total = t_pass1 + t_pass2 + t_check;
    printf("%-30s %12lu %12.4f %10s\n", "Baseline (scalar)", t_base, (double)t_base/N, "");
    printf("%-30s %12lu %12.4f %9.1f%%\n", "Pass 1: Classify", t_pass1, (double)t_pass1/N, 100.0*t_pass1/total);
    printf("%-30s %12lu %12.4f %9.1f%%\n", "Pass 2: Counting sort", t_pass2, (double)t_pass2/N, 100.0*t_pass2/total);
    printf("%-30s %12lu %12.4f %9.1f%%\n", "Phase 3: AVX-512 check", t_check, (double)t_check/N, 100.0*t_check/total);
    printf("%-30s %12lu %12.4f\n", "TOTAL PIPELINE", total, (double)total/N);
    
    printf("\n=== RESULT ===\n");
    printf("Baseline: %lu cycles (%.2f cyc/constraint)\n", t_base, (double)t_base/N);
    printf("Pipeline: %lu cycles (%.2f cyc/constraint)\n", total, (double)total/N);
    printf("Speedup:  %.2fx\n", (double)t_base/total);
    
    printf("\nDistribution: INT8=%d (%.0f%%), INT16=%d (%.0f%%), INT32=%d (%.0f%%), DUAL=%d (%.0f%%)\n",
        n8, 100.0*n8/N, n16, 100.0*n16/N, n32, 100.0*n32/N, nd, 100.0*nd/N);
    printf("(void)sink=%d\n", sink);
    
    return 0;
}
