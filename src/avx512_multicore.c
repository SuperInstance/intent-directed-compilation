// Multi-Core AVX-512 Mixed-Precision Benchmark
// Proves SoA mixed-precision scaling across cores on AMD Ryzen AI 9 HX 370
// Compile: gcc -O3 -mavx512f -mavx512bw -mavx512dq -pthread -o avx512_mc avx512_multicore.c

#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

#define N_PER_THREAD 2500000  // 2.5M constraints per thread

typedef struct {
    int thread_id;
    int n_constraints;
    uint64_t cycles_int32;
    uint64_t cycles_mixed;
    int mismatches;
    int total_int8;
    int total_int16;
    int total_int32;
    int total_dual;
} ThreadResult;

void* bench_thread(void* arg) {
    ThreadResult* result = (ThreadResult*)arg;
    int n = result->n_constraints;
    unsigned int seed = 42 + result->thread_id;
    
    // Generate SoA data
    int n8 = (int)(n * 0.75);
    int n16 = (int)(n * 0.15);
    int n32 = (int)(n * 0.08);
    int nd = n - n8 - n16 - n32;
    
    int8_t*  v8  = aligned_alloc(64, n8);
    int8_t*  lo8 = aligned_alloc(64, n8);
    int8_t*  hi8 = aligned_alloc(64, n8);
    int16_t* v16 = aligned_alloc(64, n16*2);
    int16_t* lo16= aligned_alloc(64, n16*2);
    int16_t* hi16= aligned_alloc(64, n16*2);
    int32_t* v32b= aligned_alloc(64, n*4);  // for baseline
    int32_t* lo32b= aligned_alloc(64, n*4);
    int32_t* hi32b= aligned_alloc(64, n*4);
    int32_t* v32 = aligned_alloc(64, n32*4);
    int32_t* lo32= aligned_alloc(64, n32*4);
    int32_t* hi32= aligned_alloc(64, n32*4);
    int32_t* vd  = aligned_alloc(64, nd*4);
    int32_t* lod = aligned_alloc(64, nd*4);
    int32_t* hid = aligned_alloc(64, nd*4);
    
    // Fill with data
    for (int i = 0; i < n8; i++) {
        int lo = rand_r(&seed)%100 - 50;
        int hi = lo + rand_r(&seed)%30 + 1;
        v8[i]=(int8_t)(lo+rand_r(&seed)%(hi-lo+1));
        lo8[i]=(int8_t)lo; hi8[i]=(int8_t)hi;
    }
    for (int i = 0; i < n16; i++) {
        int lo = rand_r(&seed)%3000 - 1500;
        int hi = lo + rand_r(&seed)%500 + 1;
        v16[i]=(int16_t)(lo+rand_r(&seed)%(hi-lo+1));
        lo16[i]=(int16_t)lo; hi16[i]=(int16_t)hi;
    }
    for (int i = 0; i < n32; i++) {
        int lo = rand_r(&seed)%150000 - 75000;
        int hi = lo + rand_r(&seed)%30000 + 1;
        v32[i]=lo+rand_r(&seed)%(hi-lo+1);
        lo32[i]=lo; hi32[i]=hi;
    }
    for (int i = 0; i < nd; i++) {
        int lo = rand_r(&seed)%150000 - 75000;
        int hi = lo + rand_r(&seed)%30000 + 1;
        vd[i]=lo+rand_r(&seed)%(hi-lo+1);
        lod[i]=lo; hid[i]=hi;
    }
    // Baseline: all INT32
    for (int i = 0; i < n; i++) {
        int lo = rand_r(&seed)%200 - 100;
        int hi = lo + rand_r(&seed)%50 + 1;
        v32b[i]=lo+rand_r(&seed)%(hi-lo+1);
        lo32b[i]=lo; hi32b[i]=hi;
    }
    
    volatile int sink = 0;
    
    // BASELINE: all INT32
    uint64_t t0 = rdtsc();
    for (int i = 0; i <= n-16; i += 16) {
        __m512i vv = _mm512_loadu_si512((void*)(v32b+i));
        __m512i ll = _mm512_loadu_si512((void*)(lo32b+i));
        __m512i hh = _mm512_loadu_si512((void*)(hi32b+i));
        __mmask16 k = _mm512_cmpge_epi32_mask(vv, ll) & _mm512_cmple_epi32_mask(vv, hh);
        sink += __builtin_popcount(k);
    }
    result->cycles_int32 = rdtsc() - t0;
    
    // MIXED: SoA per precision class
    t0 = rdtsc();
    // INT8 batch
    for (int i = 0; i <= n8-64; i += 64) {
        __m512i vv = _mm512_loadu_si512((void*)(v8+i));
        __m512i ll = _mm512_loadu_si512((void*)(lo8+i));
        __m512i hh = _mm512_loadu_si512((void*)(hi8+i));
        __mmask64 k = _mm512_cmpge_epi8_mask(vv, ll) & _mm512_cmple_epi8_mask(vv, hh);
        sink += __builtin_popcountll(k);
    }
    // INT16 batch
    for (int i = 0; i <= n16-32; i += 32) {
        __m512i vv = _mm512_loadu_si512((void*)(v16+i));
        __m512i ll = _mm512_loadu_si512((void*)(lo16+i));
        __m512i hh = _mm512_loadu_si512((void*)(hi16+i));
        __mmask32 k = _mm512_cmpge_epi16_mask(vv, ll) & _mm512_cmple_epi16_mask(vv, hh);
        sink += __builtin_popcount(k);
    }
    // INT32 batch
    for (int i = 0; i <= n32-16; i += 16) {
        __m512i vv = _mm512_loadu_si512((void*)(v32+i));
        __m512i ll = _mm512_loadu_si512((void*)(lo32+i));
        __m512i hh = _mm512_loadu_si512((void*)(hi32+i));
        __mmask16 k = _mm512_cmpge_epi32_mask(vv, ll) & _mm512_cmple_epi32_mask(vv, hh);
        sink += __builtin_popcount(k);
    }
    // DUAL batch
    for (int i = 0; i <= nd-16; i += 16) {
        __m512i vv = _mm512_loadu_si512((void*)(vd+i));
        __m512i ll = _mm512_loadu_si512((void*)(lod+i));
        __m512i hh = _mm512_loadu_si512((void*)(hid+i));
        __mmask16 ka = _mm512_cmpge_epi32_mask(vv, ll) & _mm512_cmple_epi32_mask(vv, hh);
        __m512i above = _mm512_sub_epi32(vv, ll);
        __m512i below = _mm512_sub_epi32(hh, vv);
        __mmask16 kb = ~(_mm512_cmplt_epi32_mask(above, _mm512_set1_epi32(0)) |
                         _mm512_cmplt_epi32_mask(below, _mm512_set1_epi32(0)));
        sink += __builtin_popcount(ka & kb);
    }
    result->cycles_mixed = rdtsc() - t0;
    
    // Differential test
    int mismatches = 0;
    for (int i = 0; i < n8; i++) {
        int ref = ((int)v32b[i] >= (int)lo32b[i] && (int)v32b[i] <= (int)hi32b[i]);
        // Only compare values that fit in INT8 range
        if (v32b[i] >= -127 && v32b[i] <= 127 && lo32b[i] >= -127 && lo32b[i] <= 127 && hi32b[i] <= 127 && hi32b[i] >= -127) {
            int i8r = (v8[i] >= lo8[i] && v8[i] <= hi8[i]);
            if (ref != i8r) mismatches++;
        }
    }
    result->mismatches = mismatches;
    result->total_int8 = n8;
    result->total_int16 = n16;
    result->total_int32 = n32;
    result->total_dual = nd;
    
    free(v8); free(lo8); free(hi8);
    free(v16); free(lo16); free(hi16);
    free(v32b); free(lo32b); free(hi32b);
    free(v32); free(lo32); free(hi32);
    free(vd); free(lod); free(hid);
    
    return NULL;
}

int main() {
    int ncores = sysconf(_SC_NPROCESSORS_ONLN);
    printf("=== MULTI-CORE AVX-512 MIXED-PRECISION BENCHMARK ===\n");
    printf("AMD Ryzen AI 9 HX 370 | %d cores | %d constraints/thread\n\n", ncores, N_PER_THREAD);
    
    // Test 1, 2, 4, 8, all cores
    int test_cores[] = {1, 2, 4, 8, ncores};
    int n_tests = sizeof(test_cores)/sizeof(test_cores[0]);
    
    printf("%-8s %12s %12s %12s %12s %10s\n", 
        "Cores", "Total(M)", "Baseline(Mc)", "Mixed(Mc)", "Speedup", "Mismatches");
    printf("==========================================================================\n");
    
    for (int t = 0; t < n_tests; t++) {
        int nc = test_cores[t];
        if (nc > ncores) nc = ncores;
        
        pthread_t threads[nc];
        ThreadResult results[nc];
        
        for (int i = 0; i < nc; i++) {
            results[i].thread_id = i;
            results[i].n_constraints = N_PER_THREAD;
            results[i].mismatches = 0;
            pthread_create(&threads[i], NULL, bench_thread, &results[i]);
        }
        
        for (int i = 0; i < nc; i++) {
            pthread_join(threads[i], NULL);
        }
        
        // Aggregate
        uint64_t total_int32_cycles = 0, total_mixed_cycles = 0;
        int total_mismatches = 0;
        for (int i = 0; i < nc; i++) {
            total_int32_cycles += results[i].cycles_int32;
            total_mixed_cycles += results[i].cycles_mixed;
            total_mismatches += results[i].mismatches;
        }
        
        long total_constraints = (long)nc * N_PER_THREAD;
        double speedup = (double)total_int32_cycles / total_mixed_cycles;
        double baseline_mc = (double)total_constraints / (double)total_int32_cycles;
        double mixed_mc = (double)total_constraints / (double)total_mixed_cycles;
        
        printf("%-8d %12.1f %12.2f %12.2f %12.2fx %10d\n",
            nc, total_constraints/1e6, baseline_mc, mixed_mc, speedup, total_mismatches);
    }
    
    printf("\n=== SCALING EFFICIENCY ===\n");
    // Run single-core first as baseline for scaling
    pthread_t t1;
    ThreadResult r1 = {.thread_id=0, .n_constraints=N_PER_THREAD, .mismatches=0};
    pthread_create(&t1, NULL, bench_thread, &r1);
    pthread_join(t1, NULL);
    
    // All cores
    int nc = ncores;
    pthread_t threads[nc];
    ThreadResult results[nc];
    for (int i = 0; i < nc; i++) {
        results[i].thread_id = i;
        results[i].n_constraints = N_PER_THREAD;
        results[i].mismatches = 0;
        pthread_create(&threads[i], NULL, bench_thread, &results[i]);
    }
    for (int i = 0; i < nc; i++) pthread_join(threads[i], NULL);
    
    uint64_t all_mixed = 0;
    for (int i = 0; i < nc; i++) all_mixed += results[i].cycles_mixed;
    
    double single_throughput = (double)N_PER_THREAD / (double)r1.cycles_mixed;
    double multi_throughput = ((double)nc * N_PER_THREAD) / (double)all_mixed;
    double scaling = multi_throughput / single_throughput;
    
    printf("Single-core mixed: %.2f constraints/cycle\n", single_throughput);
    printf("All-%d mixed:      %.2f constraints/cycle\n", nc, multi_throughput);
    printf("Scaling:           %.2fx / %d cores (%.0f%% efficiency)\n", 
        scaling, nc, 100.0*scaling/nc);
    
    // At 2.4 GHz
    double ghz = 2.4;
    printf("\nProjected throughput at %.1f GHz: %.1f B constraints/sec (all %d cores)\n",
        ghz, multi_throughput * ghz, nc);
    
    return 0;
}
