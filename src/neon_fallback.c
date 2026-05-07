// ARM NEON Mixed-Precision Fallback
// For embedded safety controllers (Cortex-R5 with NEON, Cortex-A78)
// Compile: aarch64-linux-gnu-gcc -O3 -march=armv8-a+simd -o neon_check neon_fallback.c
// Or test on x86 with: gcc -O3 -DSCALAR_FALLBACK -o neon_check neon_fallback.c

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __aarch64__
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
// Scalar fallback for testing on x86
#endif

#define N 10000000

static inline uint64_t rdtsc() {
#ifdef __aarch64__
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#else
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#endif
}

int main() {
    printf("=== MIXED-PRECISION CHECK: %s ===\n", 
        HAS_NEON ? "ARM NEON (128-bit)" : "Scalar fallback (x86 simulation)");
    printf("%d constraints\n\n", N);
    
    // Generate AV mix data
    int n8 = (int)(N * 0.75);
    int n16 = (int)(N * 0.15);
    int n32 = (int)(N * 0.08);
    int nd = N - n8 - n16 - n32;
    
    int8_t*  v8  = aligned_alloc(64, n8); int8_t*  lo8 = aligned_alloc(64, n8); int8_t*  hi8 = aligned_alloc(64, n8);
    int16_t* v16 = aligned_alloc(64, n16*2); int16_t* lo16 = aligned_alloc(64, n16*2); int16_t* hi16 = aligned_alloc(64, n16*2);
    int32_t* v32 = aligned_alloc(64, n32*4); int32_t* lo32 = aligned_alloc(64, n32*4); int32_t* hi32 = aligned_alloc(64, n32*4);
    int32_t* vd  = aligned_alloc(64, nd*4); int32_t* lod  = aligned_alloc(64, nd*4); int32_t* hid  = aligned_alloc(64, nd*4);
    
    srand(42);
    for (int i = 0; i < n8; i++) { int lo=rand()%100; int hi=lo+rand()%20+1; v8[i]=(int8_t)(lo+rand()%(hi-lo+1)); lo8[i]=(int8_t)lo; hi8[i]=(int8_t)hi; }
    for (int i = 0; i < n16; i++) { int lo=rand()%3000-1500; int hi=lo+rand()%500+1; v16[i]=(int16_t)(lo+rand()%(hi-lo+1)); lo16[i]=(int16_t)lo; hi16[i]=(int16_t)hi; }
    for (int i = 0; i < n32; i++) { int lo=rand()%100000-50000; int hi=lo+rand()%30000+1; v32[i]=lo+rand()%(hi-lo+1); lo32[i]=lo; hi32[i]=hi; }
    for (int i = 0; i < nd; i++) { int lo=rand()%100000-50000; int hi=lo+rand()%30000+1; vd[i]=lo+rand()%(hi-lo+1); lod[i]=lo; hid[i]=hi; }
    
    volatile int sink = 0;
    uint64_t t0;
    
    // BASELINE: all INT32
    int32_t* bv = aligned_alloc(64, N*4); int32_t* blo = aligned_alloc(64, N*4); int32_t* bhi = aligned_alloc(64, N*4);
    for (int i = 0; i < N; i++) { blo[i]=rand()%200-100; bhi[i]=blo[i]+rand()%50+1; bv[i]=blo[i]+rand()%(bhi[i]-blo[i]+1); }
    
    t0 = rdtsc();
    for (int i = 0; i < N; i++) {
        sink += (bv[i] >= blo[i] && bv[i] <= bhi[i]);
    }
    uint64_t t_base = rdtsc() - t0;
    
    // MIXED: NEON or scalar fallback
    t0 = rdtsc();
    
#if HAS_NEON
    // ARM NEON: 128-bit registers → 16 INT8, 8 INT16, 4 INT32 per register
    for (int i = 0; i <= n8-16; i += 16) {
        int8x16_t vv = vld1q_s8(v8+i);
        int8x16_t ll = vld1q_s8(lo8+i);
        int8x16_t hh = vld1q_s8(hi8+i);
        uint8x16_t kge = vcgeq_s8(vv, ll);
        uint8x16_t kle = vcleq_s8(vv, hh);
        uint8x16_t pass = vandq_u8(kge, kle);
        sink += vaddvq_u8(vandq_u8(pass, vdupq_n_u8(1)));
    }
    for (int i = 0; i <= n16-8; i += 8) {
        int16x8_t vv = vld1q_s16(v16+i);
        int16x8_t ll = vld1q_s16(lo16+i);
        int16x8_t hh = vld1q_s16(hi16+i);
        uint16x8_t kge = vcgeq_s16(vv, ll);
        uint16x8_t kle = vcleq_s16(vv, hh);
        uint16x8_t pass = vandq_u16(kge, kle);
        sink += vaddvq_u16(vandq_u16(pass, vdupq_n_u16(1)));
    }
    for (int i = 0; i <= n32-4; i += 4) {
        int32x4_t vv = vld1q_s32(v32+i);
        int32x4_t ll = vld1q_s32(lo32+i);
        int32x4_t hh = vld1q_s32(hi32+i);
        uint32x4_t kge = vcgeq_s32(vv, ll);
        uint32x4_t kle = vcleq_s32(vv, hh);
        uint32x4_t pass = vandq_u32(kge, kle);
        sink += vaddvq_u32(vandq_u32(pass, vdupq_n_u32(1)));
    }
    // DUAL: XOR-based (NEON has no unsigned compare after XOR, use reinterpret)
    for (int i = 0; i <= nd-4; i += 4) {
        int32x4_t vv = vld1q_s32(vd+i);
        int32x4_t ll = vld1q_s32(lod+i);
        int32x4_t hh = vld1q_s32(hid+i);
        uint32x4_t ka = vandq_u32(vcgeq_s32(vv, ll), vcleq_s32(vv, hh));
        // XOR sign bit
        int32x4_t sign = vdupq_n_s32(0x80000000);
        uint32x4_t vu = vreinterpretq_u32_s32(veorq_s32(vv, sign));
        uint32x4_t lu = vreinterpretq_u32_s32(veorq_s32(ll, sign));
        uint32x4_t hu = vreinterpretq_u32_s32(veorq_s32(hh, sign));
        uint32x4_t kb = vandq_u32(vcgeq_u32(vu, lu), vcleq_u32(vu, hu));
        sink += vaddvq_u32(vandq_u32(vandq_u32(ka, kb), vdupq_n_u32(1)));
    }
#else
    // Scalar fallback (runs on any CPU)
    for (int i = 0; i < n8; i++) sink += (v8[i] >= lo8[i] && v8[i] <= hi8[i]);
    for (int i = 0; i < n16; i++) sink += (v16[i] >= lo16[i] && v16[i] <= hi16[i]);
    for (int i = 0; i < n32; i++) sink += (v32[i] >= lo32[i] && v32[i] <= hi32[i]);
    for (int i = 0; i < nd; i++) {
        int pa = (vd[i] >= lod[i] && vd[i] <= hid[i]);
        uint32_t vu = (uint32_t)vd[i] ^ 0x80000000;
        uint32_t lu = (uint32_t)lod[i] ^ 0x80000000;
        uint32_t hu = (uint32_t)hid[i] ^ 0x80000000;
        int pb = (vu >= lu && vu <= hu);
        sink += (pa && pb);
    }
#endif
    uint64_t t_mixed = rdtsc() - t0;
    
    printf("%-25s %12s %12s %12s\n", "Mode", "Cycles", "cyc/constr", "Speedup");
    printf("=============================================================\n");
    printf("%-25s %12lu %12.4f %12s\n", "Scalar INT32 baseline", t_base, (double)t_base/N, "1.00x");
    printf("%-25s %12lu %12.4f %12.2fx\n", 
        HAS_NEON ? "NEON Mixed" : "Scalar Mixed", t_mixed, (double)t_mixed/N, (double)t_base/t_mixed);
    
    printf("\nNEON vs AVX-512 register density:\n");
    printf("  AVX-512: INT8=64, INT16=32, INT32=16 per register\n");
    printf("  NEON:    INT8=16, INT16=8,  INT32=4  per register\n");
    printf("  Ratio:   4x less parallelism per register\n");
    printf("  But NEON runs at higher clock on mobile (3GHz vs 2.4GHz)\n");
    printf("  Net: NEON mixed ≈ AVX-512 mixed / 3 (not / 4 due to clock)\n");
    
    free(v8); free(lo8); free(hi8);
    free(v16); free(lo16); free(hi16);
    free(v32); free(lo32); free(hi32);
    free(vd); free(lod); free(hid);
    free(bv); free(blo); free(bhi);
    return 0;
}
