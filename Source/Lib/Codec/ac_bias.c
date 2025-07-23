/*
* Copyright(c) 2024 Gianni Rosato
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include "ac_bias.h"
#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"

/*
 * 8-bit functions
 */
static uint64_t svt_psy_sa8d_8x8(const uint8_t* s, const uint32_t sp, const uint8_t* r, const uint32_t rp) {
    int16_t diff[64];
    int32_t coeff[64];
    uint64_t sum = 0;

    for (int i = 0; i < 64; i += 8) {
        const uint8_t *s_row = s + i * sp;
        const uint8_t *r_row = r + i * rp;
        for (int j = 0; j < 8; j++)
            diff[i + j] = s_row[j] - r_row[j];
    }

    svt_aom_hadamard_8x8(diff, 8, coeff);

    for (int i = 0; i < 64; i++)
        sum += abs(coeff[i]);

    return (sum + 2) >> 2;
}

static uint64_t svt_psy_satd_4x4(const uint8_t* s, const uint32_t sp, const uint8_t* r, const uint32_t rp) {
    int16_t diff[16];
    int32_t coeff[16];
    uint64_t sum = 0;

    for (int i = 0; i < 16; i += 4) {
        const uint8_t *s_row = s + i * sp;
        const uint8_t *r_row = r + i * rp;
        for (int j = 0; j < 4; j++)
            diff[i + j] = s_row[j] - r_row[j];
    }

    svt_aom_hadamard_4x4(diff, 4, coeff);

    for (int i = 0; i < 16; i++)
        sum += abs(coeff[i]);

    return sum >> 1;
}

uint64_t svt_psy_distortion(const uint8_t* input, const uint32_t input_stride,
                            const uint8_t* recon, const uint32_t recon_stride,
                            const uint32_t width, const uint32_t height) {

    static uint8_t zero_buffer[8] = { 0 };
    uint64_t total_nrg = 0;

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint64_t i = 0; i < height; i += 8)
            for (uint64_t j = 0; j < width; j += 8) {
                const int32_t input_nrg = svt_psy_sa8d_8x8(input + i * input_stride + j, input_stride, zero_buffer, 0) -
                    (svt_aom_sad8x8(input + i * input_stride + j, input_stride, zero_buffer, 0) >> 2);
                const int32_t recon_nrg = svt_psy_sa8d_8x8(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) -
                    (svt_aom_sad8x8(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) >> 2);
                total_nrg += abs(input_nrg - recon_nrg);
            }
        return (total_nrg >> 1);
    }
    for (uint64_t i = 0; i < height; i += 4)
        for (uint64_t j = 0; j < width; j += 4) {
            const int32_t input_nrg = svt_psy_satd_4x4(input + i * input_stride + j, input_stride, zero_buffer, 0) -
                (svt_aom_sad4x4(input + i * input_stride + j, input_stride, zero_buffer, 0) >> 2);
            const int32_t recon_nrg = svt_psy_satd_4x4(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) -
                (svt_aom_sad4x4(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) >> 2);
            total_nrg += abs(input_nrg - recon_nrg);
        }
    // Energy is scaled to match equivalent HBD strengths
    return (total_nrg >> 1);
}

/*
 * 10-bit functions
 */
static uint64_t svt_psy_sa8d_8x8_hbd(const uint16_t* s, const uint32_t sp, const uint16_t* r, const uint32_t rp) {
    int16_t diff[64];
    int32_t coeff[64];
    uint64_t sum = 0;

    for (int i = 0; i < 64; i += 8) {
        const uint16_t *s_row = s + i * sp;
        const uint16_t *r_row = r + i * rp;
        for (int j = 0; j < 8; j++)
            diff[i + j] = (int16_t)(s_row[j] - r_row[j]);
    }

    svt_aom_hadamard_8x8(diff, 8, coeff);

    for (int i = 0; i < 64; i++)
        sum += abs(coeff[i]);

    return (sum + 2) >> 2;
}

static uint64_t svt_psy_satd_4x4_hbd(const uint16_t* s, const uint32_t sp, const uint16_t* r, const uint32_t rp) {
    int16_t diff[16];
    int32_t coeff[16];
    uint64_t sum = 0;

    for (int i = 0; i < 16; i += 4) {
        const uint16_t *s_row = s + i * sp;
        const uint16_t *r_row = r + i * rp;
        for (int j = 0; j < 4; j++)
            diff[i + j] = (int16_t)(s_row[j] - r_row[j]);
    }

    svt_aom_hadamard_4x4(diff, 4, coeff);

    for (int i = 0; i < 16; i++)
        sum += abs(coeff[i]);

    return sum >> 1;
}

uint64_t svt_psy_distortion_hbd(uint16_t* input, const uint32_t input_stride,
                                uint16_t* recon, const uint32_t recon_stride,
                                const uint32_t width, const uint32_t height) {

    static uint16_t zero_buffer[8] = { 0 };
    uint64_t total_nrg = 0;

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint64_t i = 0; i < height; i += 8)
            for (uint64_t j = 0; j < width; j += 8) {
                const int32_t input_nrg = (svt_psy_sa8d_8x8_hbd(input + i * input_stride + j, input_stride, zero_buffer, 0)) -
                    (sad_16b_kernel(input + i * input_stride + j, input_stride, zero_buffer, 0, 8, 8) >> 2);
                const int32_t recon_nrg = (svt_psy_sa8d_8x8_hbd(recon + i * recon_stride + j, recon_stride, zero_buffer, 0)) -
                    (sad_16b_kernel(recon + i * recon_stride + j, recon_stride, zero_buffer, 0, 8, 8) >> 2);
                total_nrg += abs(input_nrg - recon_nrg);
            }
        return (total_nrg << 2);
    }
    for (uint64_t i = 0; i < height; i += 4) /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
        for (uint64_t j = 0; j < width; j += 4) {
            const int32_t input_nrg = svt_psy_satd_4x4_hbd(input + i * input_stride + j, input_stride, zero_buffer, 0) -
                (sad_16b_kernel(input + i * input_stride + j, input_stride, zero_buffer, 0, 4, 4) >> 2);
            const int32_t recon_nrg = svt_psy_satd_4x4_hbd(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) -
                (sad_16b_kernel(recon + i * recon_stride + j, recon_stride, zero_buffer, 0, 4, 4) >> 2);
            total_nrg += abs(input_nrg - recon_nrg);
        }
    return (total_nrg << 2);
}

/*
 * Public function that mirrors the arguments of `spatial_full_dist_type_fun()`
 */
uint64_t get_svt_psy_full_dist(const void* s, const uint32_t so, const uint32_t sp,
                               const void* r, const uint32_t ro, const uint32_t rp,
                               const uint32_t w, const uint32_t h, const uint8_t is_hbd,
                               const double ac_bias) {
    if (is_hbd)
        return llrint(svt_psy_distortion_hbd((uint16_t*)s + so, sp, (uint16_t*)r + ro, rp, w, h) * ac_bias);
    else
        return llrint(svt_psy_distortion((const uint8_t*)s + so, sp, (const uint8_t*)r + ro, rp, w, h) * ac_bias);
}

double get_effective_ac_bias(const double ac_bias, const bool is_islice, const uint8_t temporal_layer_index) {
    if (is_islice) return ac_bias * 0.4;
    switch (temporal_layer_index) {
    case 0: return ac_bias * 0.75;
    case 1: return ac_bias * 0.9;
    case 2: return ac_bias * 0.95;
    default: return ac_bias;
    }
}
