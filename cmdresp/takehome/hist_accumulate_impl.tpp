// hist_accumulate_impl.tpp
// Included from gen/hist.hpp at the bottom -- types declared there (HistCmdHdr,
// HistError, streamutils, float32_array_utils) are in scope. Do not include this
// file directly except via the .hpp.
//
// Hand-written body for the templated `hist_impl::accumulate` hook (the DATA
// command).  Mechanical translation of the pysim golden-model hook
// `HistAccel.accumulate` in hist.py:
//   * emit the response header FIRST (one RespHdr per command header);
//   * gate on configuration (nbin == 0) and on a zero sample count;
//   * read `nsamp` f32 samples, binning each as  idx = truncate((x - lo) * scale)
//     with scale = nbin / (hi - lo), all in 32-bit float, clamped to [0, nbin-1];
//   * a sample outside [lo, hi) is SAMPLE_OUT_OF_RANGE (an all-or-nothing error:
//     no sample in the burst is committed);
//   * the burst must be TLAST-framed: an early TLAST is TLAST_EARLY, a burst that
//     runs out without a TLAST is NO_TLAST.
//
// Accumulation is done into a local counter buffer and committed into the
// persistent `counters` only on the success path, so that an errored burst
// leaves the running counters untouched (matching the golden model, whose
// `np.add.at` is only reached after the range check passes).

#include "include/hist_resp_hdr.h"

namespace hist_impl {

template <int in_bw, int out_bw>
ap_uint<8> accumulate(HistCmdHdr cmd_hdr,
                      hls::stream<streamutils::axi4s_word<in_bw>>& s_in,
                      hls::stream<streamutils::axi4s_word<out_bw>>& m_out,
                      ap_uint<32> nbin[1], float bounds[2], ap_uint<32> counters[16]) {
    // ----- Emit response header (ack) first -----
    HistRespHdr resp_hdr;
    resp_hdr.tx_id = cmd_hdr.tx_id;
    resp_hdr.write_axi4_stream<out_bw>(m_out, true);

    // ----- Gate: must be configured, non-empty burst -----
    const int n = (int)nbin[0];
    if (n == 0) {
        return (ap_uint<8>)static_cast<unsigned int>(HistError::DATA_BEFORE_CONFIG);
    }
    const int nsamp = (int)cmd_hdr.nsamp;
    if (nsamp == 0) {
        return (ap_uint<8>)static_cast<unsigned int>(HistError::ZERO_SAMPLES);
    }

    // ----- Binning parameters (32-bit float arithmetic throughout) -----
    const float lo    = bounds[0];
    const float hi    = bounds[1];
    const float scale = (float)n / (hi - lo);

    // Lane buffer for the vectorized read (pf lanes per AXI4-Stream word).
    static const int pf = float32_array_utils::pf<in_bw>();
    float x_lane[pf];
#pragma HLS ARRAY_PARTITION variable=x_lane complete dim=1

    // Local accumulation buffer; committed to `counters` only on success.
    // Fully partitioned into a register file (not a BRAM) so the per-sample
    // read-modify-write `local_counters[idx] += 1` closes in one cycle.  A 1R1W
    // BRAM would make that a 2-cycle read-after-write and pin the binning loop
    // at II = 2; the spec requires II = 1.  16 x u32 registers is well within
    // the device budget (see results/loop_df.csv resource table).
    ap_uint<32> local_counters[16];
#pragma HLS ARRAY_PARTITION variable=local_counters complete dim=1
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        local_counters[i] = 0;
    }

    bool   any_oor      = false;
    int    nsamp_read   = 0;
    bool   read_done    = false;
    streamutils::tlast_status samp_tlast = streamutils::tlast_status::no_tlast;

    for (int i = 0; i < nsamp && !read_done; i += pf) {
        const int nrem       = nsamp - i;
        const int lane_count = (nrem < pf) ? nrem : pf;
        streamutils::tlast_status lane_tlast = streamutils::tlast_status::no_tlast;
        float32_array_utils::read_axi4_stream_lane<in_bw>(
            s_in, x_lane, nrem, lane_tlast);

        for (int k = 0; k < pf; ++k) {
#pragma HLS UNROLL
            if (k < lane_count) {
                const float x = x_lane[k];
                // Range check: [lo, hi) -- lo inclusive, hi exclusive.
                if (x < lo || x >= hi) {
                    any_oor = true;
                } else {
                    const float diff  = x - lo;
                    const float idx_f = diff * scale;
                    int idx = (int)idx_f;   // truncate toward zero
                    if (idx < 0)      idx = 0;
                    if (idx >= n)     idx = n - 1;
                    local_counters[idx] = local_counters[idx] + 1;
                }
            }
        }

        nsamp_read += lane_count;
        if (lane_tlast == streamutils::tlast_status::tlast_at_end) {
            read_done = true;
            samp_tlast = (nrem <= pf) ? streamutils::tlast_status::tlast_at_end
                                      : streamutils::tlast_status::tlast_early;
        }
    }

    // ----- Validate (priority matches the golden model) -----
    if (any_oor) {
        return (ap_uint<8>)static_cast<unsigned int>(HistError::SAMPLE_OUT_OF_RANGE);
    }
    if (samp_tlast == streamutils::tlast_status::tlast_early) {
        return (ap_uint<8>)static_cast<unsigned int>(HistError::TLAST_EARLY);
    }
    if (samp_tlast == streamutils::tlast_status::no_tlast) {
        return (ap_uint<8>)static_cast<unsigned int>(HistError::NO_TLAST);
    }
    if (nsamp_read != nsamp) {
        return (ap_uint<8>)static_cast<unsigned int>(HistError::TLAST_EARLY);
    }

    // ----- Commit the burst into the running counters -----
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        counters[i] = counters[i] + local_counters[i];
    }
    return (ap_uint<8>)static_cast<unsigned int>(HistError::NO_ERROR);
}

} // namespace hist_impl
