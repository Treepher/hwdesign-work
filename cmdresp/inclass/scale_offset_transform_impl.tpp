// scale_offset_transform_impl.tpp
// Included from gen/scale_offset.hpp at the bottom -- types declared there
// (ScaleCmdHdr, ScaleError, streamutils, float32_array_utils) are in scope.
// Do not include this file directly except via the .hpp.
//
// Hand-written body for the templated `scale_offset_impl::transform` hook (the
// DATA command).  Mechanical translation of the golden-model hook
// `ScaleAccel.transform` in scale_offset.py:
//   * emit the response header FIRST (one RespHdr per command header);
//   * gate on configuration (configured == 0) and on a zero sample count;
//   * read `nsamp` f32 samples, applying  y[i] = a * x[i] + b  (all in 32-bit
//     float, so the Python model and the C++ kernel produce identical bits);
//   * the burst must be TLAST-framed: an early TLAST is TLAST_EARLY, a burst
//     that reaches `nsamp` without a TLAST is NO_TLAST.
//
// On any error the burst is closed with the TLAST the kernel has already seen
// (the ack was already emitted, and no outputs are written), matching the
// golden model, which returns before it emits any output on the error paths.

#include "include/scale_resp_hdr.h"

namespace scale_offset_impl {

template <int in_bw, int out_bw>
ap_uint<8> transform(ScaleCmdHdr cmd_hdr,
                     hls::stream<streamutils::axi4s_word<in_bw>>& s_in,
                     hls::stream<streamutils::axi4s_word<out_bw>>& m_out,
                     ap_uint<32> configured[1], float ab[2]) {
    // ----- Emit response header (ack) first -----
    ScaleRespHdr resp_hdr;
    resp_hdr.tx_id = cmd_hdr.tx_id;
    resp_hdr.write_axi4_stream<out_bw>(m_out, true);

    // ----- Gate: must be configured, non-empty burst -----
    if (configured[0] == 0) {
        return (ap_uint<8>)static_cast<unsigned int>(ScaleError::DATA_BEFORE_CONFIG);
    }
    const int nsamp = (int)cmd_hdr.nsamp;
    if (nsamp == 0) {
        return (ap_uint<8>)static_cast<unsigned int>(ScaleError::ZERO_SAMPLES);
    }

    // ----- Read `nsamp` f32 samples (TLAST framing is tracked) -----
    // Largest burst is 128 samples; the buffer holds the whole burst.
    float x[128];
    streamutils::tlast_status tl = streamutils::tlast_status::no_tlast;
    int nread = 0;
    float32_array_utils::read_axi4_stream<in_bw>(s_in, x, tl, nread, nsamp);

    // ----- Validate the burst framing (priority matches the golden model) -----
    if (tl == streamutils::tlast_status::tlast_early) {
        return (ap_uint<8>)static_cast<unsigned int>(ScaleError::TLAST_EARLY);
    }
    if (tl == streamutils::tlast_status::no_tlast) {
        return (ap_uint<8>)static_cast<unsigned int>(ScaleError::NO_TLAST);
    }
    // tl == tlast_at_end: the burst is exactly `nsamp` samples.

    // ----- Scale-and-offset, 32-bit float arithmetic throughout -----
    const float a = ab[0];
    const float b = ab[1];
    float y[128];
    for (int i = 0; i < nsamp; ++i) {
        #pragma HLS PIPELINE II=1
        y[i] = (a * x[i]) + b;
    }

    // ----- Emit the `nsamp` outputs, in order, TLAST on the final word -----
    float32_array_utils::write_axi4_stream<out_bw>(m_out, y, true, nsamp);
    return (ap_uint<8>)static_cast<unsigned int>(ScaleError::NO_ERROR);
}

} // namespace scale_offset_impl
