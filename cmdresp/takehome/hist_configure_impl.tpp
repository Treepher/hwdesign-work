// hist_configure_impl.tpp
// Included from gen/hist.hpp at the bottom -- types declared there (HistCmdHdr,
// HistError, streamutils, float32_array_utils) are in scope. Do not include this
// file directly except via the .hpp.
//
// Hand-written body for the templated `hist_impl::configure` hook (the CONFIG
// command).  Mechanical translation of the pysim golden-model hook
// `HistAccel.configure` in hist.py: emit the response header FIRST (keeps the
// stream in lockstep -- one RespHdr per command header), then read the config
// payload, validate it, and on success store the config and zero the counters.

// CfgPayload / HistRespHdr are used in the hook body but are not visible
// through the generated kernel signature, so gen/hist.hpp doesn't auto-include
// them.  Pull them in here -- each schema header has its own include guard.
#include "include/cfg_payload.h"
#include "include/hist_resp_hdr.h"

namespace hist_impl {

template <int in_bw, int out_bw>
ap_uint<8> configure(HistCmdHdr cmd_hdr,
                     hls::stream<streamutils::axi4s_word<in_bw>>& s_in,
                     hls::stream<streamutils::axi4s_word<out_bw>>& m_out,
                     ap_uint<32> nbin[1], float bounds[2], ap_uint<32> counters[16]) {
    // ----- Emit response header (ack) first -----
    HistRespHdr resp_hdr;
    resp_hdr.tx_id = cmd_hdr.tx_id;
    resp_hdr.write_axi4_stream<out_bw>(m_out, true);

    // ----- Read the configuration payload (nbin / lo / hi) -----
    CfgPayload cfg;
    cfg.read_axi4_stream<in_bw>(s_in);
    const int   n  = (int)cfg.nbin;
    const float lo = cfg.lo;
    const float hi = cfg.hi;

    // ----- Validate: 1 <= nbin <= 16 and lo < hi -----
    if (n < 1 || n > 16 || lo >= hi) {
        return (ap_uint<8>)static_cast<unsigned int>(HistError::INVALID_CONFIG);
    }

    // ----- Store config; counters start at zero -----
    nbin[0]   = (ap_uint<32>)n;
    bounds[0] = lo;
    bounds[1] = hi;
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        counters[i] = 0;
    }
    return (ap_uint<8>)static_cast<unsigned int>(HistError::NO_ERROR);
}

} // namespace hist_impl
