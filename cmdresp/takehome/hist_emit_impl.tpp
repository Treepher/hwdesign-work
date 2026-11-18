// hist_emit_impl.tpp
// Included from gen/hist.hpp at the bottom -- types declared there (HistCmdHdr,
// HistError, streamutils) are in scope. Do not include this file directly except
// via the .hpp.
//
// Hand-written body for the templated `hist_impl::emit` hook (the END command).
// Mechanical translation of the pysim golden-model hook `HistAccel.emit` in
// hist.py: emit the response header FIRST (one RespHdr per command header),
// gate on configuration (nbin == 0), then stream the `nbin` counters in bin
// order as 32-bit unsigned payload words, TLAST on the final word.

#include "include/hist_resp_hdr.h"

namespace hist_impl {

template <int out_bw>
ap_uint<8> emit(HistCmdHdr cmd_hdr,
                hls::stream<streamutils::axi4s_word<out_bw>>& m_out,
                ap_uint<32> nbin[1], ap_uint<32> counters[16]) {
    // ----- Emit response header (ack) first -----
    HistRespHdr resp_hdr;
    resp_hdr.tx_id = cmd_hdr.tx_id;
    resp_hdr.write_axi4_stream<out_bw>(m_out, true);

    // ----- Gate: must be configured -----
    const int n = (int)nbin[0];
    if (n == 0) {
        return (ap_uint<8>)static_cast<unsigned int>(HistError::DATA_BEFORE_CONFIG);
    }

    // ----- Stream the counters in bin order -----
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        const bool tlast = (i == n - 1);
        streamutils::write_axi4_word<out_bw>(m_out, (ap_uint<out_bw>)counters[i], tlast);
    }
    return (ap_uint<8>)static_cast<unsigned int>(HistError::NO_ERROR);
}

} // namespace hist_impl
