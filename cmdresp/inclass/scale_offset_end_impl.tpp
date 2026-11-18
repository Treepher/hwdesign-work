// scale_offset_end_impl.tpp
// Included from gen/scale_offset.hpp at the bottom -- types declared there
// (ScaleCmdHdr, ScaleError, streamutils) are in scope. Do not include this
// file directly except via the .hpp.
//
// Hand-written body for the templated `scale_offset_impl::end` hook (the END
// command).  Mechanical translation of the golden-model hook `ScaleAccel.end`
// in scale_offset.py: emit the response header FIRST (one RespHdr per command
// header), then gate on configuration (configured == 0).  END is terminal.

#include "include/scale_resp_hdr.h"

namespace scale_offset_impl {

template <int out_bw>
ap_uint<8> end(ScaleCmdHdr cmd_hdr,
               hls::stream<streamutils::axi4s_word<out_bw>>& m_out,
               ap_uint<32> configured[1]) {
    // ----- Emit response header (ack) first -----
    ScaleRespHdr resp_hdr;
    resp_hdr.tx_id = cmd_hdr.tx_id;
    resp_hdr.write_axi4_stream<out_bw>(m_out, true);

    // ----- Gate: must be configured (an END before any CONFIG is an error) -----
    if (configured[0] == 0) {
        return (ap_uint<8>)static_cast<unsigned int>(ScaleError::DATA_BEFORE_CONFIG);
    }
    return (ap_uint<8>)static_cast<unsigned int>(ScaleError::NO_ERROR);
}

} // namespace scale_offset_impl
