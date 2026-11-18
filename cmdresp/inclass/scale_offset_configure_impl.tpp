// scale_offset_configure_impl.tpp
// Included from gen/scale_offset.hpp at the bottom -- types declared there
// (ScaleCmdHdr, ScaleError, streamutils, float32_array_utils) are in scope.
// Do not include this file directly except via the .hpp.
//
// Hand-written body for the templated `scale_offset_impl::configure` hook (the
// CONFIG command).  Mechanical translation of the golden-model hook
// `ScaleAccel.configure` in scale_offset.py: emit the response header FIRST
// (keeps the stream in lockstep -- one RespHdr per command header), then read
// the config payload (a, b), store it, and mark the activation as configured.

// ScaleCfgPayload / ScaleRespHdr are used in the hook body but are not visible
// through the generated kernel signature, so gen/scale_offset.hpp doesn't
// auto-include them.  Pull them in here -- each schema header has its own
// include guard.
#include "include/scale_cfg_payload.h"
#include "include/scale_resp_hdr.h"

namespace scale_offset_impl {

template <int in_bw, int out_bw>
ap_uint<8> configure(ScaleCmdHdr cmd_hdr,
                     hls::stream<streamutils::axi4s_word<in_bw>>& s_in,
                     hls::stream<streamutils::axi4s_word<out_bw>>& m_out,
                     ap_uint<32> configured[1], float ab[2]) {
    // ----- Emit response header (ack) first -----
    ScaleRespHdr resp_hdr;
    resp_hdr.tx_id = cmd_hdr.tx_id;
    resp_hdr.write_axi4_stream<out_bw>(m_out, true);

    // ----- Read the configuration payload (a, b) -----
    ScaleCfgPayload cfg;
    cfg.read_axi4_stream<in_bw>(s_in);

    // ----- Store config; mark the activation as configured -----
    ab[0] = cfg.a;
    ab[1] = cfg.b;
    configured[0] = 1;
    return (ap_uint<8>)static_cast<unsigned int>(ScaleError::NO_ERROR);
}

} // namespace scale_offset_impl
