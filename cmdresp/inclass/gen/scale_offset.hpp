#pragma once

#include <ap_int.h>
#include <ap_fixed.h>

#include "include/streamutils_hls.h"
#include "include/scale_cmd_hdr.h"
#include "include/scale_cmd_type.h"
#include "include/scale_error.h"
#include "include/uint32_array_utils.h"
#include "include/float32_array_utils.h"

void scale_offset(
    hls::stream<streamutils::axi4s_word<32>>& s_in,
    hls::stream<streamutils::axi4s_word<32>>& m_out,
    ap_uint<1>& halted,
    ap_uint<8>& error,
    ap_uint<16>& tx_id
);

namespace scale_offset_impl {
    template <int in_bw, int out_bw>
    ap_uint<8> configure(ScaleCmdHdr cmd_hdr, hls::stream<streamutils::axi4s_word<in_bw>>& s_in, hls::stream<streamutils::axi4s_word<out_bw>>& m_out, ap_uint<32> configured[1], float ab[2]);
    template <int in_bw, int out_bw>
    ap_uint<8> transform(ScaleCmdHdr cmd_hdr, hls::stream<streamutils::axi4s_word<in_bw>>& s_in, hls::stream<streamutils::axi4s_word<out_bw>>& m_out, ap_uint<32> configured[1], float ab[2]);
    template <int out_bw>
    ap_uint<8> end(ScaleCmdHdr cmd_hdr, hls::stream<streamutils::axi4s_word<out_bw>>& m_out, ap_uint<32> configured[1]);
}

#include "../scale_offset_configure_impl.tpp"
#include "../scale_offset_transform_impl.tpp"
#include "../scale_offset_end_impl.tpp"
