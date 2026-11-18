#pragma once

#include <ap_int.h>
#include <ap_fixed.h>

#include "include/streamutils_hls.h"
#include "include/hist_cmd_hdr.h"
#include "include/u_int32_array.h"
#include "include/float32_array.h"
#include "include/counter_array.h"
#include "include/hist_error.h"
#include "include/uint32_array_utils.h"
#include "include/float32_array_utils.h"

void hist(
    hls::stream<streamutils::axi4s_word<32>>& s_in,
    hls::stream<streamutils::axi4s_word<32>>& m_out,
    ap_uint<1>& halted,
    ap_uint<8>& error,
    ap_uint<16>& tx_id
);

namespace hist_impl {
    template <int in_bw, int out_bw>
    ap_uint<8> configure(HistCmdHdr cmd_hdr, hls::stream<streamutils::axi4s_word<in_bw>>& s_in, hls::stream<streamutils::axi4s_word<out_bw>>& m_out, ap_uint<32> nbin[1], float bounds[2], ap_uint<32> counters[16]);
    template <int in_bw, int out_bw>
    ap_uint<8> accumulate(HistCmdHdr cmd_hdr, hls::stream<streamutils::axi4s_word<in_bw>>& s_in, hls::stream<streamutils::axi4s_word<out_bw>>& m_out, ap_uint<32> nbin[1], float bounds[2], ap_uint<32> counters[16]);
    template <int out_bw>
    ap_uint<8> emit(HistCmdHdr cmd_hdr, hls::stream<streamutils::axi4s_word<out_bw>>& m_out, ap_uint<32> nbin[1], ap_uint<32> counters[16]);
}

#include "../hist_configure_impl.tpp"
#include "../hist_accumulate_impl.tpp"
#include "../hist_emit_impl.tpp"
