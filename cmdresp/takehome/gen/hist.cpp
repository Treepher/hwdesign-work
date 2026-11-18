#include "hist.hpp"

void hist(
    hls::stream<streamutils::axi4s_word<32>>& s_in,
    hls::stream<streamutils::axi4s_word<32>>& m_out,
    ap_uint<1>& halted,
    ap_uint<8>& error,
    ap_uint<16>& tx_id
) {
#pragma HLS INTERFACE axis port=s_in
#pragma HLS INTERFACE axis port=m_out
#pragma HLS INTERFACE s_axilite port=halted       bundle=control
#pragma HLS INTERFACE s_axilite port=error        bundle=control
#pragma HLS INTERFACE s_axilite port=tx_id        bundle=control
#pragma HLS INTERFACE s_axilite port=return       bundle=control
    // Cross-firing state (HwModule.add_state) -- persists across firings.
    static ap_uint<32> nbin[1];
    static float bounds[2];
    static ap_uint<32> counters[16];
    while (true) {
        HistCmdHdr cmd_hdr;
        cmd_hdr.read_axi4_stream<32>(s_in);
        if (cmd_hdr.cmd_type == HistCmdType::CONFIG) {
            ap_uint<8> err = hist_impl::configure(cmd_hdr, s_in, m_out, nbin, bounds, counters);
            if (err != (ap_uint<8>)static_cast<unsigned int>(HistError::NO_ERROR)) {
                error = err;
                tx_id = cmd_hdr.tx_id;
                halted = 1;
                return;
            }
        } else {
            if (cmd_hdr.cmd_type == HistCmdType::DATA) {
                ap_uint<8> err = hist_impl::accumulate(cmd_hdr, s_in, m_out, nbin, bounds, counters);
                if (err != (ap_uint<8>)static_cast<unsigned int>(HistError::NO_ERROR)) {
                    error = err;
                    tx_id = cmd_hdr.tx_id;
                    halted = 1;
                    return;
                }
            } else {
                ap_uint<8> err = hist_impl::emit(cmd_hdr, m_out, nbin, counters);
                if (err != (ap_uint<8>)static_cast<unsigned int>(HistError::NO_ERROR)) {
                    error = err;
                    tx_id = cmd_hdr.tx_id;
                    halted = 1;
                }
                return;
            }
        }
    }
}

