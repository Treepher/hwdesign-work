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

    // Per-activation state, LOCAL to the kernel call.  Each activation (each
    // ap_start) re-enters this function, so `nbin` starts at 0 ("not yet
    // configured") and `counters` at zero again: a CONFIG command sets them,
    // DATA accumulates into them, END emits them -- and nothing persists across
    // activations.  This matches the prompt ("each ap_start runs a fresh kernel")
    // and the golden model, whose per-activation state is re-created for every
    // scenario.  Keeping the state local (not static) is what lets the
    // multi-activation testbench drive a DATA-before-CONFIG error and a clean
    // recovery back-to-back: every scenario genuinely starts unconfigured.
    ap_uint<32> nbin[1];
    nbin[0] = 0;
    float bounds[2];
    bounds[0] = 0.0f;
    bounds[1] = 0.0f;
    ap_uint<32> counters[16];
    for (int i = 0; i < 16; ++i) {
        counters[i] = 0;
    }

    // A fresh activation starts with clean status.  In C-simulation the
    // testbench initializes these variables, but in RTL co-simulation they are
    // DUT-driven AXI-Lite registers: without this, an activation following an
    // error would still read the previous activation's error code.  Clearing
    // them here on every ap_start implements the "fresh kernel per activation"
    // contract and keeps C-sim and co-sim bit-identical on
    // halted / error / tx_id.
    halted = 0;
    error = 0;
    tx_id = 0;

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
