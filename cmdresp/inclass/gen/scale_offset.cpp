#include "scale_offset.hpp"

void scale_offset(
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
    // ap_start) re-enters this function, so `configured` starts at 0 and `ab`
    // at 0.0 again: a CONFIG command sets them, and nothing persists across
    // activations -- matching the golden model, whose per-activation state is
    // re-created for every scenario.
    ap_uint<32> configured[1];
    configured[0] = 0;
    float ab[2];
    ab[0] = 0.0f;
    ab[1] = 0.0f;

    while (true) {
        ScaleCmdHdr cmd_hdr;
        cmd_hdr.read_axi4_stream<32>(s_in);
        if (cmd_hdr.cmd_type == ScaleCmdType::CONFIG) {
            ap_uint<8> err = scale_offset_impl::configure(cmd_hdr, s_in, m_out, configured, ab);
            if (err != (ap_uint<8>)static_cast<unsigned int>(ScaleError::NO_ERROR)) {
                error = err;
                tx_id = cmd_hdr.tx_id;
                halted = 1;
                return;
            }
        } else if (cmd_hdr.cmd_type == ScaleCmdType::DATA) {
            ap_uint<8> err = scale_offset_impl::transform(cmd_hdr, s_in, m_out, configured, ab);
            if (err != (ap_uint<8>)static_cast<unsigned int>(ScaleError::NO_ERROR)) {
                error = err;
                tx_id = cmd_hdr.tx_id;
                halted = 1;
                return;
            }
        } else {  // END -- terminal
            ap_uint<8> err = scale_offset_impl::end(cmd_hdr, m_out, configured);
            if (err != (ap_uint<8>)static_cast<unsigned int>(ScaleError::NO_ERROR)) {
                error = err;
                tx_id = cmd_hdr.tx_id;
                halted = 1;
            }
            return;
        }
    }
}
