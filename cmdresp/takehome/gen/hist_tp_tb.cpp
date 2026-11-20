// hist_tp_tb.cpp — throughput measurement testbench (take-home).
//
// The prompt's throughput target is an initiation interval of II = 1 (one
// sample accepted per clock) for the per-sample binning loop.  A single
// activation's co-sim "Total Execution Time" is dominated by the fixed
// per-activation overhead (the CONFIG command, the DATA header, the END command
// and its counter burst), so it does not directly expose the per-sample rate.
//
// This testbench therefore runs exactly ONE activation, CONFIG(16,0,16) ->
// DATA(N) -> END, where N is read from <argv[1]>/tp_nsamp.txt (so the build can
// run it twice, with N = 1 and N = 128, and take the difference).  The
// marginal per-sample cost --
//
//     throughput = (128 - 1) / (cycles(N=128) - cycles(N=1))   samples/clock
//
// -- cancels every fixed overhead and isolates the binning loop's initiation
// interval.  It is ~1.0 samples/clock iff the loop reaches II = 1.
//
// All N samples are in-range [0,16) (sample i = i * 16 / N), so the activation
// is clean and the measured time is purely the pipeline's per-sample cost.

#include "hist.hpp"
#include "include/hist_cmd_hdr.h"
#include "include/cfg_payload.h"
#include "include/hist_resp_hdr.h"

#include <cstdint>
#include <fstream>
#include <string>

namespace {
using stream_t = hls::stream<streamutils::axi4s_word<32>>;
}

int main(int argc, char** argv) {
    int N = 128;
    if (argc > 1) {
        std::ifstream f((std::string(argv[1]) + "/tp_nsamp.txt").c_str());
        if (f) f >> N;
    }
    if (N < 1)   N = 1;
    if (N > 128) N = 128;

    const int   B  = 16;
    const float lo = 0.0f;
    const float hi = 16.0f;

    float samples[128];
    for (int i = 0; i < N; ++i) samples[i] = (float)i * (hi / (float)N);   // all in [0,16)

    stream_t s_in, m_out;

    // CONFIG(16, 0.0, 16.0)
    {
        HistCmdHdr h;
        h.cmd_type = HistCmdType::CONFIG;
        h.tx_id    = 1;
        h.nsamp    = 3;
        h.write_axi4_stream<32>(s_in, true);
        CfgPayload p;
        p.nbin = (ap_uint<16>)B;
        p.lo   = lo;
        p.hi   = hi;
        p.write_axi4_stream<32>(s_in, true);
    }
    // DATA(N)
    {
        HistCmdHdr d;
        d.cmd_type = HistCmdType::DATA;
        d.tx_id    = 2;
        d.nsamp    = (unsigned short)N;
        d.write_axi4_stream<32>(s_in, true);
        float32_array_utils::write_axi4_stream<32>(s_in, samples, true, N);
    }
    // END
    {
        HistCmdHdr e;
        e.cmd_type = HistCmdType::END;
        e.tx_id    = 3;
        e.nsamp    = 0;
        e.write_axi4_stream<32>(s_in, true);
    }

    ap_uint<1>  halted;
    ap_uint<8>  error;
    ap_uint<16> tx_id;
    hist(s_in, m_out, halted, error, tx_id);

    // Drain the three response headers and the B counter words (the kernel
    // emits them); the point of this testbench is the co-sim clock count, not
    // the values.
    streamutils::tlast_status tl = streamutils::tlast_status::no_tlast;
    HistRespHdr r;
    r.read_axi4_stream<32>(m_out, tl);
    r.read_axi4_stream<32>(m_out, tl);
    r.read_axi4_stream<32>(m_out, tl);
    ap_uint<32> c[16];
    uint32_array_utils::read_axi4_stream<32>(m_out, c, tl, B);
    (void)c;
    return 0;
}
