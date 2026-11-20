// hist_tb.cpp — multi-activation C-sim / RTL co-sim testbench (take-home).
//
// The prompt requires driving, in BOTH C-sim and RTL co-simulation: basic,
// multiple-DATA accumulation, mid-activation re-CONFIG, the parameter limits,
// and every error class, with every error scenario followed by a fresh passing
// activation.  Each scenario below is a *separate activation*: a fresh set of
// hls::stream objects and status registers plus a fresh call to hist().  That
// is what exercises the kernel's "nothing persists across activations" rule —
// the kernel keeps its nbin/bounds/counters as LOCAL variables re-created for
// every call (see gen/hist.cpp), so every scenario genuinely starts from the
// unconfigured state (nbin == 0) and its counters at zero.
//
// The stimulus is the hand-written C++ mirror of the Python golden model's
// SCENARIOS in hist.py (the same command sequences and the same nbin/lo/hi/
// samples), so the two implementations are compared transaction-for-transaction.
// Two scenarios are C-sim / co-sim only, because the SimPy stream read
// (get_pipelined) reads a fixed word count and cannot surface an early or a
// missing TLAST:
//   * err_tlast_early : a DATA header declares nsamp, but the burst is shorter
//                       (TLAST lands before the declared count is reached);
//   * err_no_tlast    : the burst reaches the declared count with no TLAST.
//
// Per scenario the testbench writes two evidence files under <data_dir>/<name>/:
//   * counters.bin         : raw little-endian u32, the nbin counters in bin
//                            order (0 bytes for an error scenario);
//   * regmap_status.json   : { "halted", "error", "tx_id" } as read from the
//                            kernel's status out-ports.
//
// Response ordering (ack-first, one RespHdr per command header): a passing
// CONFIG -> DATA... -> END emits one RespHdr per command and then the nbin
// counter words on END.  An error command emits its RespHdr and then halts, so
// the stream holds exactly the acks for the commands seen so far and no
// counters.  The read sequence below matches that per scenario.

#include "hist.hpp"
#include "include/hist_cmd_hdr.h"
#include "include/cfg_payload.h"
#include "include/hist_resp_hdr.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using stream_t = hls::stream<streamutils::axi4s_word<32>>;

// --------------------------------------------------------------------------
// Stimulus helpers (drive the input stream).
// --------------------------------------------------------------------------

void drive_config(stream_t& s_in, int tx, int nbin, float lo, float hi) {
    HistCmdHdr h;
    h.cmd_type = HistCmdType::CONFIG;
    h.tx_id    = (unsigned short)tx;
    h.nsamp    = 3;                       // CfgPayload = 3 words (nbin, lo, hi)
    h.write_axi4_stream<32>(s_in, true);

    CfgPayload p;
    p.nbin = (ap_uint<16>)nbin;
    p.lo   = lo;
    p.hi   = hi;
    p.write_axi4_stream<32>(s_in, true);
}

// Drive a DATA header declaring `nsamp`, then a burst of `driven` samples with
// the given TLAST framing.  `driven < nsamp` models an early TLAST; `tlast=false`
// models a missing TLAST; a normal burst is `driven == nsamp`, `tlast == true`.
void drive_data(stream_t& s_in, int tx, int nsamp,
                const float* samples, bool tlast, int driven) {
    HistCmdHdr h;
    h.cmd_type = HistCmdType::DATA;
    h.tx_id    = (unsigned short)tx;
    h.nsamp    = (unsigned short)nsamp;
    h.write_axi4_stream<32>(s_in, true);

    if (driven > 0) {
        float32_array_utils::write_axi4_stream<32>(s_in, samples, tlast, driven);
    }
}

void drive_end(stream_t& s_in, int tx) {
    HistCmdHdr h;
    h.cmd_type = HistCmdType::END;
    h.tx_id    = (unsigned short)tx;
    h.nsamp    = 0;
    h.write_axi4_stream<32>(s_in, true);
}

// --------------------------------------------------------------------------
// Response helpers (read the output stream).
// --------------------------------------------------------------------------

void read_resp(stream_t& m_out, int& tx) {
    HistRespHdr r;
    streamutils::tlast_status tl = streamutils::tlast_status::no_tlast;
    r.read_axi4_stream<32>(m_out, tl);
    tx = (int)r.tx_id;
}

std::vector<uint32_t> read_counters(stream_t& m_out, int n) {
    ap_uint<32> buf[16];
    streamutils::tlast_status tl = streamutils::tlast_status::no_tlast;
    uint32_array_utils::read_axi4_stream<32>(m_out, buf, tl, n);
    std::vector<uint32_t> out(n);
    for (int i = 0; i < n; ++i) out[i] = (uint32_t)buf[i];
    return out;
}

// --------------------------------------------------------------------------
// Evidence writer.
// --------------------------------------------------------------------------

void write_evidence(const std::string& data_dir, const std::string& name,
                    const std::vector<uint32_t>& counters,
                    int halted, int error, int tx_id) {
    const std::string dir = data_dir + "/" + name + "/";
    {
        std::ofstream ofs((dir + "counters.bin").c_str(), std::ios::binary);
        if (!ofs) throw std::runtime_error("Failed to open counters.bin for " + name);
        for (uint32_t u : counters) ofs.write(reinterpret_cast<const char*>(&u), sizeof(uint32_t));
    }
    {
        std::ofstream s((dir + "regmap_status.json").c_str());
        if (!s) throw std::runtime_error("Failed to open regmap_status.json for " + name);
        s << "{\n";
        s << "  \"halted\": " << halted << ",\n";
        s << "  \"error\": " << error << ",\n";
        s << "  \"tx_id\": " << tx_id << "\n";
        s << "}\n";
    }
}

// Run one activation; the caller reads the status out-ports after it returns.
void activate(stream_t& s_in, stream_t& m_out,
              ap_uint<1>& halted, ap_uint<8>& error, ap_uint<16>& tx_id) {
    halted = 0;
    error  = 0;
    tx_id  = 0;
    hist(s_in, m_out, halted, error, tx_id);
}

// Read a clean passing activation's response (the n RespHdrs for the commands
// driven, then the nbin counters) and write its evidence.  The `nack` value is
// the number of response headers to consume before the counters.
void collect_passing(stream_t& m_out, const std::string& data_dir, const std::string& name,
                     int nack, int nbin,
                     int halted, int error, int tx_id) {
    int t;
    for (int i = 0; i < nack; ++i) read_resp(m_out, t);
    std::vector<uint32_t> c = read_counters(m_out, nbin);
    write_evidence(data_dir, name, c, halted, error, tx_id);
}

} // namespace

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    // ----------------------------------------------------------------------
    // 1. basic_handcheck : CONFIG(4,0,4) -> DATA(6) -> END  -> [2,1,1,2]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 1, 4, 0.0f, 4.0f);
        { const float s[6] = {0.2f, 1.8f, 2.1f, 3.9f, 0.0f, 3.2f}; drive_data(s_in, 2, 6, s, true, 6); }
        drive_end(s_in, 3);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "basic_handcheck", 3, 4,
                        (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 2. accumulate_two_data : one CONFIG, two DATA bursts, one END -> [2,1,1,1]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 1, 4, 0.0f, 4.0f);
        { const float s[2] = {0.5f, 3.5f};  drive_data(s_in, 2, 2, s, true, 2); }
        { const float s[3] = {1.5f, 2.5f, 0.1f}; drive_data(s_in, 3, 3, s, true, 3); }
        drive_end(s_in, 4);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "accumulate_two_data", 4, 4,
                        (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 3. reconfig_mid_activation : CONFIG -> DATA -> CONFIG(2) -> DATA -> END
    //    The second CONFIG resets the counters -> [1,1]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 1, 4, 0.0f, 4.0f);
        { const float s[1] = {0.5f};        drive_data(s_in, 2, 1, s, true, 1); }
        drive_config(s_in, 3, 2, 0.0f, 2.0f);
        { const float s[2] = {0.5f, 1.5f};  drive_data(s_in, 4, 2, s, true, 2); }
        drive_end(s_in, 5);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "reconfig_mid_activation", 5, 2,
                        (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 4. limit_b1 : B = 1 (single bin) -> [3]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 1, 1, 0.0f, 1.0f);
        { const float s[3] = {0.1f, 0.5f, 0.9f}; drive_data(s_in, 2, 3, s, true, 3); }
        drive_end(s_in, 3);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "limit_b1", 3, 1,
                        (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 5. limit_b16_nsamp128 : B = 16, 128 samples spanning [0,15]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 1, 16, 0.0f, 16.0f);
        // Samples = np.linspace(0.0, 15.0, 128, dtype=np.float32) -- exactly the
        // golden model's authoritative values (hist.py).  Stored as raw float32
        // bit patterns and reconstructed so the C path matches the golden bit for
        // bit regardless of the compiler's decimal-literal rounding.
        static const uint32_t s_bits[128] = {
            0x00000000u, 0x3df1e3c8u, 0x3e71e3c8u, 0x3eb56ad6u, 0x3ef1e3c8u, 0x3f172e5du, 0x3f356ad6u, 0x3f53a74fu,
            0x3f71e3c8u, 0x3f881020u, 0x3f972e5du, 0x3fa64c99u, 0x3fb56ad6u, 0x3fc48912u, 0x3fd3a74fu, 0x3fe2c58bu,
            0x3ff1e3c8u, 0x40008102u, 0x40081020u, 0x400f9f3eu, 0x40172e5du, 0x401ebd7bu, 0x40264c99u, 0x402ddbb7u,
            0x40356ad6u, 0x403cf9f4u, 0x40448912u, 0x404c1830u, 0x4053a74fu, 0x405b366du, 0x4062c58bu, 0x406a54a9u,
            0x4071e3c8u, 0x407972e6u, 0x40808102u, 0x40844891u, 0x40881020u, 0x408bd7afu, 0x408f9f3eu, 0x409366ceu,
            0x40972e5du, 0x409af5ecu, 0x409ebd7bu, 0x40a2850au, 0x40a64c99u, 0x40aa1428u, 0x40addbb7u, 0x40b1a347u,
            0x40b56ad6u, 0x40b93265u, 0x40bcf9f4u, 0x40c0c183u, 0x40c48912u, 0x40c850a1u, 0x40cc1830u, 0x40cfdfbfu,
            0x40d3a74fu, 0x40d76edeu, 0x40db366du, 0x40defdfcu, 0x40e2c58bu, 0x40e68d1au, 0x40ea54a9u, 0x40ee1c38u,
            0x40f1e3c8u, 0x40f5ab57u, 0x40f972e6u, 0x40fd3a75u, 0x41008102u, 0x410264cau, 0x41044891u, 0x41062c59u,
            0x41081020u, 0x4109f3e8u, 0x410bd7afu, 0x410dbb77u, 0x410f9f3eu, 0x41118306u, 0x411366ceu, 0x41154a95u,
            0x41172e5du, 0x41191224u, 0x411af5ecu, 0x411cd9b3u, 0x411ebd7bu, 0x4120a143u, 0x4122850au, 0x412468d2u,
            0x41264c99u, 0x41283061u, 0x412a1428u, 0x412bf7f0u, 0x412ddbb7u, 0x412fbf7fu, 0x4131a347u, 0x4133870eu,
            0x41356ad6u, 0x41374e9du, 0x41393265u, 0x413b162cu, 0x413cf9f4u, 0x413eddbbu, 0x4140c183u, 0x4142a54bu,
            0x41448912u, 0x41466cdau, 0x414850a1u, 0x414a3469u, 0x414c1830u, 0x414dfbf8u, 0x414fdfbfu, 0x4151c387u,
            0x4153a74fu, 0x41558b16u, 0x41576edeu, 0x415952a5u, 0x415b366du, 0x415d1a34u, 0x415efdfcu, 0x4160e1c4u,
            0x4162c58bu, 0x4164a953u, 0x41668d1au, 0x416870e2u, 0x416a54a9u, 0x416c3871u, 0x416e1c38u, 0x41700000u};
        float s[128];
        for (int i = 0; i < 128; ++i) {
            float f;
            std::memcpy(&f, &s_bits[i], sizeof(f));
            s[i] = f;
        }
        drive_data(s_in, 2, 128, s, true, 128);
        drive_end(s_in, 3);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "limit_b16_nsamp128", 3, 16,
                        (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 6. limit_nsamp1 : a single sample in a DATA command -> [0,0,1,0]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 1, 4, 0.0f, 4.0f);
        { const float s[1] = {2.5f}; drive_data(s_in, 2, 1, s, true, 1); }
        drive_end(s_in, 3);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "limit_nsamp1", 3, 4,
                        (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 7. err_data_before_config : DATA before any CONFIG -> halt (error=1)
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        { const float s[2] = {0.5f, 1.5f}; drive_data(s_in, 7, 2, s, true, 2); }
        activate(s_in, m_out, halted, error, tx_id);
        int t;
        read_resp(m_out, t);                       // DATA ack (then kernel halts)
        std::vector<uint32_t> c;
        write_evidence(data_dir, "err_data_before_config", c,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 8. err_invalid_config_b0 : CONFIG with nbin = 0 -> halt (error=2)
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 8, 0, 0.0f, 4.0f);
        activate(s_in, m_out, halted, error, tx_id);
        int t;
        read_resp(m_out, t);                       // CONFIG ack (then kernel halts)
        std::vector<uint32_t> c;
        write_evidence(data_dir, "err_invalid_config_b0", c,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 9. err_invalid_config_lo_ge_hi : CONFIG with lo >= hi -> halt (error=2)
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 9, 4, 4.0f, 2.0f);
        activate(s_in, m_out, halted, error, tx_id);
        int t;
        read_resp(m_out, t);                       // CONFIG ack (then kernel halts)
        std::vector<uint32_t> c;
        write_evidence(data_dir, "err_invalid_config_lo_ge_hi", c,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 10. err_zero_samples : DATA with nsamp = 0 -> halt (error=3)
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 10, 4, 0.0f, 4.0f);
        drive_data(s_in, 11, 0, nullptr, true, 0); // nsamp = 0: no payload
        activate(s_in, m_out, halted, error, tx_id);
        int t;
        read_resp(m_out, t);                       // CONFIG ack
        read_resp(m_out, t);                       // DATA ack (then kernel halts)
        std::vector<uint32_t> c;
        write_evidence(data_dir, "err_zero_samples", c,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 11. err_sample_out_of_range : one sample above hi -> halt (error=4)
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 12, 4, 0.0f, 4.0f);
        { const float s[2] = {0.5f, 4.5f}; drive_data(s_in, 13, 2, s, true, 2); }
        activate(s_in, m_out, halted, error, tx_id);
        int t;
        read_resp(m_out, t);                       // CONFIG ack
        read_resp(m_out, t);                       // DATA ack (then kernel halts)
        std::vector<uint32_t> c;
        write_evidence(data_dir, "err_sample_out_of_range", c,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 12. numpy_crosscheck : 24 in-range samples; counters must equal
    //     np.histogram(samples, bins=8, range=(0,8)).  (common)
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 20, 8, 0.0f, 8.0f);
        // Samples = np.random.default_rng(0).uniform(0.0, 8.0, 24).astype
        // (np.float32) -- exactly the golden model's values (hist.py).  Stored
        // as raw float32 bit patterns and reconstructed so the C path matches
        // the golden bit for bit regardless of the compiler's decimal-literal
        // rounding (7 significant digits does not always round-trip a float32).
        static const uint32_t s_bits[24] = {
            0x40a30fecu, 0x400a217cu, 0x3ea7d3dbu, 0x3e0764f7u, 0x40d0327au, 0x40e9aa59u, 0x409b4c7bu, 0x40bac049u,
            0x408b2b02u, 0x40ef60e8u, 0x40d0dbc7u, 0x3cb37869u, 0x40db7ed9u, 0x3e899107u, 0x40bacab3u, 0x3fb3df11u,
            0x40dcf94bu, 0x408a9d34u, 0x401973d6u, 0x40586a76u, 0x3e67fea8u, 0x3f7e883bu, 0x40abae0bu, 0x40a5ae36u};
        float s[24];
        for (int i = 0; i < 24; ++i) {
            float f;
            std::memcpy(&f, &s_bits[i], sizeof(f));
            s[i] = f;
        }
        drive_data(s_in, 21, 24, s, true, 24);
        drive_end(s_in, 22);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "numpy_crosscheck", 3, 8,
                        (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 13. err_tlast_early : DATA declares nsamp=3, burst is 2 (early TLAST)
    //     -> halt (error=5).  [C-sim / co-sim only]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 30, 4, 0.0f, 4.0f);
        { const float s[3] = {0.5f, 1.5f, 2.5f};
          drive_data(s_in, 31, 3, s, true, 2); }   // nsamp=3, drive 2 with TLAST
        activate(s_in, m_out, halted, error, tx_id);
        int t;
        read_resp(m_out, t);                       // CONFIG ack
        read_resp(m_out, t);                       // DATA ack (then kernel halts)
        std::vector<uint32_t> c;
        write_evidence(data_dir, "err_tlast_early", c,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 14. err_no_tlast : DATA declares nsamp=3, burst is 3 with no TLAST
    //     -> halt (error=6).  [C-sim / co-sim only]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 40, 4, 0.0f, 4.0f);
        { const float s[3] = {0.5f, 1.5f, 2.5f};
          drive_data(s_in, 41, 3, s, false, 3); }  // nsamp=3, drive 3, no TLAST
        activate(s_in, m_out, halted, error, tx_id);
        int t;
        read_resp(m_out, t);                       // CONFIG ack
        read_resp(m_out, t);                       // DATA ack (then kernel halts)
        std::vector<uint32_t> c;
        write_evidence(data_dir, "err_no_tlast", c,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 15-21. Fresh-activation recovery: a clean passing CONFIG -> DATA -> END
    //     that follows each of the six error classes.  Each is a separate
    //     activation (fresh streams), so the kernel starts unconfigured and the
    //     hand-check must pass again.  All yield [2,1,1,2] with status {0,0,0}.
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 50, 4, 0.0f, 4.0f);
        { const float s[6] = {0.2f, 1.8f, 2.1f, 3.9f, 0.0f, 3.2f}; drive_data(s_in, 51, 6, s, true, 6); }
        drive_end(s_in, 52);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "recover_data_before_config", 3, 4,
                        (int)halted, (int)error, (int)tx_id);
    }
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 60, 4, 0.0f, 4.0f);
        { const float s[6] = {0.2f, 1.8f, 2.1f, 3.9f, 0.0f, 3.2f}; drive_data(s_in, 61, 6, s, true, 6); }
        drive_end(s_in, 62);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "recover_invalid_config_b0", 3, 4,
                        (int)halted, (int)error, (int)tx_id);
    }
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 70, 4, 0.0f, 4.0f);
        { const float s[6] = {0.2f, 1.8f, 2.1f, 3.9f, 0.0f, 3.2f}; drive_data(s_in, 71, 6, s, true, 6); }
        drive_end(s_in, 72);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "recover_invalid_config_lo_ge_hi", 3, 4,
                        (int)halted, (int)error, (int)tx_id);
    }
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 80, 4, 0.0f, 4.0f);
        { const float s[6] = {0.2f, 1.8f, 2.1f, 3.9f, 0.0f, 3.2f}; drive_data(s_in, 81, 6, s, true, 6); }
        drive_end(s_in, 82);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "recover_zero_samples", 3, 4,
                        (int)halted, (int)error, (int)tx_id);
    }
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 90, 4, 0.0f, 4.0f);
        { const float s[6] = {0.2f, 1.8f, 2.1f, 3.9f, 0.0f, 3.2f}; drive_data(s_in, 91, 6, s, true, 6); }
        drive_end(s_in, 92);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "recover_sample_out_of_range", 3, 4,
                        (int)halted, (int)error, (int)tx_id);
    }
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 100, 4, 0.0f, 4.0f);
        { const float s[6] = {0.2f, 1.8f, 2.1f, 3.9f, 0.0f, 3.2f}; drive_data(s_in, 101, 6, s, true, 6); }
        drive_end(s_in, 102);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "recover_tlast_early", 3, 4,
                        (int)halted, (int)error, (int)tx_id);
    }
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 110, 4, 0.0f, 4.0f);
        { const float s[6] = {0.2f, 1.8f, 2.1f, 3.9f, 0.0f, 3.2f}; drive_data(s_in, 111, 6, s, true, 6); }
        drive_end(s_in, 112);
        activate(s_in, m_out, halted, error, tx_id);
        collect_passing(m_out, data_dir, "recover_no_tlast", 3, 4,
                        (int)halted, (int)error, (int)tx_id);
    }

    return 0;
}
