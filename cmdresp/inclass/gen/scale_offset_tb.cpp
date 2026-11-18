// scale_offset_tb.cpp — multi-activation C-sim testbench (in class).
//
// The in-class prompt requires driving basic, two_DATA, re-CONFIG, and all four
// error classes, each followed by a fresh passing activation.  Each scenario is
// a *separate activation*: a fresh set of hls::stream objects and status
// registers plus a fresh call to scale_offset().  That is what exercises the
// kernel's "nothing persists across activations" rule -- the kernel keeps its
// configured/ab state as LOCAL variables re-created for every call (see
// gen/scale_offset.cpp), so every scenario starts from the unconfigured state.
//
// The stimulus is the hand-written C++ mirror of the Python golden model's
// SCENARIOS in scale_offset.py (the same command sequences and the same a/b/
// samples), so the two implementations are compared transaction-for-transaction.
// Two scenarios are C-sim only, because the SimPy stream read cannot surface an
// early or a missing TLAST:
//   * err_tlast_early : a DATA header declares nsamp, but the burst is shorter
//                       (TLAST lands before the declared count is reached);
//   * err_no_tlast    : the burst reaches the declared count with no TLAST.
//
// Per scenario the testbench writes two evidence files under data/<name>/:
//   * outputs.bin          : raw little-endian f32, every valid DATA output in
//                            command order (0 bytes for an error scenario);
//   * regmap_status.json   : { "halted", "error", "tx_id" } as read from the
//                            kernel's status out-ports.

#include "scale_offset.hpp"
#include "include/scale_cfg_payload.h"
#include "include/scale_resp_hdr.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using stream_t = hls::stream<streamutils::axi4s_word<32>>;

// --------------------------------------------------------------------------
// Stimulus helpers (drive the input stream).
// --------------------------------------------------------------------------

void drive_config(stream_t& s_in, int tx, float a, float b) {
    ScaleCmdHdr h;
    h.cmd_type = ScaleCmdType::CONFIG;
    h.tx_id    = (unsigned short)tx;
    h.nsamp    = 2;                       // ScaleCfgPayload = 2 words (a, b)
    h.write_axi4_stream<32>(s_in, true);

    ScaleCfgPayload p;
    p.a = a;
    p.b = b;
    p.write_axi4_stream<32>(s_in, true);
}

// Drive a DATA header declaring `nsamp`, then a burst of `driven` samples with
// the given TLAST framing.  `driven < nsamp` models an early TLAST; `tlast=false`
// models a missing TLAST; a normal burst is `driven == nsamp`, `tlast == true`.
void drive_data(stream_t& s_in, int tx, int nsamp,
                const float* samples, bool tlast, int driven) {
    ScaleCmdHdr h;
    h.cmd_type = ScaleCmdType::DATA;
    h.tx_id    = (unsigned short)tx;
    h.nsamp    = (unsigned short)nsamp;
    h.write_axi4_stream<32>(s_in, true);

    if (driven > 0) {
        float32_array_utils::write_axi4_stream<32>(s_in, samples, tlast, driven);
    }
}

void drive_end(stream_t& s_in, int tx) {
    ScaleCmdHdr h;
    h.cmd_type = ScaleCmdType::END;
    h.tx_id    = (unsigned short)tx;
    h.nsamp    = 0;
    h.write_axi4_stream<32>(s_in, true);
}

// --------------------------------------------------------------------------
// Response helpers (read the output stream).
// --------------------------------------------------------------------------

void read_resp(stream_t& m_out, int& tx) {
    ScaleRespHdr r;
    streamutils::tlast_status tl = streamutils::tlast_status::no_tlast;
    r.read_axi4_stream<32>(m_out, tl);
    tx = (int)r.tx_id;
}

void read_outputs(stream_t& m_out, std::vector<float>& out, int n) {
    float buf[128];
    streamutils::tlast_status tl = streamutils::tlast_status::no_tlast;
    float32_array_utils::read_axi4_stream<32>(m_out, buf, tl, n);
    for (int i = 0; i < n; ++i) out.push_back(buf[i]);
}

// --------------------------------------------------------------------------
// Evidence writer.
// --------------------------------------------------------------------------

void write_evidence(const std::string& data_dir, const std::string& name,
                    const std::vector<float>& outputs,
                    int halted, int error, int tx_id) {
    const std::string dir = data_dir + "/" + name + "/";
    {
        std::ofstream ofs((dir + "outputs.bin").c_str(), std::ios::binary);
        if (!ofs) throw std::runtime_error("Failed to open outputs.bin for " + name);
        for (float f : outputs) ofs.write(reinterpret_cast<const char*>(&f), sizeof(float));
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

// Run one activation and return nothing; the caller reads the status out-ports
// after the kernel returns.
void activate(stream_t& s_in, stream_t& m_out,
              ap_uint<1>& halted, ap_uint<8>& error, ap_uint<16>& tx_id) {
    halted = 0;
    error  = 0;
    tx_id  = 0;
    scale_offset(s_in, m_out, halted, error, tx_id);
}

} // namespace

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    // ----------------------------------------------------------------------
    // 1. basic_handcheck : CONFIG(a=2,b=1) -> DATA(3) -> END.  [common]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 1, 2.0f, 1.0f);
        { const float s[3] = {0.5f, -1.25f, 3.0f}; drive_data(s_in, 2, 3, s, true, 3); }
        drive_end(s_in, 3);
        activate(s_in, m_out, halted, error, tx_id);

        int t;
        read_resp(m_out, t);                       // CONFIG ack
        read_resp(m_out, t);                       // DATA ack
        std::vector<float> out;
        read_outputs(m_out, out, 3);               // DATA payload
        read_resp(m_out, t);                       // END ack
        write_evidence(data_dir, "basic_handcheck", out,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 2. two_data : one CONFIG, two DATA bursts, one END.  [common]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 1, 2.0f, 1.0f);
        { const float s[2] = {0.5f, -1.25f}; drive_data(s_in, 2, 2, s, true, 2); }
        { const float s[2] = {3.0f, 0.25f}; drive_data(s_in, 3, 2, s, true, 2); }
        drive_end(s_in, 4);
        activate(s_in, m_out, halted, error, tx_id);

        int t;
        read_resp(m_out, t);                       // CONFIG ack
        read_resp(m_out, t);                       // DATA #1 ack
        std::vector<float> out;
        read_outputs(m_out, out, 2);
        read_resp(m_out, t);                       // DATA #2 ack
        read_outputs(m_out, out, 2);
        read_resp(m_out, t);                       // END ack
        write_evidence(data_dir, "two_data", out,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 3. reconfig : CONFIG -> DATA -> CONFIG -> DATA -> END.  [common]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 1,  2.0f, 1.0f);
        { const float s[1] = {1.0f};        drive_data(s_in, 2, 1, s, true, 1); }
        drive_config(s_in, 3, -1.0f, 0.5f);
        { const float s[2] = {2.0f, -3.0f}; drive_data(s_in, 4, 2, s, true, 2); }
        drive_end(s_in, 5);
        activate(s_in, m_out, halted, error, tx_id);

        int t;
        read_resp(m_out, t);                       // CONFIG #1 ack
        read_resp(m_out, t);                       // DATA #1 ack
        std::vector<float> out;
        read_outputs(m_out, out, 1);
        read_resp(m_out, t);                       // CONFIG #2 ack
        read_resp(m_out, t);                       // DATA #2 ack
        read_outputs(m_out, out, 2);
        read_resp(m_out, t);                       // END ack
        write_evidence(data_dir, "reconfig", out,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 4. err_data_before_config : DATA before any CONFIG -> halt.  [common]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        { const float s[2] = {0.5f, 1.5f}; drive_data(s_in, 7, 2, s, true, 2); }
        activate(s_in, m_out, halted, error, tx_id);

        int t;
        read_resp(m_out, t);                       // DATA ack (then kernel halts)
        std::vector<float> out;                    // no outputs on the error path
        write_evidence(data_dir, "err_data_before_config", out,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 5. err_zero_samples : DATA with nsamp = 0 -> halt.  [common]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 9, 1.0f, 0.0f);
        drive_data(s_in, 10, 0, nullptr, true, 0); // nsamp = 0: no payload
        activate(s_in, m_out, halted, error, tx_id);

        int t;
        read_resp(m_out, t);                       // CONFIG ack
        read_resp(m_out, t);                       // DATA ack (then kernel halts)
        std::vector<float> out;
        write_evidence(data_dir, "err_zero_samples", out,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 6. fresh_pass : the clean recovery that follows each error.  [common]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 20, -2.0f, 3.0f);
        { const float s[2] = {1.5f, -0.5f}; drive_data(s_in, 21, 2, s, true, 2); }
        drive_end(s_in, 22);
        activate(s_in, m_out, halted, error, tx_id);

        int t;
        read_resp(m_out, t);                       // CONFIG ack
        read_resp(m_out, t);                       // DATA ack
        std::vector<float> out;
        read_outputs(m_out, out, 2);
        read_resp(m_out, t);                       // END ack
        write_evidence(data_dir, "fresh_pass", out,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 7. err_tlast_early : DATA declares nsamp=3 but the burst is 2 (early
    //    TLAST) -> halt.  [C-sim only]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 30, 2.0f, 1.0f);
        { const float s[3] = {0.5f, -1.25f, 3.0f};
          drive_data(s_in, 31, 3, s, true, 2); }   // nsamp=3, drive 2 with TLAST
        activate(s_in, m_out, halted, error, tx_id);

        int t;
        read_resp(m_out, t);                       // CONFIG ack
        read_resp(m_out, t);                       // DATA ack (then kernel halts)
        std::vector<float> out;
        write_evidence(data_dir, "err_tlast_early", out,
                       (int)halted, (int)error, (int)tx_id);
    }

    // ----------------------------------------------------------------------
    // 8. err_no_tlast : DATA declares nsamp=3, burst is 3 but with no TLAST
    //    -> halt.  [C-sim only]
    // ----------------------------------------------------------------------
    {
        stream_t s_in, m_out;
        ap_uint<1> halted; ap_uint<8> error; ap_uint<16> tx_id;
        drive_config(s_in, 40, 2.0f, 1.0f);
        { const float s[3] = {0.5f, -1.25f, 3.0f};
          drive_data(s_in, 41, 3, s, false, 3); }  // nsamp=3, drive 3, no TLAST
        activate(s_in, m_out, halted, error, tx_id);

        int t;
        read_resp(m_out, t);                       // CONFIG ack
        read_resp(m_out, t);                       // DATA ack (then kernel halts)
        std::vector<float> out;
        write_evidence(data_dir, "err_no_tlast", out,
                       (int)halted, (int)error, (int)tx_id);
    }

    return 0;
}
