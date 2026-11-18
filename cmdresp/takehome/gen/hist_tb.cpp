#include "hist.hpp"
#include "include/streamutils_tb.h"
#include "include/float32_array_utils_tb.h"
#include "include/uint32_array_utils_tb.h"
#include "include/hist_cmd_hdr.h"
#include "include/cfg_payload.h"
#include "include/hist_resp_hdr.h"
#include <fstream>
#include <cstdint>
#include <string>
#include <stdexcept>

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";
    (void)data_dir;
    hls::stream<streamutils::axi4s_word<32>> s_in;
    hls::stream<streamutils::axi4s_word<32>> m_out;
    ap_uint<1> halted = 0;
    ap_uint<8> error = 0;
    ap_uint<16> tx_id = 0;
    HistCmdHdr config_hdr;
    streamutils::read_uint32_file(config_hdr, (data_dir + std::string("/config_cmd_hdr.bin")).c_str());
    CfgPayload config_payload;
    streamutils::read_uint32_file(config_payload, (data_dir + std::string("/config_payload.bin")).c_str());
    HistCmdHdr data_hdr;
    streamutils::read_uint32_file(data_hdr, (data_dir + std::string("/data_cmd_hdr.bin")).c_str());
    float samp_in[128] = {};
    float32_array_utils::read_uint32_file_array(samp_in, (data_dir + std::string("/samp_in_data.bin")).c_str(), data_hdr.nsamp);
    HistCmdHdr end_hdr;
    streamutils::read_uint32_file(end_hdr, (data_dir + std::string("/end_cmd_hdr.bin")).c_str());
    config_hdr.write_axi4_stream<32>(s_in, true);
    config_payload.write_axi4_stream<32>(s_in, true);
    data_hdr.write_axi4_stream<32>(s_in, true);
    float32_array_utils::write_axi4_stream<32>(s_in, samp_in, true, data_hdr.nsamp);
    end_hdr.write_axi4_stream<32>(s_in, true);
    hist(s_in, m_out, halted, error, tx_id);
    HistRespHdr resp_cfg;
    streamutils::tlast_status _tlast_resp_cfg = streamutils::tlast_status::no_tlast;
    resp_cfg.read_axi4_stream<32>(m_out, _tlast_resp_cfg);
    HistRespHdr resp_data;
    streamutils::tlast_status _tlast_resp_data = streamutils::tlast_status::no_tlast;
    resp_data.read_axi4_stream<32>(m_out, _tlast_resp_data);
    HistRespHdr resp_end;
    streamutils::tlast_status _tlast_resp_end = streamutils::tlast_status::no_tlast;
    resp_end.read_axi4_stream<32>(m_out, _tlast_resp_end);
    ap_uint<32> counters[16] = {};
    streamutils::tlast_status _tlast_counters = streamutils::tlast_status::no_tlast;
    uint32_array_utils::read_axi4_stream<32>(m_out, counters, _tlast_counters, config_payload.nbin);
    streamutils::write_uint32_file(resp_cfg, (data_dir + std::string("/resp_cfg_data.bin")).c_str());
    streamutils::write_uint32_file(resp_data, (data_dir + std::string("/resp_data_data.bin")).c_str());
    streamutils::write_uint32_file(resp_end, (data_dir + std::string("/resp_end_data.bin")).c_str());
    uint32_array_utils::write_uint32_file_array(counters, (data_dir + std::string("/counters_data.bin")).c_str(), config_payload.nbin);
    {
        std::ofstream _status_ofs((data_dir + std::string("/regmap_status.json")));
        if (!_status_ofs) {
            throw std::runtime_error("Failed to open status JSON file for writing.");
        }
        _status_ofs << "{\n";
        _status_ofs << "  \"halted\": " << (int)halted << ",\n";
        _status_ofs << "  \"error\": " << (int)error << ",\n";
        _status_ofs << "  \"tx_id\": " << (int)tx_id << "\n";
        _status_ofs << "}\n";
    }
    return 0;
}
