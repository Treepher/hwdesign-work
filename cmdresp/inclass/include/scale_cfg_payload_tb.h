#ifndef INCLUDE_SCALE_CFG_PAYLOAD_TB_H
#define INCLUDE_SCALE_CFG_PAYLOAD_TB_H

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include "streamutils_tb.h"

#define WAVEFLOW_ENABLE_SCALE_CFG_PAYLOAD_TB_H_MEMBERS
#include "scale_cfg_payload.h"
#undef WAVEFLOW_ENABLE_SCALE_CFG_PAYLOAD_TB_H_MEMBERS

inline void ScaleCfgPayload::dump_json(std::ostream& os, int indent, int level) const {
    const int step = (indent < 0) ? 0 : indent;
    os << "{";
    os << "\n";
    for (int i = 0; i < (level + 1) * step; ++i) { os << ' '; }
    os << "\"a\": ";
    os << this->a;
    os << ",";
    os << "\n";
    for (int i = 0; i < (level + 1) * step; ++i) { os << ' '; }
    os << "\"b\": ";
    os << this->b;
    os << "\n";
    for (int i = 0; i < (level) * step; ++i) { os << ' '; }
    os << "}";
}

inline void ScaleCfgPayload::load_json(const std::string& json_text, size_t& pos) {
    streamutils::json_expect_char(json_text, pos, '{');
    bool seen_root_a = false;
    bool seen_root_b = false;
    bool first = true;
    while (true) {
    streamutils::json_skip_ws(json_text, pos);
    if (pos < json_text.size() && json_text[pos] == '}') {
        ++pos;
        break;
    }
    if (!first) {
        streamutils::json_expect_char(json_text, pos, ',');
    }
    first = false;
    std::string key = streamutils::json_parse_string(json_text, pos);
    streamutils::json_expect_char(json_text, pos, ':');
    if (key == "a") {
        seen_root_a = true;
        this->a = static_cast<float>(streamutils::json_parse_number(json_text, pos));
    }
    else if (key == "b") {
        seen_root_b = true;
        this->b = static_cast<float>(streamutils::json_parse_number(json_text, pos));
    }
    else {
        throw std::runtime_error("Malformed JSON: unexpected key for schema.");
    }
    }
    if (!seen_root_a) {
    throw std::runtime_error("Malformed JSON: missing required key 'a'.");
    }
    if (!seen_root_b) {
    throw std::runtime_error("Malformed JSON: missing required key 'b'.");
    }
}

inline void ScaleCfgPayload::load_json(std::istream& is) {
    std::string json_text((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    size_t pos = 0;
    streamutils::json_skip_ws(json_text, pos);
    this->load_json(json_text, pos);
    streamutils::json_skip_ws(json_text, pos);
    if (pos != json_text.size()) {
        throw std::runtime_error("Trailing characters after JSON object.");
    }
}

inline void ScaleCfgPayload::dump_json_file(const char* file_path, int indent) const {
    std::ofstream ofs(file_path);
    if (!ofs) {
        throw std::runtime_error("Failed to open output JSON file.");
    }
    this->dump_json(ofs, indent);
}

inline void ScaleCfgPayload::load_json_file(const char* file_path) {
    std::ifstream ifs(file_path);
    if (!ifs) {
        throw std::runtime_error("Failed to open input JSON file.");
    }
    this->load_json(ifs);
}

#endif // INCLUDE_SCALE_CFG_PAYLOAD_TB_H