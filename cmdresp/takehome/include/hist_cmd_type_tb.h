#ifndef INCLUDE_HIST_CMD_TYPE_TB_H
#define INCLUDE_HIST_CMD_TYPE_TB_H

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include "streamutils_tb.h"

#define WAVEFLOW_ENABLE_HIST_CMD_TYPE_TB_H_MEMBERS
#include "hist_cmd_type.h"
#undef WAVEFLOW_ENABLE_HIST_CMD_TYPE_TB_H_MEMBERS

inline const char* enum_to_string(HistCmdType value) {
    switch (value) {
    case HistCmdType::CONFIG:
        return "CONFIG";
    case HistCmdType::DATA:
        return "DATA";
    case HistCmdType::END:
        return "END";
    default:
        return "UNKNOWN";
    }
}

#endif // INCLUDE_HIST_CMD_TYPE_TB_H