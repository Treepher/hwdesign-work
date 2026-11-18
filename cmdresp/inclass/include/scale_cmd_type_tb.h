#ifndef INCLUDE_SCALE_CMD_TYPE_TB_H
#define INCLUDE_SCALE_CMD_TYPE_TB_H

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include "streamutils_tb.h"

#define WAVEFLOW_ENABLE_SCALE_CMD_TYPE_TB_H_MEMBERS
#include "scale_cmd_type.h"
#undef WAVEFLOW_ENABLE_SCALE_CMD_TYPE_TB_H_MEMBERS

inline const char* enum_to_string(ScaleCmdType value) {
    switch (value) {
    case ScaleCmdType::CONFIG:
        return "CONFIG";
    case ScaleCmdType::DATA:
        return "DATA";
    case ScaleCmdType::END:
        return "END";
    default:
        return "UNKNOWN";
    }
}

#endif // INCLUDE_SCALE_CMD_TYPE_TB_H