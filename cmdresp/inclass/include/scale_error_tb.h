#ifndef INCLUDE_SCALE_ERROR_TB_H
#define INCLUDE_SCALE_ERROR_TB_H

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include "streamutils_tb.h"

#define WAVEFLOW_ENABLE_SCALE_ERROR_TB_H_MEMBERS
#include "scale_error.h"
#undef WAVEFLOW_ENABLE_SCALE_ERROR_TB_H_MEMBERS

inline const char* enum_to_string(ScaleError value) {
    switch (value) {
    case ScaleError::NO_ERROR:
        return "NO_ERROR";
    case ScaleError::DATA_BEFORE_CONFIG:
        return "DATA_BEFORE_CONFIG";
    case ScaleError::ZERO_SAMPLES:
        return "ZERO_SAMPLES";
    case ScaleError::TLAST_EARLY:
        return "TLAST_EARLY";
    case ScaleError::NO_TLAST:
        return "NO_TLAST";
    default:
        return "UNKNOWN";
    }
}

#endif // INCLUDE_SCALE_ERROR_TB_H