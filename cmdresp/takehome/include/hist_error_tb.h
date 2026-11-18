#ifndef INCLUDE_HIST_ERROR_TB_H
#define INCLUDE_HIST_ERROR_TB_H

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include "streamutils_tb.h"

#define WAVEFLOW_ENABLE_HIST_ERROR_TB_H_MEMBERS
#include "hist_error.h"
#undef WAVEFLOW_ENABLE_HIST_ERROR_TB_H_MEMBERS

inline const char* enum_to_string(HistError value) {
    switch (value) {
    case HistError::NO_ERROR:
        return "NO_ERROR";
    case HistError::DATA_BEFORE_CONFIG:
        return "DATA_BEFORE_CONFIG";
    case HistError::INVALID_CONFIG:
        return "INVALID_CONFIG";
    case HistError::ZERO_SAMPLES:
        return "ZERO_SAMPLES";
    case HistError::SAMPLE_OUT_OF_RANGE:
        return "SAMPLE_OUT_OF_RANGE";
    case HistError::TLAST_EARLY:
        return "TLAST_EARLY";
    case HistError::NO_TLAST:
        return "NO_TLAST";
    default:
        return "UNKNOWN";
    }
}

#endif // INCLUDE_HIST_ERROR_TB_H