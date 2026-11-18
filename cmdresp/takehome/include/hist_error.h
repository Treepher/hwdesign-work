#ifndef INCLUDE_HIST_ERROR_H
#define INCLUDE_HIST_ERROR_H

#include <ap_int.h>
#include <hls_stream.h>
#if __has_include(<hls_axi_stream.h>)
#include <hls_axi_stream.h>
#else
#include <ap_axi_sdata.h>
#endif
#include "streamutils_hls.h"

enum class HistError {
    NO_ERROR = 0,
    DATA_BEFORE_CONFIG = 1,
    INVALID_CONFIG = 2,
    ZERO_SAMPLES = 3,
    SAMPLE_OUT_OF_RANGE = 4,
    TLAST_EARLY = 5,
    NO_TLAST = 6,
};

#endif // INCLUDE_HIST_ERROR_H