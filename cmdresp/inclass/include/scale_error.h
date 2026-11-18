#ifndef INCLUDE_SCALE_ERROR_H
#define INCLUDE_SCALE_ERROR_H

#include <ap_int.h>
#include <hls_stream.h>
#if __has_include(<hls_axi_stream.h>)
#include <hls_axi_stream.h>
#else
#include <ap_axi_sdata.h>
#endif
#include "streamutils_hls.h"

enum class ScaleError {
    NO_ERROR = 0,
    DATA_BEFORE_CONFIG = 1,
    ZERO_SAMPLES = 2,
    TLAST_EARLY = 3,
    NO_TLAST = 4,
};

#endif // INCLUDE_SCALE_ERROR_H