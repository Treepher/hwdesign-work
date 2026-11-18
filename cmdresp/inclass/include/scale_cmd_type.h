#ifndef INCLUDE_SCALE_CMD_TYPE_H
#define INCLUDE_SCALE_CMD_TYPE_H

#include <ap_int.h>
#include <hls_stream.h>
#if __has_include(<hls_axi_stream.h>)
#include <hls_axi_stream.h>
#else
#include <ap_axi_sdata.h>
#endif
#include "streamutils_hls.h"

enum class ScaleCmdType {
    CONFIG = 0,
    DATA = 1,
    END = 2,
};

#endif // INCLUDE_SCALE_CMD_TYPE_H