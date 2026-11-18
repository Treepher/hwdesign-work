#ifndef INCLUDE_CFG_PAYLOAD_H
#define INCLUDE_CFG_PAYLOAD_H

#include <ap_int.h>
#include <hls_stream.h>
#if __has_include(<hls_axi_stream.h>)
#include <hls_axi_stream.h>
#else
#include <ap_axi_sdata.h>
#endif
#include "streamutils_hls.h"

struct CfgPayload {
    ap_uint<16> nbin;  // Number of bins (1..16)
    float lo;  // Lower bound (inclusive)
    float hi;  // Upper bound (exclusive)

    static constexpr int bitwidth = 80;

    template<int word_bw>
    struct word_bw_tag {};

    template<int word_bw>
    static constexpr int nwords_value(word_bw_tag<word_bw>) {
            static_assert(word_bw < 0, "Unsupported word_bw for nwords");
            return 0;
    }

    static constexpr int nwords_value(word_bw_tag<32>) {
            return 3;
    }

    static constexpr int nwords_value(word_bw_tag<64>) {
            return 2;
    }

    template<int word_bw>
    static constexpr int nwords() {
        return nwords_value(word_bw_tag<word_bw>{});
    }

    static ap_uint<bitwidth> pack_to_uint(const CfgPayload& data) {
        ap_uint<bitwidth> res = 0;
        res.range(15, 0) = data.nbin;
        res.range(47, 16) = streamutils::float_to_uint(data.lo);
        res.range(79, 48) = streamutils::float_to_uint(data.hi);
        return res;
    }

    static CfgPayload unpack_from_uint(const ap_uint<bitwidth>& packed) {
        CfgPayload data;
        data.nbin = (ap_uint<16>)(packed.range(15, 0));
        data.lo = streamutils::uint_to_float((uint32_t)(packed.range(47, 16)));
        data.hi = streamutils::uint_to_float((uint32_t)(packed.range(79, 48)));
        return data;
    }

    template<int word_bw>
    static void write_array_impl(word_bw_tag<word_bw>, const CfgPayload* self, ap_uint<word_bw> x[]) {
        static_assert(word_bw < 0, "Unsupported word_bw for write_array");
        (void)self;
        (void)x;
    }

    static void write_array_impl(word_bw_tag<32>, const CfgPayload* self, ap_uint<32> x[]) {
        x[0] = 0;
        x[0].range(15, 0) = self->nbin;
        x[1] = streamutils::float_to_uint(self->lo);
        x[2] = streamutils::float_to_uint(self->hi);
    }

    static void write_array_impl(word_bw_tag<64>, const CfgPayload* self, ap_uint<64> x[]) {
        x[0] = 0;
        x[0].range(15, 0) = self->nbin;
        x[0].range(47, 16) = streamutils::float_to_uint(self->lo);
        x[1] = 0;
        x[1].range(31, 0) = streamutils::float_to_uint(self->hi);
    }

    template<int word_bw>
    void write_array(ap_uint<word_bw> x[]) const {
        write_array_impl(word_bw_tag<word_bw>{}, this, x);
    }

    template<int word_bw>
    static void write_stream_impl(word_bw_tag<word_bw>, const CfgPayload* self, hls::stream<ap_uint<word_bw>> &s) {
        static_assert(word_bw < 0, "Unsupported word_bw for write_stream");
        (void)self;
        (void)s;
    }

    static void write_stream_impl(word_bw_tag<32>, const CfgPayload* self, hls::stream<ap_uint<32>> &s) {
            ap_uint<32> w = 0;
        w.range(15, 0) = self->nbin;
        s.write(w);
        w = 0;
        w = streamutils::float_to_uint(self->lo);
        s.write(w);
        w = 0;
        w = streamutils::float_to_uint(self->hi);
        s.write(w);
        w = 0;
    }

    static void write_stream_impl(word_bw_tag<64>, const CfgPayload* self, hls::stream<ap_uint<64>> &s) {
            ap_uint<64> w = 0;
        w.range(15, 0) = self->nbin;
        w.range(47, 16) = streamutils::float_to_uint(self->lo);
        s.write(w);
        w = 0;
        w.range(31, 0) = streamutils::float_to_uint(self->hi);
        s.write(w);
    }

    template<int word_bw>
    void write_stream(hls::stream<ap_uint<word_bw>> &s) const {
        write_stream_impl(word_bw_tag<word_bw>{}, this, s);
    }

    template<int word_bw>
    static void write_axi4_stream_impl(word_bw_tag<word_bw>, const CfgPayload* self, hls::stream<streamutils::axi4s_word<word_bw>> &s, bool tlast) {
        static_assert(word_bw < 0, "Unsupported word_bw for write_axi4_stream");
        (void)self;
        (void)s;
        (void)tlast;
    }

    static void write_axi4_stream_impl(word_bw_tag<32>, const CfgPayload* self, hls::stream<streamutils::axi4s_word<32>> &s, bool tlast) {
            ap_uint<32> w = 0;
        w.range(15, 0) = self->nbin;
        streamutils::write_axi4_word<32>(s, w, false);
        w = 0;
        w = streamutils::float_to_uint(self->lo);
        streamutils::write_axi4_word<32>(s, w, false);
        w = 0;
        w = streamutils::float_to_uint(self->hi);
        streamutils::write_axi4_word<32>(s, w, tlast);
        w = 0;
    }

    static void write_axi4_stream_impl(word_bw_tag<64>, const CfgPayload* self, hls::stream<streamutils::axi4s_word<64>> &s, bool tlast) {
            ap_uint<64> w = 0;
        w.range(15, 0) = self->nbin;
        w.range(47, 16) = streamutils::float_to_uint(self->lo);
        streamutils::write_axi4_word<64>(s, w, false);
        w = 0;
        w.range(31, 0) = streamutils::float_to_uint(self->hi);
        streamutils::write_axi4_word<64>(s, w, tlast);
    }

    template<int word_bw>
    void write_axi4_stream(hls::stream<streamutils::axi4s_word<word_bw>> &s, bool tlast = true) const {
        write_axi4_stream_impl(word_bw_tag<word_bw>{}, this, s, tlast);
    }

    template<int word_bw>
    static void read_array_impl(word_bw_tag<word_bw>, CfgPayload* self, const ap_uint<word_bw> x[]) {
        static_assert(word_bw < 0, "Unsupported word_bw for read_array");
        (void)self;
        (void)x;
    }

    static void read_array_impl(word_bw_tag<32>, CfgPayload* self, const ap_uint<32> x[]) {
        self->nbin = (ap_uint<16>)(x[0].range(15, 0));
        self->lo = streamutils::uint_to_float((uint32_t)(x[1]));
        self->hi = streamutils::uint_to_float((uint32_t)(x[2]));
    }

    static void read_array_impl(word_bw_tag<64>, CfgPayload* self, const ap_uint<64> x[]) {
        self->nbin = (ap_uint<16>)(x[0].range(15, 0));
        self->lo = streamutils::uint_to_float((uint32_t)(x[0].range(47, 16)));
        self->hi = streamutils::uint_to_float((uint32_t)(x[1].range(31, 0)));
    }

    template<int word_bw>
    void read_array(const ap_uint<word_bw> x[]) {
        read_array_impl(word_bw_tag<word_bw>{}, this, x);
    }

    template<int word_bw>
    static void read_stream_impl(word_bw_tag<word_bw>, CfgPayload* self, hls::stream<ap_uint<word_bw>> &s) {
        static_assert(word_bw < 0, "Unsupported word_bw for read_stream");
        (void)self;
        (void)s;
    }

    static void read_stream_impl(word_bw_tag<32>, CfgPayload* self, hls::stream<ap_uint<32>> &s) {
            ap_uint<32> w = 0;
        w = s.read();
        self->nbin = (ap_uint<16>)(w.range(15, 0));
        w = s.read();
        self->lo = streamutils::uint_to_float((uint32_t)(w));
        w = s.read();
        self->hi = streamutils::uint_to_float((uint32_t)(w));
    }

    static void read_stream_impl(word_bw_tag<64>, CfgPayload* self, hls::stream<ap_uint<64>> &s) {
            ap_uint<64> w = 0;
        w = s.read();
        self->nbin = (ap_uint<16>)(w.range(15, 0));
        self->lo = streamutils::uint_to_float((uint32_t)(w.range(47, 16)));
        w = s.read();
        self->hi = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
    }

    template<int word_bw>
    void read_stream(hls::stream<ap_uint<word_bw>> &s) {
        read_stream_impl(word_bw_tag<word_bw>{}, this, s);
    }

    template<int word_bw>
    static void read_axi4_stream_impl(word_bw_tag<word_bw>, CfgPayload* self, hls::stream<streamutils::axi4s_word<word_bw>> &s, streamutils::tlast_status &tl) {
        static_assert(word_bw < 0, "Unsupported word_bw for read_axi4_stream");
        (void)self;
        (void)s;
        (void)tl;
    }

    static void read_axi4_stream_impl(word_bw_tag<32>, CfgPayload* self, hls::stream<streamutils::axi4s_word<32>> &s, streamutils::tlast_status &tl) {
            ap_uint<32> w = 0;
            tl = streamutils::tlast_status::no_tlast;
            bool last = false;
        if (last) {
            tl = streamutils::tlast_status::tlast_early;
            return;
        }
        {
            auto axis_word = s.read();
            w = axis_word.data;
            last = axis_word.last;
        }
        self->nbin = (ap_uint<16>)(w.range(15, 0));
        if (tl != streamutils::tlast_status::no_tlast) {
            tl = streamutils::tlast_status::tlast_early;
            return;
        }
        if (last) {
            tl = streamutils::tlast_status::tlast_early;
            return;
        }
        {
            auto axis_word = s.read();
            w = axis_word.data;
            last = axis_word.last;
        }
        self->lo = streamutils::uint_to_float((uint32_t)(w));
        if (tl != streamutils::tlast_status::no_tlast) {
            tl = streamutils::tlast_status::tlast_early;
            return;
        }
        if (last) {
            tl = streamutils::tlast_status::tlast_early;
            return;
        }
        {
            auto axis_word = s.read();
            w = axis_word.data;
            last = axis_word.last;
        }
        self->hi = streamutils::uint_to_float((uint32_t)(w));
        if (tl != streamutils::tlast_status::no_tlast) {
            return;
        }
        if (last) {
            tl = streamutils::tlast_status::tlast_at_end;
        }
    }

    static void read_axi4_stream_impl(word_bw_tag<64>, CfgPayload* self, hls::stream<streamutils::axi4s_word<64>> &s, streamutils::tlast_status &tl) {
            ap_uint<64> w = 0;
            tl = streamutils::tlast_status::no_tlast;
            bool last = false;
        if (last) {
            tl = streamutils::tlast_status::tlast_early;
            return;
        }
        {
            auto axis_word = s.read();
            w = axis_word.data;
            last = axis_word.last;
        }
        self->nbin = (ap_uint<16>)(w.range(15, 0));
        if (tl != streamutils::tlast_status::no_tlast) {
            tl = streamutils::tlast_status::tlast_early;
            return;
        }
        self->lo = streamutils::uint_to_float((uint32_t)(w.range(47, 16)));
        if (tl != streamutils::tlast_status::no_tlast) {
            tl = streamutils::tlast_status::tlast_early;
            return;
        }
        if (last) {
            tl = streamutils::tlast_status::tlast_early;
            return;
        }
        {
            auto axis_word = s.read();
            w = axis_word.data;
            last = axis_word.last;
        }
        self->hi = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
        if (tl != streamutils::tlast_status::no_tlast) {
            return;
        }
        if (last) {
            tl = streamutils::tlast_status::tlast_at_end;
        }
    }

    template<int word_bw>
    void read_axi4_stream(hls::stream<streamutils::axi4s_word<word_bw>> &s, streamutils::tlast_status &tl) {
        read_axi4_stream_impl(word_bw_tag<word_bw>{}, this, s, tl);
    }

    template<int word_bw>
    void read_axi4_stream(hls::stream<streamutils::axi4s_word<word_bw>> &s) {
        streamutils::tlast_status tl = streamutils::tlast_status::no_tlast;
        read_axi4_stream<word_bw>(s, tl);
    }

#ifdef WAVEFLOW_ENABLE_CFG_PAYLOAD_TB_H_MEMBERS
    void dump_json(std::ostream& os, int indent = 2, int level = 0) const;
    void load_json(const std::string& json_text, size_t& pos);
    void load_json(std::istream& is);
    void dump_json_file(const char* file_path, int indent = 2) const;
    void load_json_file(const char* file_path);
#endif
};

#endif // INCLUDE_CFG_PAYLOAD_H