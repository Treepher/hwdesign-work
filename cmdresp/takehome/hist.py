"""Histogram accelerator — bit-exact Python golden model (take-home).

Command-response streaming pattern, following the `stream_inband` Waveflow
reference (``poly.py``) for structure but differing where the pattern requires:

* Configuration travels on the **input stream as a CONFIG command**, not on the
  AXI-Lite register map.  The regmap therefore carries only the status
  registers ``halted`` / ``error`` / ``tx_id`` (plus ``ap_start`` / ``ap_done``).
* Commands are ``CONFIG`` / ``DATA`` / ``END``.  Counters **accumulate** across
  DATA commands within one activation and are emitted by END.

Every step of the bin-index arithmetic is done in 32-bit float so the Python
model and the C++ kernel produce identical bits.

Run the model end-to-end (all verification scenarios) with::

    python hist.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from waveflow.hw.arrayutils import array, read_array, write_array, write_uint32_file
from waveflow.hw.clock import Clock
from waveflow.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField
from waveflow.hw.hw_module import HwConst, HwParam
from waveflow.hw.hw_hostactivated import HostActivated
from waveflow.hw.hw_state import HwState
from waveflow.hw.interface import StreamIF, StreamIFMaster, StreamIFSlave
from waveflow.hw.memif import DirectMMIF, MMIFMaster
from waveflow.hw.regmap import (
    Bit,
    RegAccess,
    RegField,
    VitisRegMap,
    VitisRegMapMMIFSlave,
)
from waveflow.hw.synth import sim_only, synthesizable
from waveflow.hw.hw_testbench import SeqTB
from waveflow.simulation.logger import Logger, NullLogger
from waveflow.simulation.simobj import ProcessGen, SimObj
from waveflow.simulation.simulation import Simulation


INCLUDE_DIR = "include"
WORD_BW_SUPPORTED = [32, 64]

TxIdField  = IntField.specialize(bitwidth=16, signed=False)
NsampField = IntField.specialize(bitwidth=16, signed=False)
NbinField  = IntField.specialize(bitwidth=16, signed=False)
U32        = IntField.specialize(bitwidth=32, signed=False, include_dir=INCLUDE_DIR)
Float32    = FloatField.specialize(bitwidth=32, include_dir=INCLUDE_DIR)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HistError(IntEnum):
    NO_ERROR            = 0
    DATA_BEFORE_CONFIG  = 1
    INVALID_CONFIG      = 2
    ZERO_SAMPLES        = 3
    SAMPLE_OUT_OF_RANGE = 4
    TLAST_EARLY         = 5
    NO_TLAST            = 6


HistErrorField = EnumField.specialize(enum_type=HistError)


class HistCmdType(IntEnum):
    CONFIG = 0
    DATA   = 1
    END    = 2


HistCmdTypeField = EnumField.specialize(enum_type=HistCmdType)


# ---------------------------------------------------------------------------
# Stream schema
# ---------------------------------------------------------------------------

class CfgPayload(DataList):
    """CONFIG command payload: bin count and the half-open range [lo, hi)."""
    elements = {
        "nbin": {"schema": NbinField, "description": "Number of bins (1..16)"},
        "lo":   {"schema": Float32,   "description": "Lower bound (inclusive)"},
        "hi":   {"schema": Float32,   "description": "Upper bound (exclusive)"},
    }


class HistCmdHdr(DataList):
    """Command header.  ``nsamp`` is the payload word count: 3 for CONFIG
    (the ``CfgPayload``), the sample count for DATA, and 0 for END."""
    elements = {
        "cmd_type": {"schema": HistCmdTypeField, "description": "CONFIG / DATA / END"},
        "tx_id":    {"schema": TxIdField,        "description": "Transaction ID"},
        "nsamp":    {"schema": NsampField,       "description": "Payload word count"},
    }


class HistRespHdr(DataList):
    """Response header: echo of the transaction ID."""
    elements = {
        "tx_id": {"schema": TxIdField, "description": "Echo of the transaction ID"},
    }


class CounterArray(DataArray):
    """Output counters: one 32-bit unsigned count per bin (max 16)."""
    ncounter: HwConst[int] = 16
    element_type = U32
    static = True
    max_shape = (ncounter,)
    cpp_storage = "raw"


class SampArr(DataArray):
    """Float32 input-sample buffer for the codegen-source testbench.

    Sized to ``max_shape=(128,)`` so the C++ testbench can allocate a static
    ``float samp_in[128]``; the runtime element count is ``count=data_hdr.nsamp``.
    """
    element_type = Float32
    static = True
    max_shape = (128,)
    cpp_storage = "raw"


# Persistent per-activation storage, declared via add_state (not the regmap):
# the CONFIG payload and the accumulating counters travel on the STREAM, so the
# kernel keeps them as zero-initialised ``static``s that outlive each command.
# ``cpp_storage="raw"`` lowers each to a bare ``elem name[N]`` — the form a
# static wants, and the form a hook can write through (the array decays to a
# pointer in the hook signature).
NbinArr   = DataArray.specialize(U32,     max_shape=(1,), cpp_storage="raw")  # [nbin]
BoundsArr = DataArray.specialize(Float32, max_shape=(2,), cpp_storage="raw")  # [lo, hi]


SCHEMA_CLASSES = [
    HistErrorField,
    HistCmdTypeField,
    CfgPayload,
    HistCmdHdr,
    HistRespHdr,
    CounterArray,
    SampArr,
    NbinArr,
    BoundsArr,
]


# ---------------------------------------------------------------------------
# Accelerator (SimPy golden model)
# ---------------------------------------------------------------------------

@dataclass
class HistAccel(HostActivated):
    """SimPy golden model of the histogram accelerator kernel.

    Control/status is an AXI-Lite ``VitisRegMap`` of the status registers
    only.  The host writes ``ap_start`` to launch ``on_start``, which loops
    reading CONFIG / DATA / END command headers from the input stream.
    """

    cpp_kernel_name: ClassVar[str | None] = "hist"
    cpp_namespace:   ClassVar[str | None] = "hist_impl"

    in_bw:     HwParam[int] = 32
    out_bw:    HwParam[int] = 32
    aximm_bw:  HwParam[int] = 32
    clk:       Clock = field(default_factory=lambda: Clock(freq=1e9))
    proc_ii:   int = 1
    proc_latency: int = 40
    logger:    Logger | NullLogger = field(default_factory=NullLogger)
    unroll_factor: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.s_in  = StreamIFSlave( name=f'{self.name}_s_in',  sim=self.sim, bitwidth=self.in_bw)
        self.m_out = StreamIFMaster(name=f'{self.name}_m_out', sim=self.sim, bitwidth=self.out_bw)
        self.regmap = VitisRegMap({
            "halted": RegField(Bit,           RegAccess.R, description="1 = halted on error"),
            "error":  RegField(HistErrorField, RegAccess.R, description="Last error code"),
            "tx_id":  RegField(TxIdField,     RegAccess.R, description="TX id of halted txn"),
        }, bitwidth=self.aximm_bw)
        self.s_lite = VitisRegMapMMIFSlave(
            name=f'{self.name}_s_lite', sim=self.sim, bitwidth=self.aximm_bw,
            regmap=self.regmap, on_start=self.on_start,
        )
        for ep in (self.s_in, self.m_out, self.s_lite):
            self.add_endpoint(ep)
        self._job: int = 0
        # Per-activation state, declared via add_state so the generated kernel
        # keeps it as a zero-initialised ``static`` that outlives each command.
        #   nbin     = number of bins   (u32[1]; 0 until a valid CONFIG)
        #   bounds   = [lo, hi)         (f32[2])
        #   counters = 16 bin counts    (u32[16]; reset by every CONFIG)
        # A valid CONFIG sets nbin to >= 1, so ``nbin == 0`` is the "not yet
        # configured" condition the DATA/END hooks check.
        self.nbin     = HwState(NbinArr())
        self.bounds   = HwState(BoundsArr())
        self.counters = HwState(CounterArray())
        self.add_state(self.nbin)
        self.add_state(self.bounds)
        self.add_state(self.counters)

    @sim_only
    def _inc_job(self) -> None:
        self._job += 1

    # -- per-command handlers (synthesizable hooks) ---------------------------
    #
    # Each hook is the pysim twin of one hand-written C++ body
    # (``hist_configure_impl.tpp`` / ``hist_accumulate_impl.tpp`` /
    # ``hist_emit_impl.tpp``).  Persistent storage is passed in as ``HwState``
    # args and indexed through ``.val``; the generated kernel keeps it as a
    # zero-initialised ``static`` and hands the hook a writable pointer to it.

    @synthesizable
    def configure(self, cmd_hdr: HistCmdHdr,
                  s_in: StreamIFSlave, m_out: StreamIFMaster,
                  nbin: HwState, bounds: HwState, counters: HwState
                  ) -> ProcessGen[HistError]:
        """CONFIG: ack, read the payload, validate, set config, reset counters.

        The RespHdr is emitted before the payload is read so the stream stays in
        lockstep (one RespHdr per command header).  An invalid config is
        reported through the status registers by the caller.
        """
        resp = HistRespHdr()
        resp.tx_id = cmd_hdr.tx_id
        self.logger.log(event='config_ack', job=self._job)
        yield from m_out.write(resp)

        cfg = yield from s_in.get_schema(CfgPayload)
        n  = int(cfg.nbin)
        lo = np.float32(cfg.lo)
        hi = np.float32(cfg.hi)

        if n < 1 or n > 16 or lo >= hi:
            return HistError.INVALID_CONFIG

        # Store config; counters start at zero.
        nbin.val[0]     = np.uint32(n)
        bounds.val[0]   = lo
        bounds.val[1]   = hi
        counters.val[:] = 0
        return HistError.NO_ERROR

    @synthesizable
    def accumulate(self, cmd_hdr: HistCmdHdr,
                   s_in: StreamIFSlave, m_out: StreamIFMaster,
                   nbin: HwState, bounds: HwState, counters: HwState
                   ) -> ProcessGen[HistError]:
        """DATA: ack, read ``nsamp`` samples, bin each into the running counters.

        The bin index is ``(x - lo) * scale`` in 32-bit float, truncated toward
        zero, clamped to ``[0, nbin-1]``.  A sample outside ``[lo, hi)`` is an
        error.  The body's NumPy math is the pysim behaviour; the C++ kernel
        supplies the equivalent in ``hist_accumulate_impl.tpp``.
        """
        resp = HistRespHdr()
        resp.tx_id = cmd_hdr.tx_id
        self.logger.log(event='data_ack', job=self._job)
        yield from m_out.write(resp)

        if nbin.val[0] == 0:
            return HistError.DATA_BEFORE_CONFIG
        nsamp = int(cmd_hdr.nsamp)
        if nsamp == 0:
            return HistError.ZERO_SAMPLES

        samp_in, tstart = yield from s_in.get_pipelined(Float32, count=nsamp)
        samp_in = np.asarray(samp_in, dtype=np.float32)

        # Bin each sample (32-bit float arithmetic throughout).
        n     = int(nbin.val[0])
        lo    = np.float32(bounds.val[0])
        hi    = np.float32(bounds.val[1])
        scale = np.float32(n) / (hi - lo)
        diff  = samp_in - lo              # f32
        idx_f = diff * scale              # f32
        idx   = idx_f.astype(np.int64)    # truncate toward zero
        idx   = np.clip(idx, 0, n - 1)
        for i, x in enumerate(samp_in):
            if x < lo or x >= hi:
                return HistError.SAMPLE_OUT_OF_RANGE
        np.add.at(counters.val, idx, 1)

        # Model the per-sample processing time (II=1) for timing fidelity.
        proc_time = nsamp / self.unroll_factor * self.proc_ii * self.clk.period
        t_out_start = tstart + self.proc_latency * self.clk.period
        proc_time = max(0.0, proc_time + (t_out_start - self.env.now))
        if proc_time > 0:
            yield self.timeout(proc_time)

        if len(samp_in) != nsamp:
            return HistError.TLAST_EARLY
        return HistError.NO_ERROR

    @synthesizable
    def emit(self, cmd_hdr: HistCmdHdr, m_out: StreamIFMaster,
             nbin: HwState, counters: HwState
             ) -> ProcessGen[HistError]:
        """END: ack, then (if configured) emit the ``nbin`` counters in bin order."""
        resp = HistRespHdr()
        resp.tx_id = cmd_hdr.tx_id
        self.logger.log(event='end_ack', job=self._job)
        yield from m_out.write(resp)

        if nbin.val[0] == 0:
            return HistError.DATA_BEFORE_CONFIG

        t_out_start = self.env.now
        n = int(nbin.val[0])
        cvals = np.asarray(counters.val, dtype=np.uint32)[:n]
        yield from m_out.write_pipelined(array(U32, cvals), t_out_start)
        self.logger.log(event='counters_write_end', job=self._job)
        return HistError.NO_ERROR

    # -- kernel body ----------------------------------------------------------

    def on_start(self) -> ProcessGen[None]:
        """Kernel body — invoked by ``VitisRegMapMMIFSlave`` on ap_start."""
        while True:
            self.logger.log(event='proc_begin', job=self._job)
            cmd_hdr = yield from self.s_in.get_schema(HistCmdHdr)
            if cmd_hdr.cmd_type == HistCmdType.CONFIG:
                err = yield from self.configure(cmd_hdr, self.s_in, self.m_out,
                                                self.nbin, self.bounds, self.counters)
                self._inc_job()
                if err != HistError.NO_ERROR:
                    self.regmap.set("error",  err)
                    self.regmap.set("tx_id",  cmd_hdr.tx_id)
                    self.regmap.set("halted", 1)
                    return
            elif cmd_hdr.cmd_type == HistCmdType.DATA:
                err = yield from self.accumulate(cmd_hdr, self.s_in, self.m_out,
                                                 self.nbin, self.bounds, self.counters)
                self._inc_job()
                if err != HistError.NO_ERROR:
                    self.regmap.set("error",  err)
                    self.regmap.set("tx_id",  cmd_hdr.tx_id)
                    self.regmap.set("halted", 1)
                    return
            else:  # END — terminal
                err = yield from self.emit(cmd_hdr, self.m_out,
                                           self.nbin, self.counters)
                self._inc_job()
                if err != HistError.NO_ERROR:
                    self.regmap.set("error",  err)
                    self.regmap.set("tx_id",  cmd_hdr.tx_id)
                    self.regmap.set("halted", 1)
                return


# ---------------------------------------------------------------------------
# SimPy testbench
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class HistTB(SimObj):
    """Drives a command sequence (CONFIG / DATA... / END) and captures the
    responses plus the final ``halted`` / ``error`` / ``tx_id`` status.

    ``commands`` is a list of dicts:
        {"type": "config", "tx_id": .., "nbin": .., "lo": .., "hi": ..}
        {"type": "data",   "tx_id": .., "nsamp": .., "samples": np.ndarray}
        {"type": "end",    "tx_id": .., "nbin": ..}
    """
    commands:  list[dict]
    word_bw:   int = 32
    base_addr: int = 0x0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.m_in   = StreamIFMaster(name=f'{self.name}_m_in',   sim=self.sim, bitwidth=self.word_bw)
        self.s_out  = StreamIFSlave( name=f'{self.name}_s_out',  sim=self.sim, bitwidth=self.word_bw)
        self.m_lite = MMIFMaster(    name=f'{self.name}_m_lite', sim=self.sim, bitwidth=32)
        self.resp_hdrs:     list[HistRespHdr] = []
        self.counters:      npt.NDArray[np.uint32] | None = None
        self.halted:        int | None = None
        self.error:         HistError | None = None
        self.tx_id_status:  int | None = None
        self._regmap_ref:   VitisRegMap | None = None

    def _regmap(self) -> VitisRegMap:
        if self._regmap_ref is None:
            raise RuntimeError("HistTB._regmap_ref is unset; call connect() before run_sim().")
        return self._regmap_ref

    def _read_resp(self) -> ProcessGen[HistRespHdr]:
        words = yield from self.s_out.get()
        return HistRespHdr().deserialize(words, word_bw=self.word_bw)

    def _halted(self, rm) -> ProcessGen[bool]:
        """Read the ``halted`` status bit; the kernel sets it (with error/tx_id)
        on any error, then returns — the host must not drive further commands."""
        return (yield from rm.get("halted")) != 0

    def run_proc(self) -> ProcessGen[None]:
        bw = self.word_bw
        rm = self._regmap().bind_master(self.m_lite, base_addr=self.base_addr)
        yield from rm.start()

        halted = False
        for cmd in self.commands:
            if halted:
                break  # activation already over — stream contents are undefined

            if cmd["type"] == "config":
                hdr = HistCmdHdr()
                hdr.cmd_type = HistCmdType.CONFIG
                hdr.tx_id    = int(cmd["tx_id"])
                hdr.nsamp    = 3
                yield from self.m_in.write(hdr.serialize(word_bw=bw))
                payload = CfgPayload()
                payload.nbin = int(cmd["nbin"])
                payload.lo   = float(np.float32(cmd["lo"]))
                payload.hi   = float(np.float32(cmd["hi"]))
                yield from self.m_in.write(payload.serialize(word_bw=bw))
                resp = yield from self._read_resp()
                self.resp_hdrs.append(resp)

            elif cmd["type"] == "data":
                nsamp = int(cmd["nsamp"])
                hdr = HistCmdHdr()
                hdr.cmd_type = HistCmdType.DATA
                hdr.tx_id    = int(cmd["tx_id"])
                hdr.nsamp    = nsamp
                yield from self.m_in.write(hdr.serialize(word_bw=bw))
                samples = np.asarray(cmd["samples"], dtype=np.float32)[:nsamp]
                yield from self.m_in.write(write_array(samples, elem_type=Float32, word_bw=bw))
                resp = yield from self._read_resp()
                self.resp_hdrs.append(resp)

            else:  # end
                hdr = HistCmdHdr()
                hdr.cmd_type = HistCmdType.END
                hdr.tx_id    = int(cmd["tx_id"])
                hdr.nsamp    = 0
                yield from self.m_in.write(hdr.serialize(word_bw=bw))
                resp = yield from self._read_resp()
                self.resp_hdrs.append(resp)
                if not (yield from self._halted(rm)):
                    n = int(cmd["nbin"])
                    cwords = yield from self.s_out.get()
                    self.counters = np.asarray(
                        read_array(cwords, elem_type=U32, word_bw=bw, shape=n),
                        dtype=np.uint32,
                    )

            halted = yield from self._halted(rm)

        # Wait for the kernel to finish (ap_done) so the status write has landed,
        # then read the final status registers.
        try:
            yield from rm.poll_end(interval=0.0, max_polls=1000)
        except RuntimeError:
            pass
        self.halted       = yield from rm.get("halted")
        self.error        = yield from rm.get("error")
        self.tx_id_status = yield from rm.get("tx_id")


def connect(sim: Simulation, tb: HistTB, accel: HistAccel, clk: Clock) -> None:
    """Wire the testbench to the accelerator via two StreamIFs and a DirectMMIF."""
    in_stream  = StreamIF(sim=sim, clk=clk)
    out_stream = StreamIF(sim=sim, clk=clk)
    lite_link  = DirectMMIF(sim=sim, clk=clk, byte_addressable=True)
    in_stream.bind( "master", tb.m_in)
    in_stream.bind( "slave",  accel.s_in)
    out_stream.bind("master", accel.m_out)
    out_stream.bind("slave",  tb.s_out)
    lite_link.bind( "master", tb.m_lite)
    lite_link.bind( "slave",  accel.s_lite)
    tb._regmap_ref = accel.regmap


class HistTBHls(SeqTB):
    """Codegen-source testbench for the histogram kernel (lowered to gen/hist_tb.cpp).

    ``main()`` is a sequential Python program: instantiate the DUT, drive one
    clean CONFIG -> DATA -> END activation, run the kernel, drain the three
    response headers plus the ``nbin`` counters, and emit ``regmap_status.json``.
    The codegen pipeline lowers this directly to ``gen/hist_tb.cpp``.

    Coexists with :class:`HistTB` (the SimPy timing-accurate model used by
    ``PySimStep``); ``HistTBHls`` is the codegen-only counterpart.
    """

    cpp_kernel_name: ClassVar[str | None] = "hist"

    def main(self) -> None:
        dut = HistAccel()

        config_hdr = HistCmdHdr()
        config_hdr.read_uint32_file(self.data_dir + "/config_cmd_hdr.bin")
        config_payload = CfgPayload()
        config_payload.read_uint32_file(self.data_dir + "/config_payload.bin")
        data_hdr = HistCmdHdr()
        data_hdr.read_uint32_file(self.data_dir + "/data_cmd_hdr.bin")
        samp_in = SampArr()
        samp_in.read_uint32_file_array(self.data_dir + "/samp_in_data.bin", count=data_hdr.nsamp)
        end_hdr = HistCmdHdr()
        end_hdr.read_uint32_file(self.data_dir + "/end_cmd_hdr.bin")

        dut.s_in.push(config_hdr)
        dut.s_in.push(config_payload)
        dut.s_in.push(data_hdr)
        dut.s_in.push_array(samp_in, count=data_hdr.nsamp)
        dut.s_in.push(end_hdr)

        dut.run()

        resp_cfg = HistRespHdr();  dut.m_out.pop(resp_cfg)
        resp_data = HistRespHdr(); dut.m_out.pop(resp_data)
        resp_end = HistRespHdr();  dut.m_out.pop(resp_end)
        counters = CounterArray(); dut.m_out.pop_array(counters, count=config_payload.nbin)

        resp_cfg.write_uint32_file(self.data_dir + "/resp_cfg_data.bin")
        resp_data.write_uint32_file(self.data_dir + "/resp_data_data.bin")
        resp_end.write_uint32_file(self.data_dir + "/resp_end_data.bin")
        counters.write_uint32_file_array(self.data_dir + "/counters_data.bin", count=config_payload.nbin)
        dut.regmap.write_status_json(self.data_dir + "/regmap_status.json",
                                     fields=["halted", "error", "tx_id"])


# ---------------------------------------------------------------------------
# Verification scenarios
# ---------------------------------------------------------------------------

def run_scenario(sim, clk, commands, logger=None) -> HistTB:
    logger = logger or NullLogger()
    accel = HistAccel(name="hist_accel", sim=sim, clk=clk, logger=logger,
                      in_bw=32, out_bw=32)
    tb = HistTB(name="hist_tb", sim=sim, commands=commands, word_bw=32)
    connect(sim, tb, accel, clk)
    sim.run_sim()
    return tb


def reference_histogram(samples, nbin, lo, hi) -> np.ndarray:
    """The prompt's bin rule, computed independently (hand-checkable)."""
    nbin = int(nbin); lo = np.float32(lo); hi = np.float32(hi)
    scale = np.float32(nbin) / (hi - lo)
    out = np.zeros(nbin, dtype=np.uint32)
    for x in samples:
        x = np.float32(x)
        if x < lo or x >= hi:
            raise ValueError("out-of-range sample in reference")
        idx = int(np.float32(np.float32(x - lo) * scale))
        idx = min(max(idx, 0), nbin - 1)
        out[idx] += 1
    return out


SCENARIOS: list[tuple[str, list[dict], dict]] = [
    # 1. Basic hand-check: B=4, lo=0, hi=4, six samples -> [2,1,1,2]
    ("basic_handcheck",
     [
         {"type": "config", "tx_id": 1, "nbin": 4, "lo": 0.0, "hi": 4.0},
         {"type": "data",   "tx_id": 2, "nsamp": 6,
          "samples": [0.2, 1.8, 2.1, 3.9, 0.0, 3.2]},
         {"type": "end",    "tx_id": 3, "nbin": 4},
     ],
     {"counters": [2, 1, 1, 2], "error": 0, "halted": 0}),

    # 2. Two DATA commands that accumulate, then END.
    ("accumulate_two_data",
     [
         {"type": "config", "tx_id": 1, "nbin": 4, "lo": 0.0, "hi": 4.0},
         {"type": "data",   "tx_id": 2, "nsamp": 2, "samples": [0.5, 3.5]},
         {"type": "data",   "tx_id": 3, "nsamp": 3, "samples": [1.5, 2.5, 0.1]},
         {"type": "end",    "tx_id": 4, "nbin": 4},
     ],
     {"counters": [2, 1, 1, 1], "error": 0, "halted": 0}),

    # 3. Re-CONFIG mid-activation resets the counters, then DATA, then END.
    ("reconfig_mid_activation",
     [
         {"type": "config", "tx_id": 1, "nbin": 4, "lo": 0.0, "hi": 4.0},
         {"type": "data",   "tx_id": 2, "nsamp": 1, "samples": [0.5]},
         {"type": "config", "tx_id": 3, "nbin": 2, "lo": 0.0, "hi": 2.0},
         {"type": "data",   "tx_id": 4, "nsamp": 2, "samples": [0.5, 1.5]},
         {"type": "end",    "tx_id": 5, "nbin": 2},
     ],
     {"counters": [1, 1], "error": 0, "halted": 0}),

    # 4a. Limits: B=1 (single bin).
    ("limit_b1",
     [
         {"type": "config", "tx_id": 1, "nbin": 1, "lo": 0.0, "hi": 1.0},
         {"type": "data",   "tx_id": 2, "nsamp": 3, "samples": [0.1, 0.5, 0.9]},
         {"type": "end",    "tx_id": 3, "nbin": 1},
     ],
     {"counters": [3], "error": 0, "halted": 0}),

    # 4b. Limits: B=16, nsamp=128.
    ("limit_b16_nsamp128",
     [
         {"type": "config", "tx_id": 1, "nbin": 16, "lo": 0.0, "hi": 16.0},
         {"type": "data",   "tx_id": 2, "nsamp": 128,
          "samples": (np.linspace(0.0, 15.0, 128, dtype=np.float32)).tolist()},
         {"type": "end",    "tx_id": 3, "nbin": 16},
     ],
     {"counters": reference_histogram(
         np.linspace(0.0, 15.0, 128, dtype=np.float32), 16, 0.0, 16.0), "error": 0, "halted": 0}),

    # 4c. Limits: nsamp=1 (a single sample in a DATA command).
    ("limit_nsamp1",
     [
         {"type": "config", "tx_id": 1, "nbin": 4, "lo": 0.0, "hi": 4.0},
         {"type": "data",   "tx_id": 2, "nsamp": 1, "samples": [2.5]},
         {"type": "end",    "tx_id": 3, "nbin": 4},
     ],
     {"counters": [0, 0, 1, 0], "error": 0, "halted": 0}),

    # 5a. DATA before CONFIG.
    ("err_data_before_config",
     [
         {"type": "data",   "tx_id": 7, "nsamp": 2, "samples": [0.5, 1.5]},
     ],
     {"error": int(HistError.DATA_BEFORE_CONFIG), "halted": 1}),

    # 5b. INVALID_CONFIG (B out of range).
    ("err_invalid_config_b0",
     [
         {"type": "config", "tx_id": 8, "nbin": 0, "lo": 0.0, "hi": 4.0},
     ],
     {"error": int(HistError.INVALID_CONFIG), "halted": 1}),

    # 5c. INVALID_CONFIG (lo >= hi).
    ("err_invalid_config_lo_ge_hi",
     [
         {"type": "config", "tx_id": 9, "nbin": 4, "lo": 4.0, "hi": 2.0},
     ],
     {"error": int(HistError.INVALID_CONFIG), "halted": 1}),

    # 5d. ZERO_SAMPLES (DATA with nsamp=0).
    ("err_zero_samples",
     [
         {"type": "config", "tx_id": 10, "nbin": 4, "lo": 0.0, "hi": 4.0},
         {"type": "data",   "tx_id": 11, "nsamp": 0, "samples": []},
     ],
     {"error": int(HistError.ZERO_SAMPLES), "halted": 1}),

    # 5e. SAMPLE_OUT_OF_RANGE (one sample above hi).
    ("err_sample_out_of_range",
     [
         {"type": "config", "tx_id": 12, "nbin": 4, "lo": 0.0, "hi": 4.0},
         {"type": "data",   "tx_id": 13, "nsamp": 2, "samples": [0.5, 4.5]},
     ],
     {"error": int(HistError.SAMPLE_OUT_OF_RANGE), "halted": 1}),

    # 6. NumPy cross-check on an in-range burst.
    ("numpy_crosscheck",
     [
         {"type": "config", "tx_id": 20, "nbin": 8, "lo": 0.0, "hi": 8.0},
         {"type": "data",   "tx_id": 21, "nsamp": 24,
          "samples": (np.random.default_rng(0).uniform(0.0, 8.0, 24)).astype(np.float32).tolist()},
         {"type": "end",    "tx_id": 22, "nbin": 8},
     ],
     {"counters": None, "error": 0, "halted": 0}),  # compared to np.histogram below
]


def verify_all() -> tuple[bool, list[str]]:
    """Run every scenario and return ``(all_passed, report_lines)``."""
    import json
    import tempfile
    from pathlib import Path

    clk_freq = 100e6
    lines: list[str] = []
    all_ok = True

    for name, commands, expect in SCENARIOS:
        sim = Simulation()
        clk = Clock(freq=clk_freq)
        tb = run_scenario(sim, clk, commands)

        ok = True
        notes = []
        if tb.error is None or int(tb.error) != int(expect.get("error", 0)):
            ok = False
            notes.append(f"error={tb.error} expected={expect.get('error')}")
        if expect.get("halted") is not None and int(tb.halted) != int(expect["halted"]):
            ok = False
            notes.append(f"halted={tb.halted} expected={expect['halted']}")

        if expect.get("counters") is not None:
            want = np.asarray(expect["counters"], dtype=np.uint32)
            got = np.asarray(tb.counters, dtype=np.uint32) if tb.counters is not None else None
            if got is None or got.shape != want.shape or not np.array_equal(got, want):
                ok = False
                notes.append(f"counters={None if got is None else got.tolist()} expected={want.tolist()}")

        # NumPy cross-check (scenario 6) — counters must equal np.histogram.
        if name == "numpy_crosscheck":
            cfg = commands[0]
            samples = np.asarray(commands[1]["samples"], dtype=np.float32)
            np_ref, _ = np.histogram(samples, bins=int(cfg["nbin"]),
                                     range=(float(cfg["lo"]), float(cfg["hi"])))
            got = np.asarray(tb.counters, dtype=np.uint32) if tb.counters is not None else None
            if got is None or not np.array_equal(got, np_ref.astype(np.uint32)):
                ok = False
                notes.append(f"counters={None if got is None else got.tolist()} np.histogram={np_ref.tolist()}")

        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}"
        if not ok:
            line += "  (" + "; ".join(notes) + ")"
        lines.append(line)

    # Fresh-activation recovery: each error scenario is followed by a passing
    # activation, per the prompt.  Verify the hand-check once more in a clean
    # sim to confirm the model recovers state correctly between activations.
    sim = Simulation()
    clk = Clock(freq=clk_freq)
    recover = run_scenario(sim, clk, SCENARIOS[0][1])
    rec_ok = (int(recover.error) == 0 and int(recover.halted) == 0
              and np.array_equal(np.asarray(recover.counters, dtype=np.uint32),
                                 np.asarray([2, 1, 1, 2], dtype=np.uint32)))
    all_ok = all_ok and rec_ok
    lines.append(f"[{'PASS' if rec_ok else 'FAIL'}] fresh_activation_recovery")

    return all_ok, lines


def main() -> int:
    import json
    ok, lines = verify_all()
    print("=== Histogram golden-model verification ===")
    for ln in lines:
        print(ln)
    print(f"{'ALL PASSED' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
