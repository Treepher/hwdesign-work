"""Scale-and-offset accelerator — bit-exact Python golden model (in class).

Command-response streaming pattern, following the ``stream_inband`` Waveflow
reference (``poly.py``) for structure but differing where the pattern requires:

* Configuration travels on the **input stream as a CONFIG command**, not on the
  AXI-Lite register map.  The regmap therefore carries only the status
  registers ``halted`` / ``error`` / ``tx_id`` (plus ``ap_start`` / ``ap_done``).
* Commands are ``CONFIG`` / ``DATA`` / ``END``.  A DATA command delivers a burst
  of ``nsamp`` f32 samples and the kernel emits ``nsamp`` f32 outputs
  ``y[i] = a * x[i] + b`` (in order) inline with the command.  ``a`` / ``b``
  persist across DATA commands within one activation; a second CONFIG replaces
  them; nothing persists across activations.

Every step of the scale-and-offset arithmetic is done in 32-bit float
(``(a * x) + b``) so the Python model and the C++ kernel produce identical bits.

One honest limitation (documented in ``results.md``): the SimPy stream read
(``get_pipelined``) reads a fixed ``count`` of words and does not surface whether
the delivered burst was TLAST-framed.  So the golden model cannot distinguish an
early TLAST (TLAST_EARLY) or a missing TLAST (NO_TLAST) from a valid burst — it
reports both as NO_ERROR.  Those two framing errors are therefore exercised on the
C-sim path only, where the kernel's TLAST-tracking ``read_axi4_stream`` reports
the real code, and are excluded from the golden-vs-C bit-for-bit comparison (which
covers the remaining scenarios on both data outputs and status registers).

Run the model end-to-end (all verification scenarios) with::

    python scale_offset.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from waveflow.hw.arrayutils import array, read_array, write_array
from waveflow.hw.clock import Clock
from waveflow.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField
from waveflow.hw.hw_module import HwParam
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
from waveflow.simulation.logger import Logger, NullLogger
from waveflow.simulation.simobj import ProcessGen, SimObj
from waveflow.simulation.simulation import Simulation


INCLUDE_DIR = "include"
WORD_BW_SUPPORTED = [32, 64]

TxIdField  = IntField.specialize(bitwidth=16, signed=False)
NsampField = IntField.specialize(bitwidth=16, signed=False)
U32        = IntField.specialize(bitwidth=32, signed=False, include_dir=INCLUDE_DIR)
Float32    = FloatField.specialize(bitwidth=32, include_dir=INCLUDE_DIR)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScaleError(IntEnum):
    NO_ERROR           = 0
    DATA_BEFORE_CONFIG = 1
    ZERO_SAMPLES       = 2
    TLAST_EARLY        = 3
    NO_TLAST           = 4


ScaleErrorField = EnumField.specialize(enum_type=ScaleError)


class ScaleCmdType(IntEnum):
    CONFIG = 0
    DATA   = 1
    END    = 2


ScaleCmdTypeField = EnumField.specialize(enum_type=ScaleCmdType)


# ---------------------------------------------------------------------------
# Stream schema
# ---------------------------------------------------------------------------

class ScaleCfgPayload(DataList):
    """CONFIG command payload: the scale ``a`` and the offset ``b`` (both f32)."""
    elements = {
        "a": {"schema": Float32, "description": "Scale factor (f32)"},
        "b": {"schema": Float32, "description": "Offset (f32)"},
    }


class ScaleCmdHdr(DataList):
    """Command header.  ``nsamp`` is the payload word count: 2 for CONFIG (the
    ``ScaleCfgPayload``), the sample count for DATA, and 0 for END."""
    elements = {
        "cmd_type": {"schema": ScaleCmdTypeField, "description": "CONFIG / DATA / END"},
        "tx_id":    {"schema": TxIdField,          "description": "Transaction ID"},
        "nsamp":    {"schema": NsampField,         "description": "Payload word count"},
    }


class ScaleRespHdr(DataList):
    """Response header: echo of the transaction ID."""
    elements = {
        "tx_id": {"schema": TxIdField, "description": "Echo of the transaction ID"},
    }


# Persistent per-activation storage, declared via add_state (not the regmap):
# the CONFIG payload travels on the STREAM, so the kernel keeps ``a`` / ``b``
# and an explicit ``configured`` flag as storage that outlives each command.
#
# ``a`` and ``b`` can legitimately be 0.0, so the "not yet configured"
# condition CANNOT be inferred from their value (unlike the take-home, where
# ``nbin == 0`` signals unconfigured).  An explicit ``configured`` flag (u32)
# is required: 0 initially (DATA/END -> DATA_BEFORE_CONFIG), set 1 by a valid
# CONFIG.
ConfiguredArr = DataArray.specialize(U32,     max_shape=(1,), cpp_storage="raw")  # [configured]
ABArr         = DataArray.specialize(Float32, max_shape=(2,), cpp_storage="raw")  # [a, b]


SCHEMA_CLASSES = [
    ScaleErrorField,
    ScaleCmdTypeField,
    ScaleCfgPayload,
    ScaleCmdHdr,
    ScaleRespHdr,
    ConfiguredArr,
    ABArr,
]


# ---------------------------------------------------------------------------
# Accelerator (SimPy golden model)
# ---------------------------------------------------------------------------

@dataclass
class ScaleAccel(HostActivated):
    """SimPy golden model of the scale-and-offset accelerator kernel.

    Control/status is an AXI-Lite ``VitisRegMap`` of the status registers
    only.  The host writes ``ap_start`` to launch ``on_start``, which loops
    reading CONFIG / DATA / END command headers from the input stream.
    """

    cpp_kernel_name: ClassVar[str | None] = "scale_offset"
    cpp_namespace:   ClassVar[str | None] = "scale_offset_impl"

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
            "error":  RegField(ScaleErrorField, RegAccess.R, description="Last error code"),
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
        # keeps it as a zero-initialised ``static`` that outlives each command:
        #   configured = "a CONFIG has been seen" (u32[1]; 0 until a valid CONFIG)
        #   ab         = [a, b]                   (f32[2])
        # ``a`` / ``b`` may be 0.0, so the unconfigured condition is the explicit
        # ``configured`` flag, not the value of ``a`` / ``b``.
        self.configured = HwState(ConfiguredArr())
        self.ab         = HwState(ABArr())
        self.add_state(self.configured)
        self.add_state(self.ab)

    @sim_only
    def _inc_job(self) -> None:
        self._job += 1

    # -- per-command handlers (synthesizable hooks) ---------------------------

    @synthesizable
    def configure(self, cmd_hdr: ScaleCmdHdr,
                  s_in: StreamIFSlave, m_out: StreamIFMaster,
                  configured: HwState, ab: HwState
                  ) -> ProcessGen[ScaleError]:
        """CONFIG: ack, read the payload (a, b), store it, set configured.

        The RespHdr is emitted before the payload is read so the stream stays in
        lockstep (one RespHdr per command header).
        """
        resp = ScaleRespHdr()
        resp.tx_id = cmd_hdr.tx_id
        self.logger.log(event='config_ack', job=self._job)
        yield from m_out.write(resp)

        cfg = yield from s_in.get_schema(ScaleCfgPayload)
        a = np.float32(cfg.a)
        b = np.float32(cfg.b)

        # Store config; mark the activation as configured.
        ab.val[0]       = a
        ab.val[1]       = b
        configured.val[0] = np.uint32(1)
        return ScaleError.NO_ERROR

    @synthesizable
    def transform(self, cmd_hdr: ScaleCmdHdr,
                  s_in: StreamIFSlave, m_out: StreamIFMaster,
                  configured: HwState, ab: HwState
                  ) -> ProcessGen[ScaleError]:
        """DATA: ack, read ``nsamp`` samples, emit ``y[i] = a*x[i] + b`` in order.

        The output is computed entirely in 32-bit float so the Python model and
        the C++ kernel produce identical bits.  A burst shorter than ``nsamp``
        (early TLAST) is TLAST_EARLY; a missing TLAST (NO_TLAST) is not
        distinguishable in the SimPy stream model and is exercised in C-sim only.
        """
        resp = ScaleRespHdr()
        resp.tx_id = cmd_hdr.tx_id
        self.logger.log(event='data_ack', job=self._job)
        yield from m_out.write(resp)

        if configured.val[0] == 0:
            return ScaleError.DATA_BEFORE_CONFIG
        nsamp = int(cmd_hdr.nsamp)
        if nsamp == 0:
            return ScaleError.ZERO_SAMPLES

        samp_in, tstart = yield from s_in.get_pipelined(Float32, count=nsamp)
        samp_in = np.asarray(samp_in, dtype=np.float32)

        # Burst must be complete (a shorter burst means an early TLAST).
        if len(samp_in) != nsamp:
            return ScaleError.TLAST_EARLY

        # Scale-and-offset, 32-bit float arithmetic throughout.
        a = np.float32(ab.val[0])
        b = np.float32(ab.val[1])
        y = (a * samp_in) + b

        # Model the per-sample processing time (II=1) for timing fidelity.
        proc_time = nsamp / self.unroll_factor * self.proc_ii * self.clk.period
        t_out_start = tstart + self.proc_latency * self.clk.period
        proc_time = max(0.0, proc_time + (t_out_start - self.env.now))
        if proc_time > 0:
            yield self.timeout(proc_time)

        yield from m_out.write_pipelined(array(Float32, y), t_out_start)
        self.logger.log(event='data_out_write_end', job=self._job)
        return ScaleError.NO_ERROR

    @synthesizable
    def end(self, cmd_hdr: ScaleCmdHdr, m_out: StreamIFMaster,
            configured: HwState
            ) -> ProcessGen[ScaleError]:
        """END: ack, then (if configured) return.  Terminal."""
        resp = ScaleRespHdr()
        resp.tx_id = cmd_hdr.tx_id
        self.logger.log(event='end_ack', job=self._job)
        yield from m_out.write(resp)

        if configured.val[0] == 0:
            return ScaleError.DATA_BEFORE_CONFIG
        return ScaleError.NO_ERROR

    # -- kernel body ----------------------------------------------------------

    def on_start(self) -> ProcessGen[None]:
        """Kernel body — invoked by ``VitisRegMapMMIFSlave`` on ap_start."""
        while True:
            self.logger.log(event='proc_begin', job=self._job)
            cmd_hdr = yield from self.s_in.get_schema(ScaleCmdHdr)
            if cmd_hdr.cmd_type == ScaleCmdType.CONFIG:
                err = yield from self.configure(cmd_hdr, self.s_in, self.m_out,
                                                self.configured, self.ab)
                self._inc_job()
                if err != ScaleError.NO_ERROR:
                    self.regmap.set("error",  err)
                    self.regmap.set("tx_id",  cmd_hdr.tx_id)
                    self.regmap.set("halted", 1)
                    return
            elif cmd_hdr.cmd_type == ScaleCmdType.DATA:
                err = yield from self.transform(cmd_hdr, self.s_in, self.m_out,
                                                self.configured, self.ab)
                self._inc_job()
                if err != ScaleError.NO_ERROR:
                    self.regmap.set("error",  err)
                    self.regmap.set("tx_id",  cmd_hdr.tx_id)
                    self.regmap.set("halted", 1)
                    return
            else:  # END — terminal
                err = yield from self.end(cmd_hdr, self.m_out, self.configured)
                self._inc_job()
                if err != ScaleError.NO_ERROR:
                    self.regmap.set("error",  err)
                    self.regmap.set("tx_id",  cmd_hdr.tx_id)
                    self.regmap.set("halted", 1)
                return


# ---------------------------------------------------------------------------
# SimPy testbench
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class ScaleTB(SimObj):
    """Drives a command sequence (CONFIG / DATA... / END) and captures the
    responses, the inline DATA outputs, and the final ``halted`` / ``error`` /
    ``tx_id`` status.

    ``commands`` is a list of dicts:
        {"type": "config", "tx_id": .., "a": .., "b": ..}
        {"type": "data",   "tx_id": .., "nsamp": .., "samples": np.ndarray}
        {"type": "end",    "tx_id": ..}

    For a TLAST_EARLY stimulus, ``len(samples) < nsamp`` (the burst is shorter
    than the command declares).
    """
    commands:  list[dict]
    word_bw:   int = 32
    base_addr: int = 0x0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.m_in   = StreamIFMaster(name=f'{self.name}_m_in',   sim=self.sim, bitwidth=self.word_bw)
        self.s_out  = StreamIFSlave( name=f'{self.name}_s_out',  sim=self.sim, bitwidth=self.word_bw)
        self.m_lite = MMIFMaster(    name=f'{self.name}_m_lite', sim=self.sim, bitwidth=32)
        self.resp_hdrs:    list[ScaleRespHdr] = []
        self.data_outputs: list[npt.NDArray[np.float32]] = []
        self.halted:       int | None = None
        self.error:        ScaleError | None = None
        self.tx_id_status: int | None = None
        self._regmap_ref:  VitisRegMap | None = None

    def _regmap(self) -> VitisRegMap:
        if self._regmap_ref is None:
            raise RuntimeError("ScaleTB._regmap_ref is unset; call connect() before run_sim().")
        return self._regmap_ref

    def _read_resp(self) -> ProcessGen[ScaleRespHdr]:
        words = yield from self.s_out.get()
        return ScaleRespHdr().deserialize(words, word_bw=self.word_bw)

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
                hdr = ScaleCmdHdr()
                hdr.cmd_type = ScaleCmdType.CONFIG
                hdr.tx_id    = int(cmd["tx_id"])
                hdr.nsamp    = 2                      # ScaleCfgPayload = 2 words (a, b)
                yield from self.m_in.write(hdr.serialize(word_bw=bw))
                payload = ScaleCfgPayload()
                payload.a = float(np.float32(cmd["a"]))
                payload.b = float(np.float32(cmd["b"]))
                yield from self.m_in.write(payload.serialize(word_bw=bw))
                resp = yield from self._read_resp()
                self.resp_hdrs.append(resp)

            elif cmd["type"] == "data":
                nsamp = int(cmd["nsamp"])
                hdr = ScaleCmdHdr()
                hdr.cmd_type = ScaleCmdType.DATA
                hdr.tx_id    = int(cmd["tx_id"])
                hdr.nsamp    = nsamp
                yield from self.m_in.write(hdr.serialize(word_bw=bw))
                samples = np.asarray(cmd["samples"], dtype=np.float32)[:nsamp]
                yield from self.m_in.write(write_array(samples, elem_type=Float32, word_bw=bw))
                resp = yield from self._read_resp()
                self.resp_hdrs.append(resp)
                # A valid DATA burst emits nsamp f32 outputs inline; an error
                # (DATA_BEFORE_CONFIG / ZERO_SAMPLES / TLAST_EARLY) emits none.
                if not (yield from self._halted(rm)):
                    outwords = yield from self.s_out.get()
                    outputs = np.asarray(
                        read_array(outwords, elem_type=Float32, word_bw=bw, shape=nsamp),
                        dtype=np.float32,
                    )
                    self.data_outputs.append(outputs)

            else:  # end
                hdr = ScaleCmdHdr()
                hdr.cmd_type = ScaleCmdType.END
                hdr.tx_id    = int(cmd["tx_id"])
                hdr.nsamp    = 0
                yield from self.m_in.write(hdr.serialize(word_bw=bw))
                resp = yield from self._read_resp()
                self.resp_hdrs.append(resp)

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


def connect(sim: Simulation, tb: ScaleTB, accel: ScaleAccel, clk: Clock) -> None:
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


# ---------------------------------------------------------------------------
# Verification scenarios
# ---------------------------------------------------------------------------

def run_scenario(sim, clk, commands, logger=None) -> ScaleTB:
    logger = logger or NullLogger()
    accel = ScaleAccel(name="scale_offset_accel", sim=sim, clk=clk, logger=logger,
                       in_bw=32, out_bw=32)
    tb = ScaleTB(name="scale_offset_tb", sim=sim, commands=commands, word_bw=32)
    connect(sim, tb, accel, clk)
    sim.run_sim()
    return tb


def reference_scale_offset(samples, a, b) -> np.ndarray:
    """The prompt's transform rule, computed independently (hand-checkable)."""
    a = np.float32(a); b = np.float32(b)
    x = np.asarray(samples, dtype=np.float32)
    return (a * x) + b


SCENARIOS: list[tuple[str, list[dict], dict]] = [
    # 1. Basic hand-check: a=2.0, b=1.0, [0.5, -1.25, 3.0] -> [2.0, -1.5, 7.0].
    ("basic_handcheck",
     [
         {"type": "config", "tx_id": 1, "a": 2.0, "b": 1.0},
         {"type": "data",   "tx_id": 2, "nsamp": 3, "samples": [0.5, -1.25, 3.0]},
         {"type": "end",    "tx_id": 3},
     ],
     {"outputs": [[2.0, -1.5, 7.0]], "error": 0, "halted": 0}),

    # 2. Two DATA commands in one activation — both use the same a/b.
    ("two_data",
     [
         {"type": "config", "tx_id": 1, "a": 2.0, "b": 1.0},
         {"type": "data",   "tx_id": 2, "nsamp": 2, "samples": [0.5, -1.25]},
         {"type": "data",   "tx_id": 3, "nsamp": 2, "samples": [3.0, 0.25]},
         {"type": "end",    "tx_id": 4},
     ],
     {"outputs": [[2.0, -1.5], [7.0, 1.5]], "error": 0, "halted": 0}),

    # 3. A re-CONFIG mid-activation — the new a/b apply to the DATA after it
    #    (negative a and non-zero b on the second CONFIG).
    ("reconfig",
     [
         {"type": "config", "tx_id": 1, "a": 2.0,  "b": 1.0},
         {"type": "data",   "tx_id": 2, "nsamp": 1, "samples": [1.0]},
         {"type": "config", "tx_id": 3, "a": -1.0, "b": 0.5},
         {"type": "data",   "tx_id": 4, "nsamp": 2, "samples": [2.0, -3.0]},
         {"type": "end",    "tx_id": 5},
     ],
     {"outputs": [[3.0], [-1.5, 3.5]], "error": 0, "halted": 0}),

    # 4a. DATA_BEFORE_CONFIG: a DATA command before any CONFIG.
    ("err_data_before_config",
     [
         {"type": "data",   "tx_id": 7, "nsamp": 2, "samples": [0.5, 1.5]},
     ],
     {"error": int(ScaleError.DATA_BEFORE_CONFIG), "halted": 1}),

    # 4b. ZERO_SAMPLES: a DATA command with nsamp = 0.
    ("err_zero_samples",
     [
         {"type": "config", "tx_id": 9,  "a": 1.0, "b": 0.0},
         {"type": "data",   "tx_id": 10, "nsamp": 0, "samples": []},
     ],
     {"error": int(ScaleError.ZERO_SAMPLES), "halted": 1}),

    # Fresh passing activation (the recovery that follows each error scenario):
    # a clean CONFIG -> DATA -> END with a non-trivial a/b.
    ("fresh_pass",
     [
         {"type": "config", "tx_id": 20, "a": -2.0, "b": 3.0},
         {"type": "data",   "tx_id": 21, "nsamp": 2, "samples": [1.5, -0.5]},
         {"type": "end",    "tx_id": 22},
     ],
     {"outputs": [[0.0, 4.0]], "error": 0, "halted": 0}),
]


def verify_all() -> tuple[bool, list[str]]:
    """Run every scenario and return ``(all_passed, report_lines)``."""
    clk_freq = 100e6
    lines: list[str] = []
    all_ok = True

    for name, commands, expect in SCENARIOS:
        sim = Simulation()
        clk = Clock(freq=clk_freq)
        tb = run_scenario(sim, clk, commands)

        ok = True
        notes = []
        if int(tb.error) != int(expect.get("error", 0)):
            ok = False
            notes.append(f"error={tb.error} expected={expect.get('error')}")
        if expect.get("halted") is not None and int(tb.halted) != int(expect["halted"]):
            ok = False
            notes.append(f"halted={tb.halted} expected={expect['halted']}")

        # Inline DATA outputs (valid activations only).
        want_out = expect.get("outputs")
        if want_out is not None:
            got = (np.asarray(o, dtype=np.float32) for o in tb.data_outputs) \
                  if tb.data_outputs else []
            got = list(got)
            want = [np.asarray(w, dtype=np.float32) for w in want_out]
            if len(got) != len(want) or any(
                    g.shape != w.shape or not np.array_equal(g, w)
                    for g, w in zip(got, want)):
                ok = False
                got_list = [None if g is None else g.tolist() for g in got]
                want_list = [w.tolist() for w in want]
                notes.append(f"outputs={got_list} expected={want_list}")

        # Independent reference check (hand-checkable arithmetic).
        if name == "basic_handcheck":
            ref = reference_scale_offset([0.5, -1.25, 3.0], 2.0, 1.0)
            if not np.array_equal(ref, np.asarray([2.0, -1.5, 7.0], dtype=np.float32)):
                ok = False
                notes.append(f"reference mismatch: {ref.tolist()}")

        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}"
        if not ok:
            line += "  (" + "; ".join(notes) + ")"
        lines.append(line)

    return all_ok, lines


def main() -> int:
    ok, lines = verify_all()
    print("=== Scale-and-offset golden-model verification ===")
    for ln in lines:
        print(ln)
    print(f"{'ALL PASSED' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
