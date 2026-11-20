# Histogram accelerator — results

All files cited below are committed in this folder (`cmdresp/takehome/`). Evidence
tree: `results/sim/` (Python golden model, one directory per scenario), `data/`
(C-sim, one directory per scenario), and `results/vitis/cosim/` (RTL co-sim, one
directory per scenario); the cross-simulator comparison records `results/verify_csim.json`
and `results/verify_cosim.json`; the C-sim log `results/csim.log` and the co-sim log
`results/vitis/cosim.log` (with the co-sim report `results/vitis/hist_cosim.rpt`); the
synthesis report `results/vitis/csynth.rpt` and `results/loop_df.csv`; and the co-sim
throughput runs `results/vitis/tp_1/`, `results/vitis/tp_16/`, and
`results/vitis/throughput.json`. The kernel is
`gen/hist.cpp`; the CONFIG / DATA / END bodies are the hand-written hooks
`hist_configure_impl.tpp`, `hist_accumulate_impl.tpp`, and `hist_emit_impl.tpp`,
included by `gen/hist.hpp`. The bit-exact golden model is `hist.py`; the functional
and throughput testbenches are `gen/hist_tb.cpp` and `gen/hist_tp_tb.cpp`; the build
driver is `hist_build.py`.

## 1. Interface

**Transport.** All commands and responses travel on two 32-bit AXI4-Stream ports,
`s_in` (commands + payload) and `m_out` (responses + counters). Every command header
and every response header is a TLAST. The AXI-Lite port `s_axi_control` carries only
`ap_start` / `ap_done` plus the status registers (`results/vitis/csynth.rpt`, HW
Interfaces).

**Command header** `HistCmdHdr` = `(cmd_type: 2 b, tx_id: 16 b, nsamp: 16 b)` = 34
bits, packed little-endian into two 32-bit words by `include/hist_cmd_hdr.h`: word0
`bits[1:0]=cmd_type`, `bits[17:2]=tx_id`, `bits[31:18]=nsamp[13:0]`; word1
`bits[15:0]=nsamp[15:14]` (rest zero).

**CONFIG (cmd_type 0).** Header `(0, tx_id, 3)` followed by payload `CfgPayload` =
`(nbin: 16 b, lo: 32 b, hi: 32 b)` = 80 bits, packed by `include/cfg_payload.h` into
three 32-bit words (word0 `bits[15:0]=nbin`, `bits[31:16]=lo[15:0]`; word1
`bits[15:0]=lo[31:16]`, `bits[31:16]=hi[15:0]`; word2 `bits[15:0]=hi[31:16]`).

**DATA (cmd_type 1).** Header `(1, tx_id, nsamp)` followed by `nsamp` × f32 samples,
one 32-bit word each, the burst TLAST-framed.

**END (cmd_type 2).** Header `(2, tx_id, 0)`, no payload.

**Response.** Every command is acknowledged with one `HistRespHdr` (echo of `tx_id`,
u16 → one 32-bit word, `include/hist_resp_hdr.h`). END additionally emits the `nbin`
counters in bin order, each a u32 word, TLAST on the final word. For the basic
activation the counters are byte-identical in all three simulators —
`results/sim/basic_handcheck/counters.bin`, `data/basic_handcheck/counters.bin`, and
`results/vitis/cosim/basic_handcheck/counters.bin` —
= `02000000 01000000 01000000 02000000` = `[2,1,1,2]`.

**Status registers** (`ap_uint<1>& halted`, `ap_uint<8>& error`, `ap_uint<16>& tx_id`
in `gen/hist.hpp:18-20`), dumped to `regmap_status.json` in each scenario directory.
Basic activation: `results/sim/basic_handcheck/regmap_status.json` =
`{halted: 0, error: 0, tx_id: 0}`.

**Error codes** (`include/hist_error.h`): 0 NO_ERROR, 1 DATA_BEFORE_CONFIG,
2 INVALID_CONFIG, 3 ZERO_SAMPLES, 4 SAMPLE_OUT_OF_RANGE, 5 TLAST_EARLY, 6 NO_TLAST.

## 2. Design

**Where state lives.** Configuration and counters are **local to the kernel call** —
`nbin[1]` (u32), `bounds[2]` (f32 lo/hi), `counters[16]` (u32) declared in `hist()`
(`gen/hist.cpp:26-34`) and threaded through the three command hooks. Because they are
local (not static), each `ap_start` re-enters `hist()` with `nbin` reset to 0 ("not yet
configured") and the counters re-zeroed, so nothing persists across activations —
exactly the prompt's "each `ap_start` runs a fresh kernel" and the golden model's
per-activation state. The status registers are cleared at the start of every activation
(`gen/hist.cpp:43-45`) so an activation following an error does not read the previous
error code; this is what keeps C-sim and co-sim bit-identical on `halted`/`error`/
`tx_id`.

**Loop structure.**
- `configure` (`hist_configure_impl.tpp:21-51`): emit ack (25-28); read the 3-word
  payload (30-35); validate `n∉[1,16]` or `lo>=hi` → INVALID_CONFIG (37-40); store the
  config and zero the 16 counters (42-49, unrolled). No data loop.
- `accumulate` / DATA (`hist_accumulate_impl.tpp:28-128`): emit ack (32-35); gate on
  `n==0` → DATA_BEFORE_CONFIG and `nsamp==0` → ZERO_SAMPLES (37-45); then the
  per-sample loop `for (i = 0; i < nsamp && !read_done; i += pf)` (75) reads `pf` lanes
  per AXI word, the inner lane loop `#pragma HLS UNROLL` (82-98). Each in-range sample
  does `local_counters[idx] = local_counters[idx] + 1` (95) with
  `idx = truncate((x-lo)*scale)`, `scale = B/(hi-lo)`, all in f32, clamped to
  `[0, B-1]`. After the loop it validates (108-120) and, only on the success path,
  commits the local buffer into the persistent `counters` (122-126).
- `emit` / END (`hist_emit_impl.tpp:17-38`): emit ack (20-23); gate on `n==0` →
  DATA_BEFORE_CONFIG (25-29); then `for (i = 0; i < n; ++i)` with
  `#pragma HLS PIPELINE II=1` (32-36) streaming the counters, TLAST on the last.

**How II = 1 is reached.** The per-sample binning loop's only loop-carried dependency
is the counter read-modify-write. Accumulation is therefore done into a local buffer
`local_counters[16]` that is `#pragma HLS ARRAY_PARTITION ... complete dim=1`
(`hist_accumulate_impl.tpp:63-64`) — a fully partitioned register file, not a BRAM — so
the read+write closes in one cycle and the loop runs at II = 1. A 1R1W BRAM would make
that read-after-write a 2-cycle access and pin the loop at II = 2. The local buffer is
committed into the persistent `counters` only on the success path
(`hist_accumulate_impl.tpp:122-126`), so the per-sample loop never touches the
persistent array. Result: the binning loop `VITIS_LOOP_75_2`
(`hist_accumulate_impl.tpp:75`) is Interval = 1, Pipelined = yes, and the END
counter-emission loop (`hist_emit_impl.tpp:32`) is Interval = 1, Pipelined = yes
(`results/vitis/csynth.rpt`; `results/loop_df.csv`).

## 3. Targets

| Target | Required | Measured | Result | File |
|--------|----------|----------|--------|------|
| Binning-loop initiation interval | II = 1 | Interval = 1 (`VITIS_LOOP_75_2`, Pipelined yes) | PASS | `results/vitis/csynth.rpt` |
| END counter-emission interval | II = 1 | Interval = 1 (`hist_emit_impl.tpp:32`, Pipelined yes) | PASS | `results/vitis/csynth.rpt` |
| Throughput (per-sample) | 1 sample / clock (II = 1) at 100 MHz | 1.0 sample / clock (co-sim delta method, (16-1)/(134-119)) | PASS | `results/vitis/throughput.json` (runs `results/vitis/tp_1/hist_cosim.rpt`, `results/vitis/tp_16/hist_cosim.rpt`) |
| Frequency | 100 MHz | Estimated Fmax = 137.82 MHz | PASS | `results/vitis/solution1.log` |
| Bit-exactness (golden / C-sim / co-sim) | identical bits | counters `[2,1,1,2]` and status `{0,0,0}` identical in all three on the basic activation, and identical on all 21 scenarios | PASS | `results/sim/basic_handcheck/`, `data/basic_handcheck/`, `results/vitis/cosim/basic_handcheck/` (+ `results/verify_csim.json`, `results/verify_cosim.json`) |
| Latency | no target | 25 / 105 / 246 cycles (min / avg / max) | (no target) | `results/vitis/hist_cosim.rpt` |
| Resource usage | no target | 5 DSP, 4363 FF, 5700 LUT, 0 BRAM, 0 URAM | (no target) | `results/vitis/csynth.rpt` |

**Throughput is measured from co-simulation, not from the synthesis estimate.** The
delta method takes two RTL co-sim runs of the same throughput testbench
(`gen/hist_tp_tb.cpp`) — 1 sample in 119 cycles
(`results/vitis/tp_1/hist_cosim.rpt`) and 16 samples in 134 cycles
(`results/vitis/tp_16/hist_cosim.rpt`) — and computes the marginal per-sample rate
`(16-1)/(134-119) = 15/15 = 1.0` sample per clock. That is the steady-state per-sample
rate, i.e. the loop initiation interval, and it equals the II = 1 target (one sample
per clock at 100 MHz). `results/vitis/throughput.json` records the two runs, the delta,
and the pass.

## 4. Verification

**Independent check (golden model).** (a) The hand-computed example — `B=4, lo=0, hi=4`,
samples `[0.2, 1.8, 2.1, 3.9, 0.0, 3.2]` → `[2,1,1,2]` — plus two further small cases
(`accumulate_two_data`, `reconfig_mid_activation`) pass (`results/sim/py_model.log`).
(b) NumPy reference — `numpy_crosscheck` passes (`results/sim/py_model.log`). The golden
model (`hist.py`) is a separate Python implementation of the same f32 arithmetic; its
outputs match the C/RTL path bit for bit on every scenario it can run (below).

**Coverage.** All 21 scenarios are verified in **C-sim** (`data/<scenario>/`) and in
**co-sim** (`results/vitis/cosim/<scenario>/`), with `results/verify_csim.json` and
`results/verify_cosim.json` both recording `all_passed: true`. The golden model covers
the same 21 except the two TLAST *error* scenarios (`err_tlast_early`, `err_no_tlast`):
the golden model is event-driven and cannot surface a timing fault on the sample burst,
so it verifies the *recovery* after a TLAST fault but not the fault itself. Those two
are therefore exercised on the C/RTL path (C-sim and co-sim), where they are genuinely
detected — `err_tlast_early` → `error=5, tx_id=31` and `err_no_tlast` → `error=6,
tx_id=41`, in both `data/` and `results/vitis/cosim/`.

| # | Scenario | Golden | C-sim | Co-sim |
|---|----------|--------|-------|--------|
| 1 | Basic hand-check → `[2,1,1,2]` | PASS (`results/sim/basic_handcheck/`) | PASS (`data/basic_handcheck/`, `results/verify_csim.json`) | PASS (`results/vitis/cosim/basic_handcheck/`, `results/verify_cosim.json`) |
| 2 | Two DATA accumulate, then END | PASS (`results/sim/accumulate_two_data/`) | PASS (`data/accumulate_two_data/`) | PASS (`results/vitis/cosim/accumulate_two_data/`) |
| 3 | Re-CONFIG mid-activation, then DATA, then END | PASS (`results/sim/reconfig_mid_activation/`) | PASS (`data/reconfig_mid_activation/`) | PASS (`results/vitis/cosim/reconfig_mid_activation/`) |
| 4 | Limits `B=1`, `B=16`, `nsamp=1`, `nsamp=128` | PASS (`results/sim/limit_b1/`, `limit_b16_nsamp128/`, `limit_nsamp1/`) | PASS (`data/limit_b1/`, `limit_b16_nsamp128/`, `limit_nsamp1/`) | PASS (`results/vitis/cosim/limit_b1/`, `limit_b16_nsamp128/`, `limit_nsamp1/`) |
| 5 | Each error once — DATA_BEFORE_CONFIG, INVALID_CONFIG (×2), ZERO_SAMPLES, SAMPLE_OUT_OF_RANGE, TLAST_EARLY, NO_TLAST — each followed by a fresh passing activation | PASS for the first five and for every `recover_*` (`results/sim/py_model.log`); TLAST fault not surfaceable in the golden model | PASS for all six (`data/err_*/`, `results/verify_csim.json`) | PASS for all six (`results/vitis/cosim/err_*/`, `results/verify_cosim.json`) |
| 6 | NumPy cross-check | PASS (`results/sim/numpy_crosscheck/`) | PASS (`data/numpy_crosscheck/`) | PASS (`results/vitis/cosim/numpy_crosscheck/`) |

**TLAST detail (the two scenarios previously unexercised).** `err_tlast_early` drives a
sample burst that raises TLAST before `nsamp` samples: `data/err_tlast_early/` (C-sim)
and `results/vitis/cosim/err_tlast_early/` (co-sim) both report
`{halted: 1, error: 5, tx_id: 31}`. `err_no_tlast` drives a burst that runs out without
a TLAST: `data/err_no_tlast/` and `results/vitis/cosim/err_no_tlast/` both report
`{halted: 1, error: 6, tx_id: 41}`. The following `recover_tlast_early` and
`recover_no_tlast` activations pass in all three simulators (`[2,1,1,2]`, `{0,0,0}`),
confirming fresh-stream recovery after a TLAST fault.

Bit-exact agreement across all three simulators on the basic activation:
`results/sim/basic_handcheck/counters.bin`, `data/basic_handcheck/counters.bin`, and
`results/vitis/cosim/basic_handcheck/counters.bin` are byte-identical
(`02000000 01000000 01000000 02000000`), as are their `regmap_status.json` files
(`{halted:0, error:0, tx_id:0}`).

## 5. Assumptions

- **Word packing / bit layout.** The prompt fixes field order (cmd_type, tx_id, nsamp)
  and payload order (B, lo, hi) but not exact bit positions or word boundaries. I used
  the generated little-endian packing as-is: 34-bit header into two 32-bit words, 80-bit
  config payload into three 32-bit words (`include/hist_cmd_hdr.h`,
  `include/cfg_payload.h`).
- **Stream width.** The prompt states samples/lo/hi are 32-bit; I ran both `s_in` and
  `m_out` at 32-bit words, so `pf` (floats per word) = 1.
- **Acknowledgment timing.** Settled by prompt v2: each command emits its `RespHdr`
  (ack) immediately after its header is accepted, before its payload is processed, and a
  later payload error emits no further response — one response header per command
  header, so the output stream stays in lockstep with the input
  (`hist_configure_impl.tpp:25-28`, `hist_accumulate_impl.tpp:32-35`,
  `hist_emit_impl.tpp:20-23`).
- **DATA error priority.** Settled by prompt v2: for a DATA command the reported code is
  the first match in the order SAMPLE_OUT_OF_RANGE → TLAST_EARLY → NO_TLAST, with
  ZERO_SAMPLES and DATA_BEFORE_CONFIG checked before the payload is read
  (`hist_accumulate_impl.tpp:108-120`).
- **Fresh state per activation.** Each `ap_start` re-runs `hist()` with the
  config/counter state local to the call (`gen/hist.cpp:26-34`) and the status
  registers cleared (`gen/hist.cpp:43-45`); nothing persists across activations.
- **Target part / clock.** The prompt sets a 100 MHz clock but no part; I ran
  `xc7z020clg484-1` (Zynq-7020 die, `-1` grade) at a 10 ns (100 MHz) clock —
  `run.tcl`, `results/vitis/csynth.rpt` ("Target device: xc7z020-clg484-1").

## 6. Open Items

- **TLAST timing faults are C/RTL-only.** The golden model is event-driven and cannot
  surface TLAST_EARLY / NO_TLAST, so those two *error* scenarios are verified in C-sim
  and co-sim only (both report the correct error code and tx_id); the *recovery* after
  a TLAST fault is verified in all three. This is a property of the golden model, not a
  gap in the kernel — the C/RTL path detects both faults correctly.
- **Throughput is a two-point measurement.** The per-sample throughput is the marginal
  rate over N=1 and N=16 (`results/vitis/throughput.json`). It equals the II = 1 target
  and is independently consistent with the binning loop's Interval = 1 in
  `results/vitis/csynth.rpt`.
- **In-class part.** Submission parts 1–2 (in-class prompt + results) are a separate
  deliverable; `cmdresp/inclass/` is not populated in this folder.

## 7. Run History

| Prompt version | What was built / changed | What passed |
|----------------|--------------------------|-------------|
| 1 (first run) | Built golden model, kernel hooks, generated TB. First C-sim + co-sim of the basic activation. Binning loop measured II = 2 (counters in BRAM) — a miss vs the II = 1 target. | C-sim and co-sim of the basic activation; all golden scenarios. II = 1 target: FAIL at this run. |
| 1 (after fix) | Moved accumulation into a fully partitioned `local_counters` register file (committed to `counters` only on success). Re-ran C-sim, co-sim, and synthesis with `--force-step`. | All golden scenarios (`results/sim/py_model.log`); C-sim + co-sim of the basic activation bit-exact; binning loop Interval = 1 (`results/vitis/csynth.rpt`, `results/loop_df.csv`); Fmax 137.82 MHz (`results/vitis/solution1.log`). |
| 2 (current) | Made the kernel state local to `hist()` so each activation is genuinely fresh (`gen/hist.cpp:26-34`) and cleared the status registers on each `ap_start` (`gen/hist.cpp:43-45`); rebuilt the testbench to drive all 21 scenarios in C-sim and co-sim — every error code including TLAST_EARLY and NO_TLAST, each followed by a fresh passing activation; added a throughput testbench (`gen/hist_tp_tb.cpp`) that measures per-sample throughput from two co-sim runs. Prompt v2 added the acknowledgment-timing and DATA-error-priority clauses. | C-sim 21/21 and co-sim 21/21 (`results/verify_csim.json`, `results/verify_cosim.json`); TLAST_EARLY (`error=5`) and NO_TLAST (`error=6`) detected in C-sim and co-sim with passing recovery; per-sample throughput 1.0 sample/clock = II = 1 target (`results/vitis/throughput.json`); binning and emit loops Interval = 1 (`results/vitis/csynth.rpt`); Fmax 137.82 MHz (`results/vitis/solution1.log`). |
