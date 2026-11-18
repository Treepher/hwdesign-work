# Histogram accelerator — results

All files cited below are committed in this folder (`cmdresp/takehome/`). Evidence
tree: `results/sim/` (Python golden model), `data/` (C-sim), `results/vitis/` (co-sim
log/report, synthesis report, and the co-sim data outputs). `gen/hist_tb.cpp` is the
generated testbench; the kernel bodies are the hand-written `hist_*_impl.tpp` hooks
included by `gen/hist.hpp`.

## 1. Interface

**Transport.** All commands and responses travel on two 32-bit AXI4-Stream ports,
`s_in` (commands + payload) and `m_out` (responses + counters). Every command header
and every response header is a TLAST. The AXI-Lite port `s_axi_control` carries only
`ap_start` / `ap_done` plus the status registers (`results/vitis/csynth.rpt`,
HW Interfaces).

**Command header** `HistCmdHdr` = `(cmd_type: 2 b, tx_id: 16 b, nsamp: 16 b)` = 34
bits, packed little-endian into two 32-bit words by `include/hist_cmd_hdr.h`:
word0 `bits[1:0]=cmd_type`, `bits[17:2]=tx_id`, `bits[31:18]=nsamp[13:0]`;
word1 `bits[15:0]=nsamp[15:14]` (rest zero). Stimulus in `data/config_cmd_hdr.bin`
(`04000000 03000000` → cmd_type 0, tx_id 1, nsamp 3) confirms the layout.

**CONFIG (cmd_type 0).** Header `(0, tx_id, 3)` followed by payload `CfgPayload` =
`(nbin: 16 b, lo: 32 b, hi: 32 b)` = 80 bits, packed by `include/cfg_payload.h` into
three 32-bit words: word0 `bits[15:0]=nbin`, `bits[31:16]=lo[15:0]`;
word1 `bits[15:0]=lo[31:16]`, `bits[31:16]=hi[15:0]`; word2 `bits[15:0]=hi[31:16]`.
`data/config_payload.bin` = `04000000 00000000 00008040` → nbin 4, lo 0.0, hi 4.0.

**DATA (cmd_type 1).** Header `(1, tx_id, nsamp)` followed by `nsamp` × f32 samples,
one 32-bit word each. `data/data_cmd_hdr.bin` = `09000000 06000000` → cmd_type 1,
tx_id 2, nsamp 6; `data/samp_in_data.bin` holds the six f32 samples.

**END (cmd_type 2).** Header `(2, tx_id, 0)`, no payload.
`data/end_cmd_hdr.bin` = `0e000000 00000000` → cmd_type 2, tx_id 3.

**Response.** Every command is acknowledged with one `HistRespHdr` (echo of `tx_id`,
u16 → one 32-bit word, `include/hist_resp_hdr.h`). END additionally emits the `nbin`
counters in bin order, each a u32 word, TLAST on the final word. For the basic
activation the outputs are `results/sim/resp_cfg.bin` = `01000000` (tx_id 1),
`resp_data.bin` = `02000000` (tx_id 2), `resp_end.bin` = `03000000` (tx_id 3), and
`results/sim/counters.bin` = `02000000 01000000 01000000 02000000` = [2,1,1,2].

**Status registers** (`ap_uint<1>& halted`, `ap_uint<8>& error`, `ap_uint<16>& tx_id`
in `gen/hist.hpp:18-20`), dumped to `regmap_status.json`: `halted` (1 bit), `error`
(code), `tx_id` (offending command). Basic activation:
`results/sim/regmap_status.json` = `{halted: 0, error: 0, tx_id: 0}`.

**Error codes** (`include/hist_error.h`): 0 NO_ERROR, 1 DATA_BEFORE_CONFIG,
2 INVALID_CONFIG, 3 ZERO_SAMPLES, 4 SAMPLE_OUT_OF_RANGE, 5 TLAST_EARLY, 6 NO_TLAST.

## 2. Design

**Where state lives.** Configuration and counters are zero-initialized statics
`nbin[1]` (u32), `bounds[2]` (f32 lo/hi), `counters[16]` (u32) threaded through the
three command hooks (`configure`, `accumulate`, `emit`) — see `gen/hist.hpp:25-29`.
They persist across command hooks within one activation and are re-zeroed on each
`ap_start`, so nothing carries across activations.

**Loop structure.**
- `configure` (`hist_configure_impl.tpp`): emit ack, read the 3-word payload,
  validate (`nbin∉[1,16]` or `lo>=hi` → INVALID_CONFIG), store config, zero the
  16 counters (unrolled). No data loop.
- `accumulate` / DATA (`hist_accumulate_impl.tpp:75`): outer loop
  `for (i = 0; i < nsamp && !read_done; i += pf)` reads `pf` lanes per AXI word
  (`pf = 1` at 32-bit width); inner lane loop is `#pragma HLS UNROLL`. Each in-range
  sample does `local_counters[idx] = local_counters[idx] + 1` where
  `idx = truncate((x-lo)*scale)`, `scale = nbin/(hi-lo)`, all in f32, clamped to
  `[0, nbin-1]`. Out-of-range sets a flag; the burst is TLAST-framed and validated.
- `emit` / END (`hist_emit_impl.tpp:32`): `for (i = 0; i < n; ++i)` with
  `#pragma HLS PIPELINE II=1`, streams the counters.

**How II = 1 is reached.** The per-sample binning loop's only loop-carried
dependency is the counter read-modify-write. Accumulation is therefore done into a
local buffer `local_counters[16]` that is
`#pragma HLS ARRAY_PARTITION ... complete dim=1`
(`hist_accumulate_impl.tpp:63-64`) — a fully partitioned register file, not a BRAM —
so the read+write closes in one cycle. The local buffer is committed into the
persistent `counters` only on the success path
(`hist_accumulate_impl.tpp:123-126`), keeping the per-sample loop independent of the
persistent array. A 1R1W BRAM would make that read-after-write a 2-cycle access and
pin the loop at II = 2. Result: the binning loop `VITIS_LOOP_75_2` is
Interval = 1, Pipelined = yes, and the emit loop `VITIS_LOOP_32_1` is Interval = 1,
Pipelined = yes (`results/vitis/csynth.rpt`; `results/loop_df.csv`).

## 3. Targets

| Target | Required | Measured | Result | File |
|--------|----------|----------|--------|------|
| Binning-loop initiation interval | II = 1 | Interval = 1 (`VITIS_LOOP_75_2`, Pipelined yes) | PASS | `results/vitis/csynth.rpt` |
| END counter emission interval | II = 1 | Interval = 1 (`VITIS_LOOP_32_1`, Pipelined yes) | PASS | `results/vitis/csynth.rpt` |
| Frequency | 100 MHz | Estimated Fmax = 137.82 MHz | PASS | `results/vitis/solution1.log` |
| Bit-exactness (golden / C-sim / co-sim) | identical bits | counters [2,1,1,2] and status {0,0,0} identical in all three | PASS | `results/sim/counters.bin`, `data/counters_data.bin`, `results/vitis/counters.bin` (+ matching `regmap_status.json` in each dir) |
| Latency | no target | 95 cycles (min/avg/max) | (no target) | `results/vitis/hist_cosim.rpt` |
| Throughput | no target | 1 sample / clock (from II = 1) | (no target) | `results/vitis/csynth.rpt` |
| Resource usage | no target | 5 DSP, 3902 FF, 5095 LUT, 0 BRAM, 0 URAM | (no target) | `results/vitis/csynth.rpt` |

`results/loop_df.csv` independently confirms both inner loops at PipelineII = 1
(accumulate `1,18`; emit `1,2`).

## 4. Verification

**Independent check (golden model).** (a) Hand-computed basic case — `B=4, lo=0,
hi=4`, samples `[0.2, 1.8, 2.1, 3.9, 0.0, 3.2]` → `[2,1,1,2]` — passes, and two
further small cases (`accumulate_two_data`, `reconfig_mid_activation`) pass
(`results/sim/py_model.log`; `results/sim/counters.bin`). (b) NumPy reference —
`numpy_crosscheck` passes (`results/sim/py_model.log`). The golden model is a
separate Python implementation of the same arithmetic (f32 throughout) and its
outputs match the C/RTL path bit for bit on the basic activation.

**Per-scenario status.** "Golden" = Python model (`results/sim/py_model.log`,
`results/sim/*`). "C-sim / co-sim" = Vitis C-sim and RTL co-sim. The generated
testbench (`gen/hist_tb.cpp`) drives exactly one activation of the
CONFIG→DATA→END shape and always TLAST-frames the sample burst, so only the basic
activation (scenario 1) is exercised on the C/RTL path; all other scenarios are
verified by the golden model only.

| # | Scenario | Golden | C-sim | Co-sim |
|---|----------|--------|-------|--------|
| 1 | Basic hand-check → [2,1,1,2] | PASS (`results/sim/py_model.log`) | PASS (`data/counters_data.bin`, `results/verify_csim.json`, `results/vitis/hist_csim.log`) | PASS (`results/vitis/counters.bin`, `results/vitis/hist_cosim.log`, `results/vitis/hist_cosim.rpt`) |
| 2 | Two DATA accumulate, then END | PASS (`results/sim/py_model.log`) | not run (single-activation TB) | not run |
| 3 | Re-CONFIG mid-activation | PASS (`results/sim/py_model.log`) | not run | not run |
| 4 | Limits `B=1`; `B=16`; `nsamp=1`; `nsamp=128` | PASS — `limit_b1`, `limit_b16_nsamp128`, `limit_nsamp1` (`results/sim/py_model.log`) | not run | not run |
| 5 | Each error once, then a fresh passing activation | PASS for DATA_BEFORE_CONFIG, INVALID_CONFIG (×2), ZERO_SAMPLES, SAMPLE_OUT_OF_RANGE, and `fresh_activation_recovery` (`results/sim/py_model.log`); TLAST_EARLY and NO_TLAST implemented in `hist_accumulate_impl.tpp:108-120` but **not exercised by any committed simulation** | not run | not run |
| 6 | NumPy cross-check | PASS — `numpy_crosscheck` (`results/sim/py_model.log`) | not run | not run |

Bit-exact agreement on the basic activation across all three simulators:
`results/sim/counters.bin`, `data/counters_data.bin`, and `results/vitis/counters.bin`
are byte-identical (`02000000 01000000 01000000 02000000`), as are their
`regmap_status.json` files (`{halted:0, error:0, tx_id:0}`). `results/verify_csim.json`
records the C-sim output schema/array/status checks as pass.

## 5. Assumptions

- **Word packing / bit layout.** The prompt fixes field order (cmd_type, tx_id,
  nsamp) and payload order (nbin, lo, hi) but not exact bit positions or word
  boundaries. I used the generated little-endian packing as-is: 34-bit header into
  two 32-bit words, 80-bit config payload into three 32-bit words
  (`include/hist_cmd_hdr.h`, `include/cfg_payload.h`).
- **Stream width.** The prompt states samples/lo/hi are 32-bit; I ran both `s_in`
  and `m_out` at 32-bit words, so `pf` (floats per word) = 1.
- **Response ordering.** Each command emits its `RespHdr` (ack) before processing
  the command body, keeping one response header per command header and the stream in
  lockstep (`hist_configure_impl.tpp:25-28`, `hist_accumulate_impl.tpp:32-35`,
  `hist_emit_impl.tpp:20-23`).
- **Error priority.** Where several errors could apply, the kernel returns them in
  the golden model's priority order; for DATA the check order is out-of-range (4) →
  early TLAST (5) → no TLAST (6) (`hist_accumulate_impl.tpp:108-120`).
- **Fresh state per activation.** Each `ap_start` re-runs the kernel with the
  config/counter statics zero-initialized; nothing persists across activations.
- **Target part / clock.** The prompt sets no part; the course default is
  `xc7z020clg400-1`. I ran `xc7z020clg484-1` (same Zynq-7020 die, 484-pin package,
  `-1` grade; identical resource budget) at a 10 ns (100 MHz) clock — `run.tcl`,
  `results/vitis/csynth.rpt` ("Target device: xc7z020-clg484-1").

## 6. Open Items

- **C/RTL coverage is single-activation.** Scenarios 2–6 (accumulate, re-CONFIG,
  size limits, all six error codes, NumPy cross-check) are verified only by the
  golden Python model, not through C-sim or co-sim. The generated testbench
  (`gen/hist_tb.cpp`) drives exactly one activation of the CONFIG→DATA→END shape and
  always TLAST-frames the sample burst, so multi-DATA, re-CONFIG, and error
  activations are not exercised on the C/RTL path.
- **TLAST_EARLY (5) and NO_TLAST (6) are unexercised.** They are implemented in the
  kernel (`hist_accumulate_impl.tpp:108-120`) but no committed simulation drives an
  early or missing TLAST, so they are not reported as PASS.
- **In-class part.** Submission parts 1–2 (in-class prompt + results) are a separate
  deliverable; `cmdresp/inclass/` is not populated in this folder.

## 7. Run History

| Prompt version | What was built / changed | What passed |
|----------------|--------------------------|-------------|
| 1 (first run) | Built golden model, kernel hooks, generated TB. Added the missing `uint32_array_utils.h` (ArrayUtilsStep U32). First C-sim + co-sim of the basic activation. Binning loop measured II = 2 (counters in BRAM) — a miss vs the II = 1 target. | C-sim and co-sim of the basic activation; all golden scenarios. II = 1 target: FAIL at this run. |
| 1 (after fix) | Moved accumulation into a fully partitioned `local_counters` register file (committed to `counters` only on success). Re-ran C-sim, co-sim, and synthesis with `--force-step`. | All golden scenarios (`results/sim/py_model.log`); C-sim + co-sim of the basic activation bit-exact; both loops Interval = 1 (`results/vitis/csynth.rpt`, `results/loop_df.csv`); Fmax 137.82 MHz (`results/vitis/solution1.log`). |
