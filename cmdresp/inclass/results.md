# Scale and Offset — results (in class)

Bit-exact Python golden model (`scale_offset.py`) and Vitis HLS kernel
(`gen/scale_offset.cpp` + the three `scale_offset_*_impl.tpp` hooks), verified
against each other by C-simulation. Per the prompt this stops at C-sim: no
synthesis, no RTL co-sim.

## Scenarios

Each row is one activation (a fresh `scale_offset()` call). "PASS" means the
stated status was observed and, where the Python model and C-sim both ran the
scenario, that they agree bit for bit (data + status). Files are relative to
this folder.

| # | Scenario | Expected | PASS/FAIL | Evidence file |
|---|----------|----------|-----------|---------------|
| 1 | `basic_handcheck` — `a=2,b=1`, `[0.5,-1.25,3.0]` → `[2.0,-1.5,7.0]` | out `[2.0,-1.5,7.0]`, status `{0,0,0}` | **PASS** | `data/basic_handcheck/`, `results/sim/basic_handcheck/` |
| 2 | `two_data` — one CONFIG, two DATA bursts, one END | out `[2.0,-1.5,7.0,1.5]`, status `{0,0,0}` | **PASS** | `data/two_data/`, `results/sim/two_data/` |
| 3 | `reconfig` — CONFIG→DATA→CONFIG→DATA→END (2nd CONFIG `a=-1,b=0.5`) | out `[3.0,-1.5,3.5]`, status `{0,0,0}` | **PASS** | `data/reconfig/`, `results/sim/reconfig/` |
| 4 | `err_data_before_config` — DATA before any CONFIG | no out, status `{halted:1,error:1,tx_id:7}` | **PASS** | `data/err_data_before_config/`, `results/sim/err_data_before_config/` |
| 5 | `err_zero_samples` — DATA with `nsamp=0` | no out, status `{halted:1,error:2,tx_id:10}` | **PASS** | `data/err_zero_samples/`, `results/sim/err_zero_samples/` |
| 6 | `err_tlast_early` — DATA declares 3, burst is 2 (early TLAST) | no out, status `{halted:1,error:3,tx_id:31}` | **PASS** | `data/err_tlast_early/` (C-sim only, see below) |
| 7 | `err_no_tlast` — DATA declares 3, burst is 3 with no TLAST | no out, status `{halted:1,error:4,tx_id:41}` | **PASS** | `data/err_no_tlast/` (C-sim only, see below) |
| 8 | `fresh_pass` — clean recovery (`a=-2,b=3`, `[1.5,-0.5]` → `[0.0,4.0]`) | out `[0.0,4.0]`, status `{0,0,0}` | **PASS** | `data/fresh_pass/`, `results/sim/fresh_pass/` |

Hand-check coverage (prompt requirement (b)): the negative-`a` / non-zero-`b`
case is scenario 3's second CONFIG (`a=-1,b=0.5`: `2.0→-1.5`, `-3.0→3.5`), and
`fresh_pass` is a third independent case (`a=-2,b=3`: `1.5→0.0`, `-0.5→4.0`).
`basic_handcheck` is the prompt's own example.

## Python-model vs C-simulation bit-for-bit comparison

**Result: all common scenarios agree bit for bit on both the data outputs and
the status registers (`halted` / `error` / `tx_id`); `all_passed: true`.**
The full comparison (every output value and every status field, side by side)
is in `results/verify_csim.json`.

- **Compared (6 scenarios, both implementations):** `basic_handcheck`,
  `two_data`, `reconfig`, `err_data_before_config`, `err_zero_samples`,
  `fresh_pass`. Data compared byte-for-byte as raw little-endian f32
  (`outputs.bin`), status compared field-by-field (`regmap_status.json`).
  Golden side: `results/sim/<scenario>/`. C-sim side: `data/<scenario>/`.
  The comparison logic is `ValidateStep` in `scale_offset_build.py`.
- **C-sim only (2 scenarios):** `err_tlast_early`, `err_no_tlast`. The Python
  golden model's stream read (`get_pipelined`) reads a fixed word count and does
  not surface TLAST framing, so it cannot distinguish an early or a missing
  TLAST from a valid burst — it would report `NO_ERROR` for both. The kernel's
  C-sim `read_axi4_stream` does track TLAST, so these two are exercised only on
  the C-sim path (each confirms its error code, `error=3` / `error=4`, and no
  data payload). This is an honest limitation of the SimPy read API, not a
  skip.

## Where the prompt was silent or ambiguous, and what was chosen

- **"Not yet configured" is not inferable from `a`/`b`.** Because `a` and `b`
  may legitimately be `0.0`, the unconfigured condition cannot be read off the
  config values (unlike the take-home, where `nbin==0` signalled it). Chosen an
  **explicit `configured` flag** (`u32[1]`, 0 until a valid CONFIG) as the gate
  for DATA/END.
- **Kernel state is LOCAL, not `static`.** The prompt says "each `ap_start` runs
  a fresh kernel with no configuration set." Chosen **local** `configured`/`ab`
  in `gen/scale_offset.cpp` (re-created per call) so multi-activation C-sim
  genuinely starts each scenario unconfigured. `static` would leak state across
  calls and break that guarantee.
- **Word packing / stream width.** Fixed a **32-bit** AXI-4-Stream word width
  throughout (`in_bw = out_bw = 32`), one IEEE-754 f32 per word. Command header
  is `(cmd_type, tx_id, nsamp)`; the response header is `(tx_id)`.
- **Response ordering / lockstep.** The response header is emitted **before**
  the command body is read (ack-first), so every command is acknowledged exactly
  once and the output stream stays in lockstep with the input stream.
- **Error priority.** A DATA command is checked in the order: not-configured →
  `DATA_BEFORE_CONFIG`; `nsamp==0` → `ZERO_SAMPLES`; then burst framing
  (`TLAST_EARLY` / `NO_TLAST`). Invalid-parameter checks (config, `nsamp`) happen
  **before any payload is read**, per the prompt.
- **TLAST handling.** A valid burst is exactly `nsamp` samples with a TLAST on
  the final word; TLAST before the count is reached → `TLAST_EARLY`; reaching the
  count with no TLAST → `NO_TLAST`.
- **Part and clock.** Target `xc7z020clg484-1` (Kintex-7) at 100 MHz
  (10 ns period), matching the take-home; C-sim is clock-period-agnostic for
  these functional checks but the part/clock are fixed in `run.tcl`.
- **Arithmetic.** `y = (a * x) + b` computed **entirely in 32-bit float** in both
  implementations, so the bits are identical (no 64-bit widening).

## How to reproduce

```bash
# Python env with waveflow + numpy:
python scale_offset.py            # golden model: 6 scenarios, prints ALL PASSED

# Full flow to C-sim + comparison (generates include/, golden evidence,
# C-sim evidence, and results/verify_csim.json):
WAVEFLOW_VITIS_PATH="<vitis>/Vitis/bin/vitis-run.bat" \
python scale_offset_build.py --through validate
```

## Run history

| Run | What was built / changed | Result |
|-----|--------------------------|--------|
| 1 | Golden model `scale_offset.py` (schemas, accelerator, SimPy TB, 6 scenarios) | `python scale_offset.py` → **ALL PASSED** (6/6) |
| 2 | Kernel + hooks + multi-activation TB + build DAG + `run.tcl`; generated `include/*.h` | `gen_include` + `py_sim` → **PASS** |
| 3 | C-sim via Vitis (`run.tcl`, csim only) then comparison | `CSim done with 0 errors`, `WAVEFLOW_SUCCESS: scale_offset C-Simulation passed`; `validate` → **PASS**, `all_passed: true` |

## File inventory (evidence the results above come from)

- `results/sim/<scenario>/outputs.bin` + `regmap_status.json` — Python golden
  model, 6 common scenarios.
- `results/sim/py_model.log` — golden-model per-scenario summary.
- `data/<scenario>/outputs.bin` + `regmap_status.json` — C-sim, all 8 scenarios.
- `results/csim.log` — Vitis C-sim run log (`CSim done with 0 errors`,
  `WAVEFLOW_SUCCESS: scale_offset C-Simulation passed`).
- `results/verify_csim.json` — the Python-vs-C-sim comparison report.
- `scale_offset_build.py` — the build/verify DAG (steps + comparison logic).
