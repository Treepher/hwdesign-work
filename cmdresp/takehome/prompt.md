# Command-Response Accelerator — Histogram (take-home)

Build the accelerator described below, in this folder, with Waveflow. Follow the
`stream_inband` Waveflow example (`poly.py`, `poly_build.py`, `poly_evaluate_impl.tpp`)
as the structural reference, and follow the **command-response pattern** wherever the
two differ — in particular, configuration travels on the **input stream as a CONFIG
command**, not on the AXI-Lite register map.

Write a bit-exact Python golden model and the Vitis HLS kernel. Run the Python model,
C simulation, and RTL co-simulation, and check that all of them agree **bit for bit** on
both the data outputs and the status registers. Measure latency and throughput from
co-simulation, and resource usage from synthesis. Commit the raw evidence those
numbers come from (the C-sim log, the co-sim log, the comparison output, and the
synthesis report) in this folder; do not commit Vitis project directories or bulky logs
nothing cites.

## Operation

The accelerator computes a **histogram**. It partitions the half-open interval
`[lo, hi)` into `B` equal-width bins and counts how many input samples land in each bin.

- A **CONFIG** command sets `B` (number of bins), `lo` (lower bound, inclusive) and
  `hi` (upper bound, exclusive), and **resets all `B` counters to zero**.
- A **DATA** command delivers a burst of `nsamp` 32-bit float samples. Each sample `x`
  is assigned to bin

      idx = truncate( (x - lo) * scale )      where  scale = B / (hi - lo)

  with `(x - lo)`, `scale`, and the product all computed in **32-bit float**, the result
  truncated toward zero to an integer, and clamped into `[0, B-1]`. The bin counter is
  incremented by one. A sample with `x < lo` or `x >= hi` is out of range and is an
  error (see Errors).
- Counters **accumulate across DATA commands** within one activation. A DATA command
  does not reset them.
- An **END** command emits the `B` counters on the output stream in bin order
  (bin 0 first, bin `B-1` last) and returns the kernel.

A second CONFIG in the same activation replaces `B`/`lo`/`hi` and resets the counters
again. Nothing persists across activations: each `ap_start` runs a fresh kernel.

**Hand-checkable example.** `B=4, lo=0.0, hi=4.0` (bin width 1.0). Samples
`[0.2, 1.8, 2.1, 3.9, 0.0, 3.2]` map to bins `[0, 1, 2, 3, 0, 3]`, so the counters
are `[2, 1, 1, 2]`.

## Requirements

- **Data format.** Samples, `lo`, and `hi` are IEEE-754 single-precision floats (32
  bits). Bin counters are 32-bit unsigned integers. `B`, `tx_id`, and `nsamp` are
  16-bit unsigned. Every step of the bin-index computation is done in 32-bit float so
  the Python model and the C++ kernel produce **identical** bits.
- **Rounding / saturation.** The bin index is `(x - lo) * scale` truncated toward zero
  and clamped to `[0, B-1]`; counters saturate by never overflowing (see largest sizes).
  There is no other rounding.
- **Largest sizes.** `1 <= B <= 16`; `1 <= nsamp <= 128` per DATA command; `tx_id` is
  16 bits.
- **Throughput / latency target.** The per-sample binning loop must reach an
  initiation interval of **II = 1** (one sample accepted per clock) at 100 MHz, and the
  END counter emission must stream at II = 1.
- **Independent check.** (a) The hand-computed example above, plus at least two more
  small cases worked by hand. (b) A NumPy reference: for in-range samples the counters
  must equal `np.histogram(samples, bins=B, range=(lo, hi))[0]`.

## Commands and parameters

Every command travels on the input stream and its header is a TLAST. The register map
carries only `ap_start` / `ap_done` plus the status registers.

| Command | Header `(cmd_type, tx_id, nsamp)` | Payload                       | Response                          |
|---------|-----------------------------------|-------------------------------|-----------------------------------|
| CONFIG  | `(0, tx_id, 3)`                    | `B` (u16), `lo` (f32), `hi` (f32) | `RespHdr(tx_id)`            |
| DATA    | `(1, tx_id, nsamp)`                | `nsamp` × f32 samples         | `RespHdr(tx_id)`                  |
| END     | `(2, tx_id, 0)`                    | —                             | `RespHdr(tx_id)` + `B` × u32 counters |

Invalid parameters are checked **before any payload is read**: `B < 1` or `B > 16`
raises `INVALID_CONFIG`; `lo >= hi` raises `INVALID_CONFIG`.

## Errors

Status registers on AXI-Lite: `halted` (1 bit), `error` (code), `tx_id` (of the
offending command). On any error the kernel sets these registers, **closes any output
burst it has started with a TLAST**, reads nothing further, and returns. The contents
of the input stream after an error are undefined; recovery is the host's job.

| Code | Name                  | Meaning                                            |
|------|-----------------------|----------------------------------------------------|
| 0    | NO_ERROR              | —                                                  |
| 1    | DATA_BEFORE_CONFIG    | DATA or END with no preceding CONFIG               |
| 2    | INVALID_CONFIG        | `B` out of `[1,16]` or `lo >= hi`                  |
| 3    | ZERO_SAMPLES          | DATA with `nsamp = 0`                              |
| 4    | SAMPLE_OUT_OF_RANGE   | a sample `x < lo` or `x >= hi`                     |
| 5    | TLAST_EARLY           | sample burst shorter than `nsamp` (early TLAST)    |
| 6    | NO_TLAST              | sample burst longer than `nsamp` (missing TLAST)   |

## Verification

Run the Python model, C simulation, and RTL co-simulation and confirm all three agree
bit for bit on the counters and on `halted` / `error` / `tx_id`. Each scenario below is
one activation, and **every error scenario is followed by a fresh activation that must
pass**:

1. Basic hand-check (`B=4, lo=0, hi=4`, the six samples above → `[2,1,1,2]`).
2. Two DATA commands that accumulate, then END.
3. A re-CONFIG mid-activation (new `B`/`lo`/`hi` resets the counters), then DATA, then END.
4. Sizes at the limits: `B=1`; `B=16`; `nsamp=1`; `nsamp=128`.
5. Each error once: `DATA_BEFORE_CONFIG`, `INVALID_CONFIG`, `ZERO_SAMPLES`,
   `SAMPLE_OUT_OF_RANGE`, `TLAST_EARLY`, `NO_TLAST`.
6. NumPy cross-check on an in-range burst.

Measure latency and throughput from co-simulation and resources from synthesis. Where
the prompt sets no target, write "no target".

## results.md

`results.md` must contain exactly these seven sections, in this order:

1. **Interface** — the command and response layouts, how values pack into words, the
   status registers, and the error codes.
2. **Design** — the loop structure, where state (config and counters) is kept, and how
   the II = 1 target is reached, or why not.
3. **Targets** — each target with the required value, the measured value, PASS or FAIL,
   and the file it came from.
4. **Verification** — the model's independent check, then one row per scenario above.
5. **Assumptions** — every place the prompt was silent or ambiguous, and what the agent
   chose.
6. **Open Items** — questions the agent could not settle, and anything that failed or
   was not done.
7. **Run History** — one row per run: the prompt version, what was built or changed,
   and what passed.

Every number and PASS/FAIL in Targets and Verification must name the file it came from.

## Revision history

| Version | Change        | Why |
|---------|---------------|-----|
| 1       | First version |     |
