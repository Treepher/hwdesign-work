# Command-Response Accelerator — Scale and Offset (in class)

Build the accelerator described below, in this folder, with Waveflow. Follow the
`stream_inband` Waveflow example (`poly.py`, `poly_build.py`,
`poly_evaluate_impl.tpp`) as the structural reference, and follow the
**command-response pattern** wherever the two differ — in particular,
configuration travels on the **input stream as a CONFIG command**, not on the
AXI-Lite register map.

Write a bit-exact Python golden model and the Vitis HLS kernel. Run the Python
model and C simulation, and check that they agree **bit for bit** on both the
data outputs and the status registers. **Stop once C simulation passes**: no
synthesis and no RTL co-simulation for this assignment.

## Operation

The accelerator applies a **scale and offset**: `y = a * x + b`.

- A **CONFIG** command sets the scale `a` and the offset `b` (both 32-bit
  float). It takes effect from that point in the activation.
- A **DATA** command delivers a burst of `nsamp` 32-bit float samples and the
  kernel emits `nsamp` 32-bit float outputs `y[i] = a * x[i] + b`, in order,
  using the most recent CONFIG's `a` and `b`.
- **`a` and `b` persist across DATA commands** within one activation. A second
  CONFIG replaces them. Nothing persists across activations: each `ap_start`
  runs a fresh kernel with no configuration set (see errors).
- An **END** command terminates the activation and returns the kernel.

**Hand-checkable example.** `a = 2.0, b = 1.0`. Inputs `[0.5, -1.25, 3.0]`
produce outputs `[2.0, -1.5, 7.0]`.

## Requirements

- **Data format.** Samples, outputs, `a`, and `b` are IEEE-754
  single-precision floats (32 bits). The output is computed as `(a * x) + b`
  entirely in 32-bit float so the Python model and the C++ kernel produce
  **identical** bits.
- **Largest sizes.** `1 <= nsamp <= 128` per DATA command; `tx_id` is 16 bits.
- **Independent check.** (a) The hand-computed example above. (b) At least two
  more small cases worked by hand (including a negative `a` and a non-zero `b`).

## Commands and parameters

Every command travels on the input stream and its header is a TLAST. The
register map carries only `ap_start` / `ap_done` plus the status registers.

| Command | Header `(cmd_type, tx_id, nsamp)` | Payload            | Response                          |
|---------|-----------------------------------|--------------------|-----------------------------------|
| CONFIG  | `(0, tx_id, 2)`                    | `a` (f32), `b` (f32) | `RespHdr(tx_id)`                 |
| DATA    | `(1, tx_id, nsamp)`                | `nsamp` × f32 in   | `RespHdr(tx_id)` + `nsamp` × f32 out |
| END     | `(2, tx_id, 0)`                    | —                  | `RespHdr(tx_id)`                  |

The kernel sends the response header **before** it processes the command body,
so every command is acknowledged exactly once and the output stream stays in
lockstep with the input stream.

Invalid parameters are checked **before any payload is read**: a DATA command
with `nsamp = 0` raises `ZERO_SAMPLES`.

## Errors

Status registers on AXI-Lite: `halted` (1 bit), `error` (code), `tx_id` (of the
offending command). On any error the kernel sets these registers, **closes any
output burst it has started with a TLAST**, reads nothing further, and returns.
The contents of the input stream after an error are undefined; recovery is the
host's job.

| Code | Name                  | Meaning                                            |
|------|-----------------------|----------------------------------------------------|
| 0    | NO_ERROR              | —                                                  |
| 1    | DATA_BEFORE_CONFIG    | a DATA or END command before any CONFIG            |
| 2    | ZERO_SAMPLES          | DATA with `nsamp = 0`                              |
| 3    | TLAST_EARLY           | sample burst shorter than `nsamp` (early TLAST)    |
| 4    | NO_TLAST              | sample burst reaches `nsamp` without a TLAST       |

## Verification

Run the Python model and C simulation and confirm both agree bit for bit on
the outputs and on `halted` / `error` / `tx_id`. Each scenario below is one
activation, and **every error scenario is followed by a fresh activation that
must pass**:

1. Basic hand-check (`a=2.0, b=1.0`, `[0.5, -1.25, 3.0]` → `[2.0, -1.5, 7.0]`).
2. Two DATA commands in one activation — both use the same `a`/`b`.
3. A re-CONFIG mid-activation — the new `a`/`b` apply to the DATA after it.
4. Each error once: `DATA_BEFORE_CONFIG`, `ZERO_SAMPLES`, `TLAST_EARLY`,
   `NO_TLAST`, each followed by a fresh passing activation.

Commit the raw evidence those results come from (the Python-model log and the
C-sim output) in this folder; do not commit bulky logs nothing cites.

## results.md

`results.md` must contain, at minimum:

- a table of **every scenario** above with **PASS or FAIL** and the file the
  result came from;
- the result of the Python-model vs C-simulation bit-for-bit comparison and the
  files it compared;
- a list of **every place this prompt was silent or ambiguous, and what you
  chose**;
- a run history: one row per run, what was built or changed, and what passed.

## Revision history

| Version | Change        | Why |
|---------|---------------|-----|
| 1       | First version |     |
