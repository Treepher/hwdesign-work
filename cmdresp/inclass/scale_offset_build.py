"""scale_offset_build.py — build/verify DAG for the scale-and-offset accelerator (in class).

Adapts ``takehome/hist_build.py`` to the in-class command-response kernel in
``scale_offset.py``.  The in-class deliverable stops at C-simulation per the
prompt: there is **no** C-synthesis and **no** RTL co-simulation step.

Differences from the take-home pipeline:

* The kernel (``gen/scale_offset.{hpp,cpp}``), the three hook bodies
  (``scale_offset_*_impl.tpp``), and the multi-activation testbench
  (``gen/scale_offset_tb.cpp``) are **hand-written** and committed.  The take-home
  generated the kernel and testbench from the ``HostActivated`` class via
  ``HlsCodegenStep``; that is deliberately *not* done here because the in-class
  kernel must keep its per-activation state ``configured`` / ``ab`` as **local**
  (re-created for every ``ap_start``) rather than ``static`` -- the hand-written
  ``gen/scale_offset.cpp`` does exactly that.  So the only Waveflow codegen step
  is ``HlsGenIncludeStep`` (schema / array-utility headers under ``include/``).
* The C-sim testbench drives **eight** scenarios, each as a fresh activation
  (a fresh ``scale_offset()`` call with fresh streams and status registers).
  Six of them are also run by the Python golden model and compared bit-for-bit;
  two (``err_tlast_early`` / ``err_no_tlast``) are C-sim only, because the SimPy
  stream read cannot surface an early or missing TLAST.
* ``ValidateStep`` compares the two implementations itself (byte-for-bit on the
  ``outputs.bin`` payloads, field-by-field on ``regmap_status.json``) and writes
  ``results/verify_csim.json``.

Run up to and including C-sim validation with::

    python scale_offset_build.py --through validate
"""
from __future__ import annotations

import argparse
import json
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from waveflow.build.build import BuildConfig, BuildDag, BuildStep, SourceStep
from waveflow.build.streamutils import StreamUtilsStep
from waveflow.hw.clock import Clock
from waveflow.hw.dataschema import DataSchemaStep
from waveflow.hw.arrayutils import ArrayUtilsStep
from waveflow.simulation.logger import NullLogger
from waveflow.simulation.simulation import Simulation
from waveflow.toolchain import toolchain

try:
    from cmdresp.inclass.scale_offset import (
        Float32, ScaleAccel, ScaleCmdHdr, ScaleCmdType, ScaleError, ScaleRespHdr,
        ScaleTB, U32,
        SCHEMA_CLASSES, WORD_BW_SUPPORTED,
        connect, run_scenario, SCENARIOS,
    )
except ModuleNotFoundError:
    from scale_offset import (  # type: ignore[no-redef]  # direct execution
        Float32, ScaleAccel, ScaleCmdHdr, ScaleCmdType, ScaleError, ScaleRespHdr,
        ScaleTB, U32,
        SCHEMA_CLASSES, WORD_BW_SUPPORTED,
        connect, run_scenario, SCENARIOS,
    )


_SOURCE_DIR = Path(__file__).resolve().parent

# The six scenarios that are driven by BOTH the Python golden model and the C-sim
# testbench and therefore compared bit-for-bit (data outputs + status registers).
# ``err_tlast_early`` and ``err_no_tlast`` are C-sim only (see the module docstring)
# and live only in the C-sim testbench.
COMMON_SCENARIOS = [
    "basic_handcheck",
    "two_data",
    "reconfig",
    "err_data_before_config",
    "err_zero_samples",
    "fresh_pass",
]

# The two C-sim-only framing-error scenarios, and the status code each must report.
CSIM_ONLY_SCENARIOS = {
    "err_tlast_early": int(ScaleError.TLAST_EARLY),
    "err_no_tlast":    int(ScaleError.NO_TLAST),
}


@dataclass(kw_only=True)
class HlsGenIncludeStep(BuildStep):
    """Generate the schema and array-utility headers under ``include/``."""
    description = "Generate schema and utility headers needed for the Vitis flow."
    consumes    = ["scale_source"]
    params      = {}
    include_dir: str = "include"

    @property
    def produces(self) -> dict:  # type: ignore[override]
        return {"include_dir": Path(self.include_dir)}

    def run(self, config: BuildConfig, **_) -> dict:
        inner_dag = BuildDag()
        inner_dag.add(StreamUtilsStep(output_dir=self.include_dir))
        for cls in SCHEMA_CLASSES:
            inner_dag.add(DataSchemaStep(cls, word_bw_supported=WORD_BW_SUPPORTED,
                                         include_dir=self.include_dir))
        inner_dag.add(ArrayUtilsStep(U32, WORD_BW_SUPPORTED))
        inner_dag.add(ArrayUtilsStep(Float32, WORD_BW_SUPPORTED))
        inner_results = inner_dag.run(config)
        failed = [n for n, r in inner_results.items() if not r.success]
        if failed:
            raise RuntimeError(f"Code generation failed: {failed}")
        return {"include_dir": config.root_dir / self.include_dir}


@dataclass(kw_only=True)
class PySimStep(BuildStep):
    """Run the Python golden model on each common scenario and write the golden
    per-scenario evidence: ``outputs.bin`` (raw f32, all valid DATA outputs in
    command order) and ``regmap_status.json`` (halted / error / tx_id)."""
    description = "Run the Python SimPy golden model and write per-scenario golden outputs."
    consumes    = ["scale_source"]
    produces    = {"sim_dir": Path("results/sim")}
    params      = {}

    def run(self, config: BuildConfig, **_) -> dict:
        clk_freq = 100e6
        sim_dir = config.root_dir / "results" / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)

        # SCENARIOS is the authoritative (name, commands, expect) list.
        by_name = {name: (commands, expect) for name, commands, expect in SCENARIOS}
        for name in COMMON_SCENARIOS:
            commands, expect = by_name[name]
            sim = Simulation()
            clk = Clock(freq=clk_freq)
            tb = run_scenario(sim, clk, commands)

            scen_dir = sim_dir / name
            scen_dir.mkdir(parents=True, exist_ok=True)

            # Golden data payload: every valid DATA burst's outputs, concatenated
            # in command order, as raw little-endian f32.
            if tb.data_outputs:
                out = np.concatenate([np.asarray(o, dtype=np.float32) for o in tb.data_outputs])
            else:
                out = np.array([], dtype=np.float32)
            out.tofile(scen_dir / "outputs.bin")

            status = {
                "halted": int(tb.halted) if tb.halted is not None else 0,
                "error":  int(tb.error)  if tb.error  is not None else int(ScaleError.NO_ERROR),
                "tx_id":  int(tb.tx_id_status) if tb.tx_id_status is not None else 0,
            }
            (scen_dir / "regmap_status.json").write_text(
                json.dumps(status, indent=2), encoding="utf-8")

        # A short, human-readable golden-model summary for the evidence log.
        summary_lines = ["=== Scale-and-offset golden-model evidence (per scenario) ==="]
        for name in COMMON_SCENARIOS:
            scen_dir = sim_dir / name
            out = np.fromfile(scen_dir / "outputs.bin", dtype=np.float32)
            status = json.loads((scen_dir / "regmap_status.json").read_text(encoding="utf-8"))
            summary_lines.append(
                f"{name}: outputs={out.tolist()} status={status}")
        (sim_dir / "py_model.log").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        return {"sim_dir": sim_dir}


@dataclass(kw_only=True)
class CSimStep(BuildStep):
    """Invoke Vitis HLS C-simulation (multi-activation testbench)."""
    description = "Invoke Vitis HLS C-simulation."
    consumes    = ["scale_source", "include_dir"]
    produces    = {"csim_data_dir": Path("data"), "csim_log": Path("results/csim.log")}
    params      = {"live_output": False, "clk_freq": 100e6}

    def run(self, config: BuildConfig, live_output, clk_freq, **_) -> dict:
        # The C-sim testbench writes one sub-directory per scenario (six common
        # plus the two C-sim-only framing scenarios); create them up front so the
        # testbench only has to open files.
        for name in COMMON_SCENARIOS + list(CSIM_ONLY_SCENARIOS):
            (config.root_dir / "data" / name).mkdir(parents=True, exist_ok=True)

        vitis_env = {"WAVEFLOW_SCALE_COSIM": "0",
                     "WAVEFLOW_SCALE_TRACE_LEVEL": "none",
                     "WAVEFLOW_SCALE_CLK_PERIOD_NS": f"{1e9 / clk_freq:g}"}
        result = toolchain.run_vitis_hls(
            config.root_dir / "run.tcl",
            work_dir=config.root_dir,
            capture_output=not live_output,
            env=vitis_env,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Persist the run log as evidence.
        log_path = config.root_dir / "results" / "csim.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            (result.stdout or "") + ("\n" if result.stdout and result.stderr else "")
            + (result.stderr or ""),
            encoding="utf-8",
        )
        return {"csim_data_dir": config.root_dir / "data", "csim_log": log_path}


@dataclass(kw_only=True)
class ValidateStep(BuildStep):
    """Compare the Python golden model against the C-sim, scenario by scenario.

    * For every common scenario the ``outputs.bin`` payloads must match
      byte-for-byte and the ``regmap_status.json`` fields must match exactly.
    * The two C-sim-only framing scenarios must each report their expected error
      code (and no data payload) -- there is no golden counterpart to compare.
    """
    description = "Compare golden model vs C-sim outputs and status registers."
    consumes    = ["sim_dir", "csim_data_dir"]
    produces    = {"verify_report": Path("results/verify_csim.json")}
    params      = {}

    @staticmethod
    def _read_evidence(base: Path, name: str) -> tuple[np.ndarray, dict]:
        scen = base / name
        out = np.fromfile(scen / "outputs.bin", dtype=np.float32)
        status = json.loads((scen / "regmap_status.json").read_text(encoding="utf-8"))
        return out, status

    def run(self, config: BuildConfig, sim_dir, csim_data_dir, **_) -> dict:
        sim_dir = Path(sim_dir)
        csim_dir = Path(csim_data_dir)
        report: dict = {"common": {}, "csim_only": {}, "all_passed": True}

        for name in COMMON_SCENARIOS:
            g_out, g_status = self._read_evidence(sim_dir, name)
            a_out, a_status = self._read_evidence(csim_dir, name)
            out_match = (g_out.shape == a_out.shape and
                         np.array_equal(g_out.view(np.uint32), a_out.view(np.uint32)))
            status_match = (g_status.get("halted") == a_status.get("halted") and
                            g_status.get("error")  == a_status.get("error") and
                            g_status.get("tx_id")  == a_status.get("tx_id"))
            ok = out_match and status_match
            report["all_passed"] = report["all_passed"] and ok
            report["common"][name] = {
                "outputs_match":  bool(out_match),
                "status_match":   bool(status_match),
                "golden_outputs": g_out.tolist(),
                "csim_outputs":   a_out.tolist(),
                "golden_status":  g_status,
                "csim_status":    a_status,
                "pass":           bool(ok),
            }

        for name, expected_err in CSIM_ONLY_SCENARIOS.items():
            a_out, a_status = self._read_evidence(csim_dir, name)
            ok = (a_out.size == 0 and a_status.get("error") == expected_err
                  and a_status.get("halted") == 1)
            report["all_passed"] = report["all_passed"] and ok
            report["csim_only"][name] = {
                "expected_error":  expected_err,
                "csim_outputs":    a_out.tolist(),
                "csim_status":     a_status,
                "golden":          "not exercised (SimPy read cannot surface TLAST framing)",
                "pass":            bool(ok),
            }

        report_path = config.root_dir / "results" / "verify_csim.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {"verify_report": report_path}


def build_scale_offset_dag() -> BuildDag:
    """Build the in-class scale-and-offset accelerator pipeline DAG (to C-sim)."""
    dag = BuildDag()
    dag.add(SourceStep(
        artifact="scale_source", path=_SOURCE_DIR / "scale_offset.py",
        description="Python source for the golden model, schemas, and testbench.",
    ))

    # HLS code generation (headers only; the kernel + TB are hand-written).
    dag.add(HlsGenIncludeStep(name="gen_include"))

    # Python golden model -> per-scenario golden evidence.
    dag.add(PySimStep(name="py_sim"))

    # C-sim functional verification (multi-activation), then the comparison.
    dag.add(CSimStep(name="csim"))
    dag.add(ValidateStep(name="validate"))
    return dag


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the scale-and-offset build/verify DAG (to C-sim).")
    parser.add_argument(
        "--through", metavar="STEP", default="validate",
        choices=["gen_include", "py_sim", "csim", "validate"],
        help="Run the DAG up to and including this step.",
    )
    parser.add_argument("--list-steps", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--clk-freq", type=float, default=100e6, metavar="HZ")
    parser.add_argument("--live-output", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-step", metavar="STEP", action="append", default=[])
    args = parser.parse_args()

    dag = build_scale_offset_dag()

    if args.list_steps:
        for name in dag.step_names():
            print(name)
        return

    config = BuildConfig(
        root_dir=_SOURCE_DIR,
        params={"clk_freq": args.clk_freq, "live_output": args.live_output},
    )

    if args.status:
        for entry in dag.results_status(config):
            now = _time.time()
            age = f"{(now - entry['mtime']) / 3600:.1f}h ago" if entry['mtime'] else "—"
            mark = "✓" if entry['exists'] else "✗"
            print(f"  {entry['artifact']:<16} {entry['produced_by']:<16} {mark}  {age}")
        return

    force: bool | list[str] = True if args.force else (args.force_step or False)

    def on_step_begin(step, will_run, paths):
        print(f"{step.name}:")
        for artifact, p in paths.items():
            try:
                display = p.relative_to(config.root_dir)
            except ValueError:
                display = p
            print(f"    {display}")
        if will_run:
            print("    RUNNING...")

    def on_step_end(step, result):
        if not result.success:
            print(f"    FAILED: {result.message}")
        elif result.skipped:
            print("    UP-TO-DATE")
        else:
            print("    PASSED")

    dag.run(config, through=args.through, force=force,
            on_step_begin=on_step_begin, on_step_end=on_step_end)


if __name__ == "__main__":
    _main()
