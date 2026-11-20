"""hist_build.py — build/verify DAG for the histogram accelerator (take-home).

Pipeline (the cmdresp command-response pattern is authoritative):

    golden model (SimPy)  ->  C-sim functional  ->  C-sim validation
    ->  RTL co-sim functional  ->  co-sim validation  ->  C-synth inspect
    ->  co-sim throughput (delta method)

The kernel (``gen/hist.cpp`` + the three ``hist_*_impl.tpp`` hooks), the
multi-activation functional testbench (``gen/hist_tb.cpp``), and the throughput
testbench (``gen/hist_tp_tb.cpp``) are **hand-written and committed**; the only
Waveflow codegen step is ``HlsGenIncludeStep`` (schema / array-utility headers
under ``include/``).  This mirrors the in-class flow: the kernel keeps its
per-activation state ``nbin`` / ``bounds`` / ``counters`` as **local** variables
(re-created for every ``ap_start``), so every scenario genuinely starts
unconfigured.

Scenario set (21 activations, each a fresh ``hist()`` call in the C-sim / co-sim
testbench):

* 12 are also driven by the Python golden model and compared bit-for-bit;
* 2 (``err_tlast_early`` / ``err_no_tlast``) are C-sim / co-sim only, because
  the SimPy stream read reads a fixed word count and cannot surface an early or
  a missing TLAST;
* 7 are the fresh-activation recoveries: a clean passing hand-check that follows
  each of the six error classes (the golden re-runs the hand-check for each).

Evidence layout (all committed, all read by ``results.md``):

* ``results/sim/<name>/{counters.bin, regmap_status.json}``  — golden, 19 scenarios
* ``data/<name>/{counters.bin, regmap_status.json}``          — C-sim, 21 scenarios
* ``results/vitis/cosim/<name>/{...}``                          — co-sim, 21 scenarios
* ``results/vitis/{csim.log, cosim.log, hist_cosim.rpt, csynth.rpt, loop_df.csv}``
* ``results/vitis/throughput.json`` + ``results/vitis/tp_{1,128}/hist_cosim.rpt``
* ``results/verify_csim.json`` (golden vs C-sim) and ``results/verify_cosim.json``
  (C-sim vs co-sim)

Run the whole flow with::

    WAVEFLOW_VITIS_PATH="<vitis>/Vitis/bin/vitis-run.bat" \\
    python hist_build.py --through throughput
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waveflow.build.build import BuildConfig, BuildDag, BuildStep, SourceStep
from waveflow.build.streamutils import StreamUtilsStep
from waveflow.hw.clock import Clock
from waveflow.hw.dataschema import DataSchemaStep
from waveflow.hw.arrayutils import ArrayUtilsStep
from waveflow.simulation.simulation import Simulation
from waveflow.toolchain import toolchain

try:
    from cmdresp.takehome.hist import (
        CfgPayload, CounterArray, Float32, HistAccel, HistCmdHdr, HistCmdType,
        HistError, HistRespHdr, HistTB, SampArr, U32,
        SCHEMA_CLASSES, WORD_BW_SUPPORTED,
        connect, run_scenario, SCENARIOS, verify_all,
    )
except ModuleNotFoundError:
    from hist import (  # type: ignore[no-redef]  # direct execution
        CfgPayload, CounterArray, Float32, HistAccel, HistCmdHdr, HistCmdType,
        HistError, HistRespHdr, HistTB, SampArr, U32,
        SCHEMA_CLASSES, WORD_BW_SUPPORTED,
        connect, run_scenario, SCENARIOS, verify_all,
    )


_SOURCE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Scenario set.
# ---------------------------------------------------------------------------
# Driven by BOTH the golden model and the C-sim / co-sim testbench (compared bit-for-bit).
COMMON_SCENARIOS = [
    "basic_handcheck",
    "accumulate_two_data",
    "reconfig_mid_activation",
    "limit_b1",
    "limit_b16_nsamp128",
    "limit_nsamp1",
    "err_data_before_config",
    "err_invalid_config_b0",
    "err_invalid_config_lo_ge_hi",
    "err_zero_samples",
    "err_sample_out_of_range",
    "numpy_crosscheck",
]

# C-sim / co-sim only: the SimPy read cannot surface TLAST framing.
CSIM_ONLY_SCENARIOS = {
    "err_tlast_early": int(HistError.TLAST_EARLY),   # 5
    "err_no_tlast":    int(HistError.NO_TLAST),      # 6
}

# Fresh-activation recovery: a clean passing hand-check after each error class.
RECOVER_SCENARIOS = [
    "recover_data_before_config",
    "recover_invalid_config_b0",
    "recover_invalid_config_lo_ge_hi",
    "recover_zero_samples",
    "recover_sample_out_of_range",
    "recover_tlast_early",
    "recover_no_tlast",
]

# Golden produces evidence for these (12 common + 7 recoveries).
GOLDEN_SCENARIOS = COMMON_SCENARIOS + RECOVER_SCENARIOS
# The full 21-activation set the C-sim / co-sim testbench drives.
ALL_TB_SCENARIOS = COMMON_SCENARIOS + list(CSIM_ONLY_SCENARIOS) + RECOVER_SCENARIOS


# ---------------------------------------------------------------------------
# Small shared helpers.
# ---------------------------------------------------------------------------
def _run_vitis(config: BuildConfig, env: dict, live_output: bool, log_path: Path):
    result = toolchain.run_vitis_hls(
        config.root_dir / "run.tcl",
        work_dir=config.root_dir,
        capture_output=not live_output,
        env=env,
    )
    out = result.stdout or ""
    err = result.stderr or ""
    if out:
        print(out)
    if err:
        print(err)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(out + ("\n" if out and err else "") + err, encoding="utf-8")
    return result


def _copy_latest(root: Path, pattern: str, dest: Path) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if matches:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matches[-1], dest)
        return matches[-1]
    return None


def _scenario_spec(name: str) -> tuple[list[dict], dict]:
    """Return ``(commands, expect)`` for a named scenario.  Recoveries re-run the
    hand-check (scenario 1) as a fresh activation."""
    by_name = {n: (c, e) for n, c, e in SCENARIOS}
    if name in RECOVER_SCENARIOS:
        return by_name["basic_handcheck"]
    return by_name[name]


def _read_evidence(base: Path, name: str) -> tuple[np.ndarray, dict]:
    scen = base / name
    out = np.fromfile(scen / "counters.bin", dtype=np.uint32)
    status = json.loads((scen / "regmap_status.json").read_text(encoding="utf-8"))
    return out, status


def _total_cycles_from_report(rpt: Path) -> int:
    """Total Execution Time (clock cycles) = the last numeric cell of the
    ``Verilog | Pass`` row in the Vitis co-sim report."""
    text = Path(rpt).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "Verilog" in line and "Pass" in line:
            cells = [c.strip() for c in line.split("|")]
            nums = [c for c in cells if c.isdigit()]
            if nums:
                return int(nums[-1])
    raise RuntimeError(f"could not parse Total Execution Time from {rpt}")


# ---------------------------------------------------------------------------
# Step definitions.
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class HlsGenIncludeStep(BuildStep):
    description = "Generate schema and utility headers needed for the Vitis flow."
    consumes    = ["hist_source"]
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
    """Run the golden model; self-check it, then write per-scenario golden
    ``counters.bin`` (u32) + ``regmap_status.json`` for the 19 golden scenarios."""
    description = "Run the Python golden model and write per-scenario golden counters."
    consumes    = ["hist_source"]
    produces    = {"sim_dir": Path("results/sim")}
    params      = {}

    def run(self, config: BuildConfig, **_) -> dict:
        sim_dir = config.root_dir / "results" / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)

        # 1. Golden self-check (all 12 common scenarios + a fresh recovery).
        ok, lines = verify_all()
        if not ok:
            raise RuntimeError("Golden model self-check failed:\n" + "\n".join(lines))

        # 2. Per-scenario golden evidence for the 19 golden scenarios.
        summary = ["=== Histogram golden-model evidence (per scenario) ==="]
        for name in GOLDEN_SCENARIOS:
            commands, _expect = _scenario_spec(name)
            sim = Simulation()
            clk = Clock(freq=100e6)
            tb = run_scenario(sim, clk, commands)

            scen_dir = sim_dir / name
            scen_dir.mkdir(parents=True, exist_ok=True)
            if tb.counters is not None:
                out = np.asarray(tb.counters, dtype=np.uint32)
            else:
                out = np.array([], dtype=np.uint32)
            out.tofile(scen_dir / "counters.bin")

            status = {
                "halted": int(tb.halted) if tb.halted is not None else 0,
                "error":  int(tb.error)  if tb.error  is not None else int(HistError.NO_ERROR),
                "tx_id":  int(tb.tx_id_status) if tb.tx_id_status is not None else 0,
            }
            (scen_dir / "regmap_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            summary.append(f"{name}: counters={out.tolist()} status={status}")

        (sim_dir / "py_model.log").write_text("\n".join(summary) + "\n", encoding="utf-8")
        return {"sim_dir": sim_dir}


@dataclass(kw_only=True)
class CSimStep(BuildStep):
    """Functional C-sim (21-activation testbench) -> data/<name>/."""
    description = "Invoke Vitis HLS C-simulation (functional, 21 scenarios)."
    consumes    = ["hist_source", "include_dir"]
    produces    = {"csim_data_dir": Path("data"), "csim_log": Path("results/csim.log")}
    params      = {"live_output": False, "clk_freq": 100e6}

    def run(self, config: BuildConfig, live_output, clk_freq, **_) -> dict:
        for name in ALL_TB_SCENARIOS:
            (config.root_dir / "data" / name).mkdir(parents=True, exist_ok=True)

        vitis_env = {"WAVEFLOW_HIST_TB": "func", "WAVEFLOW_HIST_COSIM": "0",
                     "WAVEFLOW_HIST_CSIM_ONLY": "1", "WAVEFLOW_HIST_TRACE_LEVEL": "none",
                     "WAVEFLOW_HIST_CLK_PERIOD_NS": f"{1e9 / clk_freq:g}"}
        _run_vitis(config, vitis_env, live_output, config.root_dir / "results" / "csim.log")
        return {"csim_data_dir": config.root_dir / "data",
                "csim_log": config.root_dir / "results" / "csim.log"}


@dataclass(kw_only=True)
class ValidateCSimStep(BuildStep):
    """Compare golden vs C-sim (19 scenarios, byte-for-bit + status) and check the
    two C-sim-only framing errors report their codes with no counters."""
    description = "Compare golden model vs C-sim counters and status registers."
    consumes    = ["sim_dir", "csim_data_dir"]
    produces    = {"verify_csim": Path("results/verify_csim.json")}
    params      = {}

    def run(self, config: BuildConfig, sim_dir, csim_data_dir, **_) -> dict:
        sim_dir = Path(sim_dir)
        csim_dir = Path(csim_data_dir)
        report: dict = {"common": {}, "csim_only": {}, "all_passed": True}

        for name in GOLDEN_SCENARIOS:
            g_out, g_status = _read_evidence(sim_dir, name)
            a_out, a_status = _read_evidence(csim_dir, name)
            out_match = (g_out.shape == a_out.shape and
                         np.array_equal(g_out.view(np.uint32), a_out.view(np.uint32)))
            status_match = (g_status.get("halted") == a_status.get("halted") and
                            g_status.get("error")  == a_status.get("error") and
                            g_status.get("tx_id")  == a_status.get("tx_id"))
            ok = out_match and status_match
            report["all_passed"] = report["all_passed"] and ok
            report["common"][name] = {
                "outputs_match": bool(out_match), "status_match": bool(status_match),
                "golden_counters": g_out.tolist(), "csim_counters": a_out.tolist(),
                "golden_status": g_status, "csim_status": a_status, "pass": bool(ok),
            }

        for name, expected_err in CSIM_ONLY_SCENARIOS.items():
            a_out, a_status = _read_evidence(csim_dir, name)
            ok = (a_out.size == 0 and a_status.get("error") == expected_err
                  and a_status.get("halted") == 1)
            report["all_passed"] = report["all_passed"] and ok
            report["csim_only"][name] = {
                "expected_error": expected_err, "csim_counters": a_out.tolist(),
                "csim_status": a_status,
                "golden": "not exercised (SimPy read cannot surface TLAST framing)",
                "pass": bool(ok),
            }

        report_path = config.root_dir / "results" / "verify_csim.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["all_passed"]:
            raise RuntimeError(f"C-sim validation FAILED: see {report_path}")
        return {"verify_csim": report_path}


@dataclass(kw_only=True)
class CoSimStep(BuildStep):
    """Functional RTL co-sim (21-activation testbench) -> results/vitis/cosim/<name>/,
    plus the co-sim report and C-synth report copied into results/vitis/."""
    description = "Invoke Vitis HLS RTL co-simulation (functional, 21 scenarios)."
    consumes    = ["hist_source", "include_dir", "csim_data_dir"]
    produces    = {"cosim_dir": Path("results/vitis/cosim"),
                   "cosim_log": Path("results/vitis/cosim.log")}
    params      = {"live_output": False, "clk_freq": 100e6}

    def run(self, config: BuildConfig, live_output, clk_freq, **_) -> dict:
        for name in ALL_TB_SCENARIOS:
            (config.root_dir / "results" / "vitis" / "cosim" / name).mkdir(parents=True, exist_ok=True)

        vitis_env = {"WAVEFLOW_HIST_TB": "func", "WAVEFLOW_HIST_COSIM": "1",
                     "WAVEFLOW_HIST_CSIM_ONLY": "0", "WAVEFLOW_HIST_TRACE_LEVEL": "none",
                     "WAVEFLOW_HIST_CLK_PERIOD_NS": f"{1e9 / clk_freq:g}"}
        _run_vitis(config, vitis_env, live_output, config.root_dir / "results" / "vitis" / "cosim.log")

        proj = config.root_dir / "waveflow_hist_proj"
        _copy_latest(proj, "**/hist_cosim.rpt", config.root_dir / "results" / "vitis" / "hist_cosim.rpt")
        _copy_latest(proj, "**/csynth.rpt", config.root_dir / "results" / "vitis" / "csynth.rpt")
        return {"cosim_dir": config.root_dir / "results" / "vitis" / "cosim",
                "cosim_log": config.root_dir / "results" / "vitis" / "cosim.log"}


@dataclass(kw_only=True)
class ValidateCoSimStep(BuildStep):
    """Compare C-sim vs RTL co-sim for all 21 scenarios (counters + status)."""
    description = "Compare C-sim vs RTL co-sim counters and status registers."
    consumes    = ["csim_data_dir", "cosim_dir"]
    produces    = {"verify_cosim": Path("results/verify_cosim.json")}
    params      = {}

    def run(self, config: BuildConfig, csim_data_dir, cosim_dir, **_) -> dict:
        csim_dir = Path(csim_data_dir)
        cosim_dir = Path(cosim_dir)
        report: dict = {"scenarios": {}, "all_passed": True}

        for name in ALL_TB_SCENARIOS:
            c_out, c_status = _read_evidence(csim_dir, name)
            o_out, o_status = _read_evidence(cosim_dir, name)
            out_match = (c_out.shape == o_out.shape and
                         np.array_equal(c_out.view(np.uint32), o_out.view(np.uint32)))
            status_match = (c_status.get("halted") == o_status.get("halted") and
                            c_status.get("error")  == o_status.get("error") and
                            c_status.get("tx_id")  == o_status.get("tx_id"))
            ok = out_match and status_match
            report["all_passed"] = report["all_passed"] and ok
            report["scenarios"][name] = {
                "outputs_match": bool(out_match), "status_match": bool(status_match),
                "csim_counters": c_out.tolist(), "cosim_counters": o_out.tolist(),
                "csim_status": c_status, "cosim_status": o_status, "pass": bool(ok),
            }

        report_path = config.root_dir / "results" / "verify_cosim.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["all_passed"]:
            raise RuntimeError(f"Co-sim validation FAILED: see {report_path}")
        return {"verify_cosim": report_path}


@dataclass(kw_only=True)
class InspectSynthStep(BuildStep):
    description = "Parse the Vitis HLS C-synthesis report and write results/loop_df.csv."
    consumes    = ["cosim_dir"]
    produces    = {"loop_df": Path("results/loop_df.csv")}
    params      = {}

    def run(self, config: BuildConfig, **_) -> dict:
        try:
            from waveflow.utils.csynthparse import CsynthParser
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"csynthparse not available: {exc}")

        report_dir = config.root_dir / "waveflow_hist_proj" / "solution1"
        if not report_dir.exists():
            raise RuntimeError(f"Solution directory not found: {report_dir}")

        parser = CsynthParser(sol_path=str(report_dir))
        parser.get_loop_pipeline_info()
        parser.get_resources()

        if parser.loop_df.empty:
            print("No loop pipeline information found in csynth.xml.")
        else:
            print(parser.loop_df.to_string())
            non_unit_ii = parser.loop_df[
                parser.loop_df["PipelineII"].apply(
                    lambda v: isinstance(v, (int, np.integer)) and v > 1)]
            if not non_unit_ii.empty:
                print(non_unit_ii.to_string())
                raise RuntimeError("Vitis synthesis produced loops with PipelineII > 1.")
            print("All reported loops have PipelineII <= 1.")

        if parser.res_df.empty:
            print("No resource information found in csynth.xml.")
        else:
            print(parser.res_df.to_string())

        loop_df_path = config.root_dir / "results" / "loop_df.csv"
        loop_df_path.parent.mkdir(parents=True, exist_ok=True)
        parser.loop_df.to_csv(loop_df_path, index=False)
        return {"loop_df": loop_df_path}


@dataclass(kw_only=True)
class ThroughputStep(BuildStep):
    """Measure throughput from co-simulation by difference.

    Run the throughput testbench (one activation CONFIG(16,0,16)->DATA(N)->END)
    with N=1 and N=128, read the co-sim "Total Execution Time" each time, and take
    the marginal per-sample cost:

        throughput = (128 - 1) / (cycles(N=128) - cycles(N=1))   samples/clock

    which cancels the fixed per-activation overhead and isolates the binning
    loop's initiation interval.  Target: II = 1 -> 1 sample/clock (prompt)."""
    description = "Measure the per-sample binning throughput from RTL co-simulation."
    consumes    = ["hist_source", "include_dir"]
    produces    = {"throughput": Path("results/vitis/throughput.json")}
    params      = {"live_output": False, "clk_freq": 100e6}

    def _run_one(self, config, live_output, clk_freq, n: int) -> dict:
        out_dir = config.root_dir / "results" / "vitis" / f"tp_{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tp_nsamp.txt").write_text(f"{n}\n", encoding="utf-8")

        vitis_env = {"WAVEFLOW_HIST_TB": "tp", "WAVEFLOW_HIST_COSIM": "1",
                     "WAVEFLOW_HIST_CSIM_ONLY": "0",
                     "WAVEFLOW_HIST_TP_NSAMP": str(n),
                     "WAVEFLOW_HIST_TRACE_LEVEL": "none",
                     "WAVEFLOW_HIST_CLK_PERIOD_NS": f"{1e9 / clk_freq:g}"}
        _run_vitis(config, vitis_env, live_output, out_dir / "cosim.log")
        rpt = _copy_latest(config.root_dir / "waveflow_hist_proj",
                           "**/hist_cosim.rpt", out_dir / "hist_cosim.rpt")
        if rpt is None:
            raise RuntimeError(f"co-sim report not found for tp_{n}")
        cycles = _total_cycles_from_report(rpt)
        return {"n": n, "cycles": cycles, "report": str(rpt)}

    def run(self, config: BuildConfig, live_output, clk_freq, **_) -> dict:
        # Second (larger) burst length.  Default 128 (the prompt's max nsamp);
        # overridable via WAVEFLOW_HIST_TP_N_HI so a smaller burst can be used to
        # cross-check the marginal rate.
        n_hi = int(os.environ.get("WAVEFLOW_HIST_TP_N_HI", "128"))
        t1 = self._run_one(config, live_output, clk_freq, 1)
        thi = self._run_one(config, live_output, clk_freq, n_hi)

        dsamp = n_hi - 1
        dcy = thi["cycles"] - t1["cycles"]
        if dcy <= 0:
            raise RuntimeError(
                f"non-positive marginal cycle count (T{n_hi}={thi['cycles']}, T1={t1['cycles']}); "
                f"co-sim may not have differentiated the two runs")
        marginal_cycles_per_sample = dcy / dsamp
        throughput = dsamp / dcy  # samples/clock

        target = 1.0  # II = 1 -> one sample per clock
        ok = marginal_cycles_per_sample <= 1.0 + 1e-9

        report = {
            "method": (f"delta method: ({n_hi}-1)/(cycles(N={n_hi})-cycles(N=1)) "
                       f"samples/clock"),
            "target": {"ii": 1, "samples_per_clock": target,
                       "source": "prompt.md Requirements (II = 1, one sample per clock)"},
            "runs": {"N1": t1, f"N{n_hi}": thi},
            "marginal_cycles_per_sample": marginal_cycles_per_sample,
            "throughput_samples_per_clock": throughput,
            "pass": bool(ok),
        }
        report_path = config.root_dir / "results" / "vitis" / "throughput.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not ok:
            raise RuntimeError(
                f"throughput FAIL: {marginal_cycles_per_sample:.4f} cycles/sample "
                f"({throughput:.4f} samples/clock) exceeds the II=1 target")
        return {"throughput": report_path}


def build_hist_dag() -> BuildDag:
    """Build the canonical histogram accelerator pipeline DAG."""
    dag = BuildDag()
    dag.add(SourceStep(
        artifact="hist_source", path=_SOURCE_DIR / "hist.py",
        description="Python source for schemas, accelerator, and testbench.",
    ))
    dag.add(HlsGenIncludeStep(name="gen_include"))

    # Golden model -> per-scenario golden evidence (self-checked).
    dag.add(PySimStep(name="py_sim"))

    # C-sim functional (21 scenarios) -> golden vs C-sim validation.
    dag.add(CSimStep(name="csim"))
    dag.add(ValidateCSimStep(name="validate_csim"))

    # RTL co-sim functional (21 scenarios) -> C-sim vs co-sim validation.
    dag.add(CoSimStep(name="cosim"))
    dag.add(ValidateCoSimStep(name="validate_cosim"))

    # C-synth inspection (II <= 1, resources) from the functional co-sim's csynth.
    dag.add(InspectSynthStep(name="inspect_synth"))

    # Throughput from co-simulation (delta method, N=1 vs N=128).
    dag.add(ThroughputStep(name="throughput"))
    return dag


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the histogram accelerator build/verify DAG.")
    parser.add_argument(
        "--through", metavar="STEP", default="throughput",
        choices=["gen_include", "py_sim", "csim", "validate_csim",
                 "cosim", "validate_cosim", "inspect_synth", "throughput"],
        help="Run the DAG up to and including this step.")
    parser.add_argument("--list-steps", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--clk-freq", type=float, default=100e6, metavar="HZ")
    parser.add_argument("--live-output", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-step", metavar="STEP", action="append", default=[])
    args = parser.parse_args()

    dag = build_hist_dag()

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
            print(f"  {entry['artifact']:<16} {entry['produced_by']:<18} {mark}  {age}")
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
