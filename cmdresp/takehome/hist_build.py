"""hist_build.py — build/verify DAG for the histogram accelerator (take-home).

Adapts ``examples/stream_inband/poly_build.py`` to the command-response
histogram kernel in ``hist.py``.  The five-group narrative is the same as poly:
Python golden model  ->  HLS codegen  ->  C-sim functional verify  ->
C-synth resource estimate  ->  RTL cosim timing.

Differences from poly (the cmdresp pattern is authoritative):

* There is **no** regmap data field.  Configuration (``nbin`` / ``lo`` / ``hi``)
  travels on the input stream as a **CONFIG command**, and the accumulating
  counters are kernel-internal state (``add_state``) emitted by END.  The regmap
  therefore carries only the status registers ``halted`` / ``error`` / ``tx_id``.
* The command sequence is **CONFIG -> DATA -> END** (poly is DATA -> END), so a
  clean activation drives two data-bearing commands and reads back three
  response headers (one per command) plus the ``nbin`` counters on END.
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
from waveflow.build.hwcodegen_steps import HlsCodegenStep
from waveflow.build.streamutils import StreamUtilsStep
from waveflow.build.verify_steps import FunctionalVerifyStep
from waveflow.hw.arrayutils import ArrayUtilsStep, read_uint32_file, write_uint32_file
from waveflow.hw.clock import Clock
from waveflow.hw.dataschema import DataSchemaStep
from waveflow.hw.hw_testbench import SeqTB
from waveflow.simulation.logger import Logger
from waveflow.simulation.simulation import Simulation
from waveflow.toolchain import toolchain

try:
    from cmdresp.takehome.hist import (
        CfgPayload, CounterArray, Float32, HistAccel, HistCmdHdr, HistCmdType,
        HistError, HistRespHdr, HistTB, HistTBHls, SampArr, U32,
        SCHEMA_CLASSES, WORD_BW_SUPPORTED,
        connect,
    )
except ModuleNotFoundError:
    from hist import (  # type: ignore[no-redef]  # direct execution
        CfgPayload, CounterArray, Float32, HistAccel, HistCmdHdr, HistCmdType,
        HistError, HistRespHdr, HistTB, HistTBHls, SampArr, U32,
        SCHEMA_CLASSES, WORD_BW_SUPPORTED,
        connect,
    )


_SOURCE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# The one clean activation used by the C-sim / golden cross-check.  Fixed so the
# input files and the golden model run the exact same transaction.
# ---------------------------------------------------------------------------
NCFG_TX   = 1
NCFG_NBIN = 4
NCFG_LO   = 0.0
NCFG_HI   = 4.0
NDATA_TX  = 2
SAMPLES   = [0.2, 1.8, 2.1, 3.9, 0.0, 3.2]   # -> counters [2, 1, 1, 2]
NEND_TX   = 3
NSAMP     = len(SAMPLES)


@dataclass(kw_only=True)
class BuildInputsStep(BuildStep):
    description = "Write the CONFIG/DATA/END command headers, the config payload, and the samples."
    consumes    = ["hist_source"]
    produces    = {
        "config_cmd_hdr": Path("data/config_cmd_hdr.bin"),
        "config_payload": Path("data/config_payload.bin"),
        "data_cmd_hdr":   Path("data/data_cmd_hdr.bin"),
        "samp_in":        Path("data/samp_in_data.bin"),
        "end_cmd_hdr":    Path("data/end_cmd_hdr.bin"),
        "data_dir":       Path("data"),
    }
    params      = {}

    def run(self, config: BuildConfig, **_) -> dict:
        out_dir = config.root_dir / "data"
        out_dir.mkdir(parents=True, exist_ok=True)

        config_hdr = HistCmdHdr()
        config_hdr.cmd_type = HistCmdType.CONFIG
        config_hdr.tx_id    = NCFG_TX
        config_hdr.nsamp    = 3                      # CfgPayload = 3 words (nbin, lo, hi)
        config_hdr.write_uint32_file(out_dir / "config_cmd_hdr.bin")

        config_payload = CfgPayload()
        config_payload.nbin = NCFG_NBIN
        config_payload.lo   = float(np.float32(NCFG_LO))
        config_payload.hi   = float(np.float32(NCFG_HI))
        config_payload.write_uint32_file(out_dir / "config_payload.bin")

        data_hdr = HistCmdHdr()
        data_hdr.cmd_type = HistCmdType.DATA
        data_hdr.tx_id    = NDATA_TX
        data_hdr.nsamp    = NSAMP
        data_hdr.write_uint32_file(out_dir / "data_cmd_hdr.bin")

        samp_in = np.asarray(SAMPLES, dtype=np.float32)
        write_uint32_file(samp_in, elem_type=Float32,
                          file_path=out_dir / "samp_in_data.bin", nwrite=NSAMP)

        end_hdr = HistCmdHdr()
        end_hdr.cmd_type = HistCmdType.END
        end_hdr.tx_id    = NEND_TX
        end_hdr.nsamp    = 0
        end_hdr.write_uint32_file(out_dir / "end_cmd_hdr.bin")

        return {
            "config_cmd_hdr": out_dir / "config_cmd_hdr.bin",
            "config_payload": out_dir / "config_payload.bin",
            "data_cmd_hdr":   out_dir / "data_cmd_hdr.bin",
            "samp_in":        out_dir / "samp_in_data.bin",
            "end_cmd_hdr":    out_dir / "end_cmd_hdr.bin",
            "data_dir":       out_dir,
        }


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
    """Run the Python golden model on the fixed activation and write the golden
    response headers + counters + status that the C-sim is checked against."""
    description = "Run the Python SimPy golden model and write the golden outputs."
    consumes    = ["hist_source", "config_cmd_hdr", "config_payload",
                   "data_cmd_hdr", "samp_in", "end_cmd_hdr"]
    produces    = {
        "sim_dir":        Path("results/sim"),
        "resp_cfg":       Path("results/sim/resp_cfg.bin"),
        "resp_data":      Path("results/sim/resp_data.bin"),
        "resp_end":       Path("results/sim/resp_end.bin"),
        "counters":       Path("results/sim/counters.bin"),
        "status":         Path("results/sim/regmap_status.json"),
        "log":            Path("results/sim_log.csv"),
    }
    params      = {"clk_freq": 100e6, "in_bw": 32, "out_bw": 32,
                   "unroll_factor": 1, "log_file": "results/sim_log.csv"}

    def expected_paths(self, config: BuildConfig) -> dict[str, Path]:
        log_file = config.params.get("log_file", self.params["log_file"])
        return {"log": config.root_dir / log_file}

    def run(self, config: BuildConfig,
            config_cmd_hdr, config_payload, data_cmd_hdr, samp_in, end_cmd_hdr,
            clk_freq, in_bw, out_bw, unroll_factor, log_file,
            **_) -> dict:
        # Read the exact inputs the C-sim will consume, so golden and actual
        # agree on the transaction.
        cfg_hdr = HistCmdHdr().read_uint32_file(config_cmd_hdr)
        cfg     = CfgPayload().read_uint32_file(config_payload)
        dhdr    = HistCmdHdr().read_uint32_file(data_cmd_hdr)
        samp    = np.array(read_uint32_file(samp_in, elem_type=Float32,
                                            shape=int(dhdr.nsamp)),
                           dtype=np.float32)

        commands = [
            {"type": "config", "tx_id": int(cfg_hdr.tx_id),
             "nbin": int(cfg.nbin), "lo": float(cfg.lo), "hi": float(cfg.hi)},
            {"type": "data",   "tx_id": int(dhdr.tx_id),
             "nsamp": int(dhdr.nsamp), "samples": samp},
            {"type": "end",    "tx_id": int(cfg_hdr.tx_id),  # placeholder, see below
             "nbin": int(cfg.nbin)},
        ]
        end_hdr = HistCmdHdr().read_uint32_file(end_cmd_hdr)
        commands[2]["tx_id"] = int(end_hdr.tx_id)

        sim = Simulation()
        clk = Clock(freq=clk_freq)
        log_path = config.root_dir / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = Logger(name="hist_log", sim=sim, file_path=log_path,
                        fields=["event", "job"])
        accel = HistAccel(
            name="hist_accel", sim=sim,
            in_bw=in_bw, out_bw=out_bw, unroll_factor=unroll_factor,
            clk=clk, logger=logger,
        )
        tb = HistTB(name="hist_tb", sim=sim, commands=commands, word_bw=in_bw)
        connect(sim, tb, accel, clk)
        sim.run_sim()

        sim_dir = config.root_dir / "results" / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)
        # One response header per command (config, data, end), in order.
        tb.resp_hdrs[0].write_uint32_file(sim_dir / "resp_cfg.bin")
        tb.resp_hdrs[1].write_uint32_file(sim_dir / "resp_data.bin")
        tb.resp_hdrs[2].write_uint32_file(sim_dir / "resp_end.bin")
        counters = np.asarray(tb.counters, dtype=np.uint32)
        write_uint32_file(counters, elem_type=U32,
                          file_path=sim_dir / "counters.bin", nwrite=counters.size)
        status = {
            "halted": int(tb.halted) if tb.halted is not None else 0,
            "error":  int(tb.error)  if tb.error  is not None else int(HistError.NO_ERROR),
            "tx_id":  int(tb.tx_id_status) if tb.tx_id_status is not None else 0,
        }
        (sim_dir / "regmap_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        return {
            "sim_dir":    sim_dir,
            "resp_cfg":   sim_dir / "resp_cfg.bin",
            "resp_data":  sim_dir / "resp_data.bin",
            "resp_end":   sim_dir / "resp_end.bin",
            "counters":   sim_dir / "counters.bin",
            "status":     sim_dir / "regmap_status.json",
            "log":        log_path,
        }


@dataclass(kw_only=True)
class CSimStep(BuildStep):
    description = "Invoke Vitis HLS C-simulation."
    consumes    = [
        "hist_cpp", "hist_hpp",
        "hist_configure_impl", "hist_accumulate_impl", "hist_emit_impl",
        "hist_tb", "include_dir", "data_dir",
    ]
    produces    = {"csim_data_dir": "data_dir"}
    params      = {"live_output": False, "clk_freq": 100e6}

    def run(self, config: BuildConfig, include_dir, data_dir, live_output, clk_freq, **_) -> dict:
        vitis_env = {"WAVEFLOW_HIST_COSIM": "0",
                     "WAVEFLOW_HIST_TRACE_LEVEL": "none",
                     "WAVEFLOW_HIST_CLK_PERIOD_NS": f"{1e9 / clk_freq:g}"}
        try:
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
        except Exception as exc:
            raise RuntimeError(str(exc))
        return {"csim_data_dir": data_dir}


@dataclass(kw_only=True)
class CSynthStep(BuildStep):
    description = "Run Vitis HLS C-synthesis and RTL co-simulation."
    consumes    = [
        "hist_cpp", "hist_hpp",
        "hist_configure_impl", "hist_accumulate_impl", "hist_emit_impl",
        "include_dir", "csim_data_dir",
    ]
    produces    = {"report_dir": Path("waveflow_hist_proj/solution1")}
    params      = {"live_output": False, "clk_freq": 100e6}

    def run(self, config: BuildConfig, include_dir, csim_data_dir, live_output, clk_freq, **_) -> dict:
        vitis_env = {"WAVEFLOW_HIST_COSIM": "1",
                     "WAVEFLOW_HIST_TRACE_LEVEL": "none",
                     "WAVEFLOW_HIST_CLK_PERIOD_NS": f"{1e9 / clk_freq:g}"}
        try:
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
        except Exception as exc:
            raise RuntimeError(str(exc))
        report_dir = config.root_dir / "waveflow_hist_proj" / "solution1"
        return {"report_dir": report_dir}


@dataclass(kw_only=True)
class InspectSynthStep(BuildStep):
    description = "Parse the Vitis HLS C-synthesis report and write results/loop_df.csv."
    consumes    = ["report_dir"]
    produces    = {"loop_df": Path("results/loop_df.csv")}
    params      = {}

    def run(self, config: BuildConfig, report_dir) -> dict:
        try:
            from waveflow.utils.csynthparse import CsynthParser
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"csynthparse not available: {exc}")

        if not report_dir.exists():
            raise RuntimeError(f"Solution directory not found: {report_dir}")

        try:
            parser = CsynthParser(sol_path=str(report_dir))
            parser.get_loop_pipeline_info()
            parser.get_resources()
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(f"Synthesis report parsing failed: {exc}")

        print("\nLatency and Initiation Interval:")
        if parser.loop_df.empty:
            print("No loop pipeline information found in csynth.xml.")
        else:
            print(parser.loop_df.to_string())
            non_unit_ii = parser.loop_df[
                parser.loop_df["PipelineII"].apply(
                    lambda v: isinstance(v, (int, np.integer)) and v > 1
                )
            ]
            if not non_unit_ii.empty:
                print("Loops with PipelineII > 1:")
                print(non_unit_ii.to_string())
                raise RuntimeError("Vitis synthesis produced loops with PipelineII > 1.")
            print("All reported loops have PipelineII <= 1.")

        print("\nResource Usage:")
        if parser.res_df.empty:
            print("No resource information found in csynth.xml.")
        else:
            print(parser.res_df.to_string())

        loop_df_path = config.root_dir / "results" / "loop_df.csv"
        loop_df_path.parent.mkdir(parents=True, exist_ok=True)
        parser.loop_df.to_csv(loop_df_path, index=False)
        return {"loop_df": loop_df_path}


def build_hist_dag() -> BuildDag:
    """Build the canonical histogram accelerator pipeline DAG."""
    dag = BuildDag()
    dag.add(SourceStep(
        artifact="hist_source", path=_SOURCE_DIR / "hist.py",
        description="Python source for schemas, accelerator, and testbench.",
    ))

    """
    Python golden model
    """
    dag.add(BuildInputsStep(name="build_inputs"))
    dag.add(PySimStep(name="py_sim"))

    """
    HLS code generation
    """
    dag.add(HlsGenIncludeStep(name="gen_include"))
    # Kernel codegen — writes gen/hist.{hpp,cpp} and the sticky hand-written
    # hook .tpp files into the source-tree root (impl_dir=".").
    dag.add(HlsCodegenStep(
        name="gen_kernel",
        comp_class=HistAccel,
        source_artifact="hist_source",
        output_dir="gen",
        impl_dir=".",
    ))
    # Testbench codegen — consumes HistTBHls and emits gen/hist_tb.cpp.
    dag.add(HlsCodegenStep(
        name="gen_tb",
        comp_class=HistTBHls,
        source_artifact="hist_source",
        output_dir="gen",
        is_testbench=True,
    ))

    """
    C-sim functional verification
    """
    dag.add(CSimStep(name="csim"))
    dag.add(FunctionalVerifyStep(
        name="validate_csim",
        golden_dir_artifact="sim_dir",
        actual_dir_artifact="csim_data_dir",
        schemas=[
            {"filename": "resp_cfg_data.bin",  "golden_filename": "resp_cfg.bin",  "schema": HistRespHdr},
            {"filename": "resp_data_data.bin", "golden_filename": "resp_data.bin", "schema": HistRespHdr},
            {"filename": "resp_end_data.bin",  "golden_filename": "resp_end.bin",  "schema": HistRespHdr},
        ],
        arrays=[
            {"filename": "counters_data.bin", "golden_filename": "counters.bin",
             "elem_type": U32, "count": NCFG_NBIN, "rtol": 0.0, "atol": 0.0},
        ],
        jsons=[
            {"filename": "regmap_status.json", "expect_zero": ["halted", "error"]},
        ],
        output_dir="results/vitis",
        output_artifact="vitis_dir",
        report_path="results/verify_csim.json",
    ))

    """
    C-synth resource estimation + RTL cosim
    """
    dag.add(CSynthStep(name="csynth"))
    dag.add(InspectSynthStep(name="inspect_synth"))
    return dag


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the histogram accelerator build/verify DAG.")
    parser.add_argument(
        "--through", metavar="STEP", default="gen_tb",
        help="Run the DAG up to and including this step.",
    )
    parser.add_argument("--list-steps", action="store_true")
    parser.add_argument("--list-artifacts", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--in-bw", type=int, default=32, choices=[32, 64])
    parser.add_argument("--out-bw", type=int, default=32, choices=[32, 64])
    parser.add_argument("--unroll-factor", type=int, default=1)
    parser.add_argument("--clk-freq", type=float, default=100e6, metavar="HZ")
    parser.add_argument("--log", metavar="FILE", default="results/sim_log.csv")
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
        params={
            'clk_freq': args.clk_freq,
            'in_bw': args.in_bw,
            'out_bw': args.out_bw,
            'unroll_factor': args.unroll_factor,
            'log_file': args.log,
            'live_output': args.live_output,
        },
    )

    if args.list_artifacts:
        all_paths = dag.artifact_paths(config)
        for artifact, step_name in dag.artifact_owners().items():
            p = all_paths.get(artifact)
            if p is not None:
                try:
                    display = p.relative_to(config.root_dir)
                except ValueError:
                    display = p
                print(f"  {artifact:<20} {step_name:<26} {display}")
            else:
                print(f"  {artifact:<20} {step_name:<26} (object)")
        return

    if args.status:
        now = _time.time()
        for entry in dag.results_status(config):
            age = f"{(now - entry['mtime']) / 3600:.1f}h ago" if entry['mtime'] else "—"
            exists_mark = "✓" if entry['exists'] else "✗"
            stale_note = (f"  STALE ({', '.join(entry['stale_because'])})"
                          if entry['stale'] else "")
            print(f"  {entry['artifact']:<16} {entry['produced_by']:<22} "
                  f"{exists_mark}  {age:<12}{stale_note}")
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
