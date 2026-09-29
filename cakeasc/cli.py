"""CLI: demo / verify / codegen / corpus / guide / state."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from . import workloads as wl
from .agent import Evolution, EvolutionOpts
from .arch import get_arch
from .codegen import generate_ascendc
from .costmodel import Calibration
from .distiller import distill
from .llm import IR_GUIDE, LLMClient
from .rules import Registry

STATE_FILE = "compiler_state.json"


def load_state(path=STATE_FILE):
    if not os.path.exists(path):
        return Registry.baseline(), Calibration()
    data = json.load(open(path, encoding="utf-8"))
    return (Registry.from_json(data.get("rules")),
            Calibration.from_json(data.get("calibration")))


def save_state(registry, calibration, path=STATE_FILE):
    json.dump({"rules": registry.to_json(),
               "calibration": calibration.to_json()},
              open(path, "w", encoding="utf-8"), indent=2)


# ---------------------------------------------------------------- commands


def cmd_demo(args):
    arch = get_arch(args.target)
    registry, calibration = (load_state(args.state) if args.use_state
                             else (Registry.baseline(), Calibration()))
    workload = wl.get(args.workload)
    opts = EvolutionOpts(seed=args.seed, propose_n=args.propose,
                         eval_topk=args.topk, distill_threshold=args.threshold,
                         apply_evolution=args.apply, use_llm=args.llm,
                         verbose=True)
    evo = Evolution(workload, arch, registry, calibration, opts=opts)
    summary = evo.run(args.iters)

    print("\n--- evolution summary ---")
    for k in ("workload", "iterations", "evaluated", "correct_count",
              "speedup_vs_naive", "best_params", "best_note"):
        print(f"  {k}: {summary[k]}")
    if summary["distilled_rules"]:
        print(f"  distilled rules: {', '.join(summary['distilled_rules'])}")
    print(f"  evidence families: {summary['evidence']}")

    if evo.best is not None:
        src = generate_ascendc(evo.best.programs, arch)
        out = args.out or f"{workload.name}_best.cpp"
        open(out, "w", encoding="utf-8").write(src)
        print(f"\nbest candidate lowered to AscendC: {out} "
              f"({len(src.splitlines())} lines)")

    if args.apply:
        save_state(registry, calibration, args.state)
        distilled = [r for r in registry.rules if r.rule_id.startswith("DISTILLED.")]
        cal_arches = sorted(calibration.per_arch)
        print(f"compiler state persisted to {args.state} "
              f"(human merge gate: {len(distilled)} distilled rule(s) "
              f"{[r.rule_id for r in distilled]}, calibration for {cal_arches})")

    if args.generalize:
        print("\n--- generalization stage (paper Sec. 6) ---")
        from .dispatcher import build_portfolio
        build_portfolio(evo, log=print)
    return 0


def _load_schedule(path):
    ns = {}
    src = open(path, encoding="utf-8").read()
    exec(compile(src, path, "exec"), ns)
    for v in ns.values():
        if callable(v) and hasattr(v, "build_all_cores"):
            return v
    if "kern" in ns:
        return ns["kern"]
    raise SystemExit(f"{path}: expected a schedule (`def kern(m)` or @asc.schedule)")


def _build_from_file(kern, path, block_dim):
    """Honor @asc.schedule metadata when present; else build with block_dim."""
    from . import builder as asc
    name = os.path.basename(path)[:-3]
    if hasattr(kern, "build_all_cores"):
        return kern.build_all_cores()
    return asc.build_all_cores(kern, name=name, block_dim=block_dim)


def cmd_verify(args):
    from .verifier import verify_cores
    from .costmodel import predict_cores
    from .diagnostics import GATE
    from .ir import IRConstructionError

    arch = get_arch(args.target)
    registry, cal = load_state(args.state)
    kern = _load_schedule(args.file)
    try:
        progs = _build_from_file(kern, args.file, args.block_dim)
    except IRConstructionError as e:
        print(f"[gate:ir_construction] IR.construction_error @ {e.region}: {e.message}")
        print(f"    repair: {e.hint}")
        return 1
    findings = verify_cores(progs, arch, registry)
    gates = [f for f in findings if f.severity == GATE]
    for f in findings:
        print(f.format())
    if gates:
        print(f"\nREJECTED by {len(gates)} gate finding(s)")
        return 1
    rep = predict_cores(progs, arch, cal)
    print()
    print(rep.format())
    print("\nACCEPTED (pre-compile gates clean)")
    return 0


def cmd_codegen(args):
    arch = get_arch(args.target)
    kern = _load_schedule(args.file)
    progs = _build_from_file(kern, args.file, args.block_dim)
    src = generate_ascendc(progs, arch)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(src)
        print(f"wrote {args.out} ({len(src.splitlines())} lines)")
    else:
        sys.stdout.write(src)
    return 0


def cmd_corpus(args):
    from .corpus import CorpusGate
    gate = CorpusGate(get_arch(args.target))
    ok_all = True
    print("valid corpus (must stay clean):")
    for name, progs in gate.valid_corpus():
        print(f"  [ok] {name}: {len(progs[0].ops)} ops (core 0)")
    print("\nrule templates vs fixtures:")
    from .rules import TEMPLATES
    for template in sorted(TEMPLATES):
        ok, detail = gate.test_rule_template(template)
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {template}: {detail}")
    print("\ncorpus gate:", "ALL PASS" if ok_all else "FAILURES")
    return 0 if ok_all else 1


def cmd_guide(_args):
    print(IR_GUIDE)
    return 0


def cmd_state(args):
    registry, cal = load_state(args.state)
    print(f"state file: {args.state}")
    print("active rules:")
    for r in registry.rules:
        prov = r.provenance.get("family", "baseline")
        print(f"  {r.rule_id:<40} {r.category:<22} {prov}")
    print("calibration:")
    print(json.dumps(cal.to_json(), indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="cakeasc",
        description="CAKE-Ascend: compiler-agent co-design for Ascend kernel evolution "
                    "(implementation of arXiv:2608.12629)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the full kernel+compiler evolution loop")
    d.add_argument("workload", choices=wl.all_names())
    d.add_argument("--iters", type=int, default=12)
    d.add_argument("--propose", type=int, default=6)
    d.add_argument("--topk", type=int, default=4)
    d.add_argument("--threshold", type=int, default=3,
                   help="evidence recurrence threshold for distillation")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--target", default="ascend910b")
    d.add_argument("--llm", action="store_true",
                   help="use a real LLM proposer if configured "
                        "(CAKEASC_LLM_BASE_URL/API_KEY/MODEL)")
    d.add_argument("--apply", action="store_true",
                   help="accept distilled compiler changes (human merge gate)")
    d.add_argument("--use-state", action="store_true",
                   help="load evolved rules/calibration from the state file")
    d.add_argument("--state", default=STATE_FILE)
    d.add_argument("--no-generalize", dest="generalize", action="store_false")
    d.add_argument("--out", default=None, help="output path for best-candidate AscendC")
    d.set_defaults(func=cmd_demo)

    v = sub.add_parser("verify", help="verify + cost-report a schedule file")
    v.add_argument("file")
    v.add_argument("--block-dim", type=int, default=1, dest="block_dim")
    v.add_argument("--target", default="ascend910b")
    v.add_argument("--state", default=STATE_FILE)
    v.set_defaults(func=cmd_verify)

    c = sub.add_parser("codegen", help="lower a schedule file to AscendC")
    c.add_argument("file")
    c.add_argument("-o", "--out", default=None)
    c.add_argument("--block-dim", type=int, default=1, dest="block_dim")
    c.add_argument("--target", default="ascend910b")
    c.set_defaults(func=cmd_codegen)

    k = sub.add_parser("corpus", help="run the corpus test gate suite")
    k.add_argument("--target", default="ascend910b")
    k.set_defaults(func=cmd_corpus)

    g = sub.add_parser("guide", help="print the schedule IR reference")
    g.set_defaults(func=cmd_guide)

    s = sub.add_parser("state", help="show evolved rules + calibration")
    s.add_argument("--state", default=STATE_FILE)
    s.set_defaults(func=cmd_state)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
