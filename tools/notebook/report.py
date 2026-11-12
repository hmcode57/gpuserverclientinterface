"""Summarise a notebook session on the laptop: python3 tools/notebook/report.py [data/reports/<session>]

Reads session.json (uploaded live by tools/notebook/session.py), prints phases, environment and resource peaks,
and writes SUMMARY.md next to it. Works on partial data (session still running or died). Without an argument
it uses the most recent session.
"""
import glob, json, os, sys


def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def main(folder):
    p = os.path.join(folder, "session.json")
    sess = json.load(open(p)) if os.path.exists(p) else {}
    L = [f"# Session {sess.get('session', os.path.basename(folder))}", "",
         f"started {sess.get('started')}, finished {sess.get('finished', 'NOT FINISHED (still running or died)')}", ""]
    ph = sess.get("phases", {})
    L += ["## Phases", "", "| phase | ok | wall s | RAM peak GB | notes |", "| --- | --- | --- | --- | --- |"]
    for k, v in ph.items():
        note = json.dumps(v.get("summary") or v.get("error") or v.get("skipped", ""), default=str)[:160]
        L.append(f"| {k} | {v.get('ok')} | {v.get('wall_s')} | {g(v, 'resources', 'ram_used_peak_gb')} | "
                 f"{note.replace('|', '/')} |")
    pf = ph.get("preflight", {})
    if pf:
        L += ["", "## Environment", "", f"- Python {pf.get('python')}, torch/CUDA: {pf.get('torch')}",
              f"- RAM: {g(pf, 'ram', 'total_gb')} GB total, cgroup limit: {pf.get('cgroup_mem_limit')}",
              f"- CPUs: {pf.get('cpus')}", "- GPUs:", "```", str(pf.get("nvidia_smi")), "```",
              f"- Disk: {json.dumps(pf.get('disk'))}",
              f"- Internet: {pf.get('internet_s', pf.get('internet'))} s, queue round trip: {pf.get('queue_rtt_s')} s",
              f"- Provider hints: {list((pf.get('env_hints') or {}).keys())[:10]}"]
    ins = ph.get("install", {})
    if ins.get("pip_s") is not None:
        L += [f"- pip install: {ins.get('pip_s')} s, torch before/after: {ins.get('torch_before')} / "
              f"{ins.get('torch_after')}" + (f" ({ins['warning']})" if ins.get("warning") else "")]
    for k, v in ph.items():
        if v.get("error"):
            L += ["", f"## {k} failed", "", "```", str(v.get("traceback") or v["error"])[-2000:], "```"]
    L += ["", "## Whole session", "", "```", json.dumps(sess.get("resources_total", {}), indent=1), "```", "",
          "Logs: " + ", ".join(sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.log")))), ""]
    out = "\n".join(L)
    open(os.path.join(folder, "SUMMARY.md"), "w").write(out)
    print(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        d = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/reports/*")),
                   key=os.path.getmtime)
        if not d:
            sys.exit("usage: report.py data/reports/<session>")
        sys.argv.append(d[-1])
    main(sys.argv[1])
