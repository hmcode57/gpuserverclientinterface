"""Example handler with no dependencies: reads the job's inputs and writes a summary. Use it to test the
whole path (laptop -> tunnel -> notebook -> worker -> back) before plugging in a real model.

  python3 laptop/submit.py --kind echo --family echo --params '{"seconds": 5, "text": "hi"}' --file in.txt=README.md

params:
  seconds  how long to "work" (default 1), with progress reported in 10 steps
  text     echoed back in result.json
  fail     true = raise, to test failure reporting and the restart loop
"""
import hashlib, json, time

from runtime import Handler


class Echo(Handler):
    def run(self, job, ctx):
        p = ctx.params
        if p.get("fail"):
            raise RuntimeError("echo handler asked to fail (params.fail)")
        inputs = {}
        for name in sorted(job["files"]):
            if name in job["outputs"]:
                continue
            data = ctx.input(name)
            inputs[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        seconds = float(p.get("seconds", 1))
        for i in range(10):
            time.sleep(seconds / 10)
            ctx.progress(step=i + 1, steps=10)
        result = {"job": job["id"], "kind": job["kind"], "text": p.get("text"), "inputs": inputs,
                  "config": ctx.config(), "device": self.device}
        ctx.output("result.json", json.dumps(result, indent=2))
        return {"inputs": len(inputs)}
