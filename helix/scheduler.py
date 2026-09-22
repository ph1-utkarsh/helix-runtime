"""Owned paged-KV allocator and deterministic serving-policy simulator."""
from dataclasses import dataclass
import math


class OutOfPages(RuntimeError): pass


class PagedKV:
    def __init__(self, pages, page_size=16):
        if pages <= 0 or page_size <= 0: raise ValueError("positive capacity required")
        self.page_size, self.free = page_size, list(range(pages)); self.tables = {}; self.refs = {}
    def reserve(self, sequence, tokens):
        needed = math.ceil(tokens / self.page_size)
        current = len(self.tables.get(sequence, []))
        if needed > current:
            count = needed - current
            if count > len(self.free): raise OutOfPages
            pages = [self.free.pop() for _ in range(count)]
            self.tables.setdefault(sequence, []).extend(pages)
            for page in pages: self.refs[page] = 1
        return tuple(self.tables.get(sequence, []))
    def fork(self, source, target):
        if target in self.tables: raise ValueError("target exists")
        self.tables[target] = list(self.tables[source])
        for page in self.tables[target]: self.refs[page] += 1
    def release(self, sequence):
        for page in self.tables.pop(sequence, []):
            self.refs[page] -= 1
            if self.refs[page] == 0: del self.refs[page]; self.free.append(page)
    @property
    def used_pages(self): return len(self.refs)


@dataclass
class Request:
    name: str; arrival: int; prompt: int; output: int


def simulate(requests, policy, budget=64, chunk=32, max_ticks=10000):
    """Token-work simulator; one prefill token and one decode token each cost one unit."""
    states = {r.name: {"r": r, "prefill": 0, "decode": 0, "first": None, "times": []} for r in requests}
    tick = 0
    while tick < max_ticks and any(s["decode"] < s["r"].output for s in states.values()):
        ready = [s for s in states.values() if s["r"].arrival <= tick and s["decode"] < s["r"].output]
        remaining = budget
        if policy == "adaptive":
            # Decode first prevents established streams from stalling.
            for s in sorted(ready, key=lambda x: x["times"][-1] if x["times"] else -1):
                if remaining and s["prefill"] == s["r"].prompt:
                    s["decode"] += 1; s["times"].append(tick); remaining -= 1
                    if s["first"] is None: s["first"] = tick
            for s in sorted(ready, key=lambda x: (x["r"].arrival, x["r"].prompt)):
                take = min(chunk, s["r"].prompt - s["prefill"], remaining)
                s["prefill"] += take; remaining -= take
        elif policy == "static":
            # Fixed FCFS batches finish each prompt before admitting useful decode work.
            for s in sorted(ready, key=lambda x: x["r"].arrival):
                if not remaining: break
                if s["prefill"] < s["r"].prompt:
                    take = min(s["r"].prompt - s["prefill"], remaining)
                    s["prefill"] += take; remaining -= take
                elif remaining:
                    # Autoregressive dependence permits one token per request per tick.
                    take = 1
                    s["times"].append(tick); s["decode"] += take; remaining -= take
                    if s["first"] is None: s["first"] = tick
        else: raise ValueError("unknown policy")
        tick += 1
    if any(s["decode"] < s["r"].output for s in states.values()): raise RuntimeError("simulation did not finish")
    rows = []
    for s in states.values():
        gaps = [b-a for a,b in zip(s["times"], s["times"][1:])]
        rows.append({"name": s["r"].name, "ttft": s["first"]-s["r"].arrival,
                     "tpot": sum(gaps)/len(gaps) if gaps else 0})
    return {"ticks": tick, "output_tokens": sum(r.output for r in requests), "tokens_per_tick": sum(r.output for r in requests)/tick, "requests": rows}
