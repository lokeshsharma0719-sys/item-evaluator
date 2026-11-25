"""Scripted fake for the LLM backend. Replaces LLMClient.backend, so the disk cache still sits above it.

Responses are looked up by (role, model, item_id, variant_id), then (role, model, item_id), then (role, item_id),
then a per-role handler. A response is a dict (sent as JSON), a str (sent raw), or a callable(data, ctx).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from item_metrics.llm import BackendResult

_DATA = re.compile(r"<data>\n(.*)\n</data>", re.S)


def extract_data(user: str) -> Any:
    m = _DATA.search(user)
    return json.loads(m.group(1)) if m else None


def label_for_text(data: dict[str, Any], text: str) -> str:
    opts = data.get("options") or data.get("question", {}).get("options", [])
    return next(o["id"] for o in opts if o["text"] == text)


class FakeLLM:
    def __init__(self, script: Optional[dict] = None, handlers: Optional[dict[str, Callable]] = None,
                 cost: float = 0.0):
        self.script = script or {}
        self.handlers = handlers or {}
        self.cost = cost
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, model: str, messages: list[dict[str, str]], role: str = "", item_id: str = "",
                       variant_id: str = "", **_: Any) -> BackendResult:
        user = messages[1]["content"]
        data = extract_data(user)
        ctx = {"role": role, "model": model, "item_id": item_id, "variant_id": variant_id, "user": user,
               "system": messages[0]["content"], "messages": messages}
        self.calls.append(ctx)
        resp = None
        for key in ((role, model, item_id, variant_id), (role, model, item_id), (role, item_id)):
            if key in self.script:
                resp = self.script[key]
                break
        if resp is None:
            resp = self.handlers.get(role)
        if resp is None:
            raise AssertionError(f"fake LLM has no response for {role} {model} {item_id} {variant_id}")
        if callable(resp):
            resp = resp(data, ctx)
        text = resp if isinstance(resp, str) else json.dumps(resp)
        return BackendResult(text=text, cost=self.cost, prompt_tokens=len(user) // 4, completion_tokens=50)


# ---------------------------------------------------------------- a well-behaved fake for the fixture CSV

MCQ_TRUTH = {"mcq_ok": "600 W/m^2", "mcq_badkey": "Prandtl number", "mcq_named": "The temperature gradient"}
NUMERIC_TRUTH = {"num_conv": (1000.0, "W/m^2")}
STAGED_TRUTH = {"staged_rad": ((2835.0, "W/m^2"), {"S1": (3543.75, "W/m^2"), "S2": (2835.0, "W/m^2")})}
TOPIC_TRUTH = {"mcq_ok": "Conduction", "mcq_badkey": "Convection", "num_conv": "Convection",
               "staged_rad": "Radiation", "st_fins": "Extended surfaces", "st_biot": "Transient conduction",
               "rt_hx": "Heat exchangers", "mcq_named": "Fourier's law"}


def solver(data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    iid = ctx["item_id"]
    if iid in MCQ_TRUTH:
        lab = label_for_text(data, MCQ_TRUTH[iid])
        return {"answer": lab, "unit": "", "stage_answers": {},
                "option_verdicts": [{"id": o["id"], "correct": o["id"] == lab, "reason": ""} for o in data["options"]],
                "ambiguous": False, "working": f"picked {lab}"}
    if iid in NUMERIC_TRUTH:
        v, u = NUMERIC_TRUTH[iid]
        return {"answer": v, "unit": u, "working": "h dT"}
    if iid in STAGED_TRUTH:
        (v, u), st = STAGED_TRUTH[iid]
        return {"answer": v, "unit": u, "stage_answers": {k: {"value": a, "unit": b} for k, (a, b) in st.items()},
                "working": "sigma T^4"}
    return {"answer": ""}


def topic_classifier(data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    truth = TOPIC_TRUTH[ctx["item_id"]]
    others = [t for t in data["candidate_topics"] if t != truth and t != "NONE"]
    if ctx["model"].endswith("t2") and ctx["item_id"] == "rt_hx":
        ranked = [others[0], truth]       # one miss by the second classifier
    else:
        ranked = [truth, others[0]]
    return {"ranked": [{"topic": t, "confidence": 0.5} for t in ranked], "reason": "content"}


def reference_writer(data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    return {"answer": " ".join(c["text"] for c in data["rubric"])}


def mutator(data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    if "target_criterion" in data:
        cid = data["target_criterion"]["criterion_id"]
        return {"criterion_id": cid, "mutated_answer": data["reference_answer"] + f" [BROKEN {cid}]",
                "what_was_broken": cid, "edit_size": "small"}
    return {"paraphrased_answer": data["reference_answer"] + " (said differently)"}


def validator(data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    if "mutated_answer" in data:
        return {"target_violated": True, "others_preserved": True, "fluent": True, "note": ""}
    return {"meaning_preserved": True, "all_criteria_still_met": True, "note": ""}


def judge(data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    ans = data["student_answer"]
    out = []
    for c in data["criteria"]:
        cid = c["criterion_id"]
        broken = f"[BROKEN {cid}]" in ans
        # the second judge never fails c2: lower sensitivity, so judges differ
        if ctx["model"].endswith("j2") and cid == "c2":
            broken = False
        out.append({"criterion_id": cid, "verdict": "fail" if broken else "pass",
                    "quote": "" if broken else ans[:25], "confidence": "high"})
    return {"verdicts": out, "injection_detected": False}


def smart_fake(cost: float = 0.0) -> FakeLLM:
    return FakeLLM(handlers={"solver": solver, "topic_classifier": topic_classifier,
                             "reference_writer": reference_writer, "mutator": mutator,
                             "mutation_validator": validator, "judge": judge,
                             "verifier": lambda d, c: {"key_correct": True, "correct_answer": "", "reason": ""}},
                   cost=cost)
