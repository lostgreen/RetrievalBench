"""Cost model, action ledger and cost-based metrics for AIF-V."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ActionName = str


@dataclass
class CostTable:
    """Holds action cost tables in two units: token-equivalent and latency-equivalent."""

    token: Mapping[ActionName, float]
    latency: Mapping[ActionName, float]

    def get(self, action: ActionName, unit: str = "token") -> float:
        if unit == "latency":
            return float(self.latency.get(action, 0.0))
        return float(self.token.get(action, 0.0))


@dataclass
class ActionEvent:
    act: ActionName
    args: Mapping[str, object] | None = None
    note: str | None = None
    units: float = 1.0
    cost_token: float = 0.0
    cost_latency: float = 0.0


@dataclass
class CostLedger:
    """Tracks actions and total costs in both units."""

    table: CostTable
    actions: List[ActionEvent] = field(default_factory=list)
    total_token: float = 0.0
    total_latency: float = 0.0

    def add(self, act: ActionName, *, args: Mapping[str, object] | None = None, note: str | None = None, units: float = 1.0) -> ActionEvent:
        c_tok = self.table.get(act, unit="token") * float(units)
        c_lat = self.table.get(act, unit="latency") * float(units)
        evt = ActionEvent(act=act, args=args, note=note, units=units, cost_token=c_tok, cost_latency=c_lat)
        self.actions.append(evt)
        self.total_token += c_tok
        self.total_latency += c_lat
        return evt

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_token": self.total_token,
            "total_latency": self.total_latency,
            "actions": [
                {
                    "act": e.act,
                    "args": e.args,
                    "note": e.note,
                    "units": e.units,
                    "cost_token": e.cost_token,
                    "cost_latency": e.cost_latency,
                }
                for e in self.actions
            ],
        }


def k_at_b(samples: Sequence[Mapping[str, object]], budgets: Sequence[float], *, unit: str = "token") -> Dict[str, float]:
    """Compute K@B for a list of per-sample dicts containing reward and costs.

    Each `samples[i]` must at least contain:
      - reward: int in {0,1}
      - costs.total_<unit>: float
    Accuracy under budget B is defined as: sum(1 for i with cost<=B and reward==1) / N.
    """
    key_total = f"total_{unit}"
    N = len(samples)
    out: Dict[str, float] = {}
    if N == 0:
        return {f"K@B/{unit}/{B}": 0.0 for B in budgets}
    for B in budgets:
        correct = 0
        for s in samples:
            costs = s.get("costs", {}) or {}
            total = float(costs.get(key_total, float("inf")))
            reward = int(s.get("reward", 0))
            if total <= float(B) and reward == 1:
                correct += 1
        out[f"K@B/{unit}/{B}"] = correct / N
    return out


def c_at_a(samples: Sequence[Mapping[str, object]], targets: Sequence[float], *, unit: str = "token") -> Dict[str, float]:
    """Compute C@A lower-bound estimate from single-run outcomes.

    We sort successful samples by cost ascending, then take the cheapest S=ceil(A*N)
    successes (if available). Lower-bound average per-item cost is sum(cost[:S]) / N.
    If not enough successes to reach A, we return 0.0.
    """
    key_total = f"total_{unit}"
    N = len(samples)
    out: Dict[str, float] = {}
    if N == 0:
        return {f"C@A/{unit}/{int(A*100)}%": 0.0 for A in targets}
    succ_costs = [
        float((s.get("costs", {}) or {}).get(key_total, float("inf")))
        for s in samples
        if int(s.get("reward", 0)) == 1
    ]
    succ_costs.sort()
    for A in targets:
        S = int(math.ceil(float(A) * N))
        if S <= len(succ_costs) and S > 0:
            cost_sum = sum(succ_costs[:S])
            out[f"C@A/{unit}/{int(A*100)}%"] = cost_sum / N
        else:
            out[f"C@A/{unit}/{int(A*100)}%"] = 0.0
    return out


def oracle_regret(samples: Sequence[Mapping[str, object]], *, unit: str = "token") -> Dict[str, float]:
    """Compute mean Oracle-Regret if oracle_cost is provided per sample.

    Regret_i = C_total_i - C_oracle_i. Returns mean over valid items; 0.0 if none.
    """
    key_total = f"total_{unit}"
    regrets: List[float] = []
    for s in samples:
        costs = s.get("costs", {}) or {}
        total = costs.get(key_total)
        oracle = s.get("oracle_cost", None)
        if total is None or oracle is None:
            continue
        try:
            regrets.append(float(total) - float(oracle))
        except Exception:
            continue
    return {f"OracleRegret/{unit}": (sum(regrets) / len(regrets) if regrets else 0.0)}

