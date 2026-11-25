from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from math import isfinite


@dataclass
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    unpriced_calls: int = 0
    estimated_usd: float = 0.0

    def record(self, response, rates: tuple[float | None, ...], *, decision: bool) -> None:
        self.calls += 1
        usage = getattr(response, 'usage', None) if decision else response
        tokens = [getattr(usage, 'input_tokens', None)]
        if not decision:
            tokens.append(getattr(usage, 'output_tokens', None))
        counts = [n if type(n) is int and n > 0 else None for n in tokens]
        self.input_tokens += counts[0] or 0
        if not decision:
            self.output_tokens += counts[1] or 0
        if any(n is None or rate is None for n, rate in zip(counts, rates, strict=True)):
            self.unpriced_calls += 1
        self.estimated_usd += sum(n * rate for n, rate in zip(counts, rates, strict=True)
                                  if n is not None and rate is not None) / 1_000_000


@dataclass
class RunUsage:
    generation: ModelUsage = field(default_factory=ModelUsage)
    decision: ModelUsage = field(default_factory=ModelUsage)

    def emit(self, run_id: str, kind: str, status: str, attempt: int) -> None:
        total = self.generation.estimated_usd + self.decision.estimated_usd
        print(json.dumps({'event': 'model_usage', 'run_id': run_id, 'kind': kind, 'status': status,
                          'attempt': attempt, 'generation': vars(self.generation), 'decision': vars(self.decision),
                          'estimated_usd': total}, ensure_ascii=False), flush=True)


run_usage: ContextVar[RunUsage | None] = ContextVar('run_usage', default=None)


def rate(value) -> float | None:
    return float(value) if type(value) in (int, float) and isfinite(value) and value >= 0 else None
