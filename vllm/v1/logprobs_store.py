from dataclasses import dataclass, field
import threading
from typing import List

@dataclass
class GlobalLogProbs:
    logprobs: List[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, value: float):
        """Thread-safe method to add a single logprob."""
        with self.lock:
            self.logprobs.append(value)

    def add_batch(self, values: List[float]):
        """Thread-safe method to add multiple logprobs at once."""
        with self.lock:
            self.logprobs.extend(values)

    def get_all(self) -> List[float]:
        """Thread-safe method to get all logprobs."""
        with self.lock:
            return self.logprobs.copy()

    def clear(self):
        """Thread-safe method to delete all logprobs."""
        with self.lock:
            self.logprobs.clear()

global_logprobs = GlobalLogProbs()