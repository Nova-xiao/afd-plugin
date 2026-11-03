# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded eager work descriptors. Tensor data travels in the UMDK window."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed import Store

POLL_SECONDS = 0.001
PUBLISH_TIMEOUT_SECONDS = 120


class ExpertWorkQueue:
    """One mailbox per (Attention DP, FFN rank), each with exactly one reader."""

    def __init__(
        self, store: Store, sources: int, experts: int, wrap: Callable[[int], int]
    ):
        self.store = store
        self.sources = sources
        self.experts = experts
        self.wrap = wrap
        self.last = [0] * sources
        self.cursor = 0

    def next_sequence(self, source: int) -> int:
        return self.wrap(self.last[source] + 1)

    def publish(self, source: int, layer: int, sequence: int) -> None:
        keys = [f"work/{source}/{ffn}" for ffn in range(self.experts)]
        deadline = time.monotonic() + PUBLISH_TIMEOUT_SECONDS
        for key in keys:
            while self.store.check([key]):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"FFN did not claim previous work in {key}")
                time.sleep(POLL_SECONDS)
            self.store.set(key, f"{layer} {sequence}")
        self.last[source] = sequence

    def poll(self, ffn: int) -> tuple[int, int, int] | None:
        for offset in range(self.sources):
            source = (self.cursor + offset) % self.sources
            key = f"work/{source}/{ffn}"
            if self.store.check([key]):
                layer, sequence = map(int, self.store.get(key).split())
                if sequence != self.next_sequence(source):
                    raise RuntimeError(f"Out-of-order MegaMoE work in {key}")
                self.store.delete_key(key)
                self.last[source] = sequence
                self.cursor = (source + 1) % self.sources
                return source, layer, sequence
        return None
