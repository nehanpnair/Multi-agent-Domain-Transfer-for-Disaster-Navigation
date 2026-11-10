"""
comms.py
--------
Explicit communication layer for the swarm.

Replaces three things that were implicit before:
  * a global, instantaneous shared memory,
  * a hard-coded 500 m danger-memory merge,
  * pheromones that were always on.

Everything an agent learns from another agent now travels as a Message through
a MessageBus, so radius, delay, message counts and bytes are all measurable and
all three channels can be switched on and off independently.

Channels (CommConfig switches)
------------------------------
  shared_memory     agents post what they sense to a global board; every agent
                    reads it. Posts go through the bus (kind="board") so
                    `board_delay_steps` models uplink latency.
  local_messaging   agent-to-agent radio. Kinds:
                      "hazard"    delta of newly sensed hazards, broadcast
                      "sync"      full knowledge exchanged when two agents
                                  come into range (optional, sync_on_contact)
                      "advisory"  consolidated hazards from an ASSESSOR hub
                      "assign"    goal assignment from an ALLOCATOR hub
  pheromones        trail deposited by agents that reach their goal.

Link rule
---------
A transmission from A reaches B iff
        distance(A, B) <= max(range(A), range(B))
where range(X) = radius_m * role radius multiplier. The max() lets a hub with a
large antenna both hear and be heard across its footprint. Distance is measured
when the message is sent.

Cost model
----------
One broadcast = one TRANSMISSION (what the sender pays) and one DELIVERY per
receiver (what the swarm receives). Bytes use a fixed, documented size model
(see payload_bytes) so runs are comparable; it is not a wire format.

Timing
------
A message sent at step t with delay d is delivered in the comms phase of step
t + d, so with d = 0 it is usable by the receiver on step t + 1.
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

BOARD = -1   # pseudo-recipient id for the shared board

HEADER_BYTES = 16
NODE_ID_BYTES = 8
DANGER_ENTRY_BYTES = 12    # node id + float32
SCALAR_BYTES = 4


@dataclass(frozen=True)
class CommConfig:
    shared_memory:    bool  = True
    local_messaging:  bool  = True
    pheromones:       bool  = True
    radius_m:         float = 500.0   # base radio range
    delay_steps:      int   = 0       # local message latency
    board_delay_steps: int  = 0       # latency before a post is visible on the board
    sync_on_contact:  bool  = True    # exchange full knowledge when agents meet

    def with_(self, **kw) -> "CommConfig":
        """Copy with overrides, for ablation sweeps: cfg.with_(local_messaging=False)."""
        from dataclasses import replace
        return replace(self, **kw)


def payload_bytes(payload: dict) -> int:
    n = HEADER_BYTES
    for k, v in payload.items():
        if k == "blocked":
            n += NODE_ID_BYTES * len(v)
        elif k == "danger":
            n += DANGER_ENTRY_BYTES * len(v)
        elif k == "goal":
            n += NODE_ID_BYTES
        else:
            n += SCALAR_BYTES
    return n


@dataclass
class Message:
    id: int
    kind: str
    sender: int
    sent_step: int
    payload: dict
    size_bytes: int = 0


@dataclass(order=True)
class _Pending:
    deliver_step: int
    seq: int
    recipient: int = field(compare=False)
    msg: Message = field(compare=False)


class MessageBus:
    def __init__(self, cfg: CommConfig):
        self.cfg = cfg
        self._heap: list = []
        self._seq = 0
        self._next_id = 0
        self.tx = defaultdict(int)            # kind -> transmissions
        self.rx = defaultdict(int)            # kind -> deliveries
        self.bytes_tx = defaultdict(int)
        self.bytes_rx = defaultdict(int)
        self.dropped = defaultdict(int)       # kind -> deliveries lost (receiver dead)
        self.tx_by_sender = defaultdict(int)
        self.bytes_by_sender = defaultdict(int)
        self.rx_by_recipient = defaultdict(int)
        self._delay_sum = 0

    # ------------------------------------------------------------------ #
    def send(self, sender: int, kind: str, payload: dict, step: int,
             recipients: Iterable[int], delay: Optional[int] = None) -> Optional[Message]:
        """One transmission, delivered to each recipient after `delay` steps.
        Returns None (and costs nothing) if there is nobody to deliver to."""
        recipients = list(recipients)
        if not recipients:
            return None
        if delay is None:
            delay = self.cfg.delay_steps
        size = payload_bytes(payload)
        msg = Message(self._next_id, kind, sender, step, payload, size)
        self._next_id += 1
        self.tx[kind] += 1
        self.bytes_tx[kind] += size
        self.tx_by_sender[sender] += 1
        self.bytes_by_sender[sender] += size
        for r in recipients:
            self._seq += 1
            heapq.heappush(self._heap, _Pending(step + delay, self._seq, r, msg))
        return msg

    def deliver(self, step: int, is_alive=lambda r: True) -> list:
        """Pop everything due at or before `step`. Returns [(recipient, msg)].
        Deliveries to dead recipients are counted as dropped, not delivered."""
        out = []
        while self._heap and self._heap[0].deliver_step <= step:
            p = heapq.heappop(self._heap)
            if p.recipient != BOARD and not is_alive(p.recipient):
                self.dropped[p.msg.kind] += 1
                continue
            self.rx[p.msg.kind] += 1
            self.bytes_rx[p.msg.kind] += p.msg.size_bytes
            self.rx_by_recipient[p.recipient] += 1
            self._delay_sum += step - p.msg.sent_step
            out.append((p.recipient, p.msg))
        return out

    def pending(self) -> int:
        return len(self._heap)

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        kinds = sorted(set(self.tx) | set(self.rx))
        n_rx = sum(self.rx.values())
        return {
            "transmissions":   sum(self.tx.values()),
            "deliveries":      n_rx,
            "dropped":         sum(self.dropped.values()),
            "bytes_tx":        sum(self.bytes_tx.values()),
            "bytes_rx":        sum(self.bytes_rx.values()),
            "mean_delay_steps": (self._delay_sum / n_rx) if n_rx else 0.0,
            "in_flight":       self.pending(),
            "by_kind": {k: {"tx": self.tx[k], "rx": self.rx[k],
                            "bytes_tx": self.bytes_tx[k], "bytes_rx": self.bytes_rx[k],
                            "dropped": self.dropped[k]} for k in kinds},
            "tx_by_sender":    dict(self.tx_by_sender),
            "bytes_by_sender": dict(self.bytes_by_sender),
        }
