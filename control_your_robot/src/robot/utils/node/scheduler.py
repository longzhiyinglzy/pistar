import time
from threading import Event, Thread
from typing import List, Optional

from robot.utils.node.node import Node


class Scheduler:
    def __init__(
        self,
        entry_nodes: List[Node],
        all_nodes: List[Node],
        final_nodes: List[Node],
        hz: float = 5.0,
    ):
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.entry_nodes = entry_nodes
        self.all_nodes = all_nodes
        self.final_nodes = final_nodes
        self.period = 1.0 / hz
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._running_episode = False

    def start(self):
        self._thread = Thread(target=self._run, name="Scheduler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        for node in self.all_nodes:
            node.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        print(f"[SCHED] start @ {1 / self.period:.1f} Hz")
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            now = time.monotonic()
            if self._running_episode:
                if self._all_final_nodes_done():
                    self._running_episode = False
                    self._reset_all_nodes()
                time.sleep(0.001)
                continue

            if now >= next_tick:
                self._trigger_entry_nodes()
                self._running_episode = True
                next_tick += self.period

            time.sleep(0.00001)

        print("[SCHED] stopped")

    def _all_final_nodes_done(self) -> bool:
        return all(node.end_event.is_set() for node in self.final_nodes)

    def _trigger_entry_nodes(self):
        for node in self.entry_nodes:
            node.forward_event.set()

    def _reset_all_nodes(self):
        for node in self.all_nodes:
            node.reset()
