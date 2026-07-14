from threading import Event, Lock, Thread
from typing import List, Optional


class Node:
    def __init__(self, name: str):
        self.name = name
        self.start_events: List[Event] = []
        self.forward_event = Event()
        self.end_event = Event()
        self.next_nodes: List["Node"] = []
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._lock = Lock()
        self._has_run = False

    def next_to(self, next_node: "Node"):
        self.next_nodes.append(next_node)
        next_node.add_start_event(self.end_event)

    def add_start_event(self, event: Event):
        self.start_events.append(event)

    def start(self):
        self._thread = Thread(target=self._run, name=f"Node-{self.name}", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.forward_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def reset(self):
        with self._lock:
            self._has_run = False
            self.end_event.clear()
            self.forward_event.clear()

    def _ready(self):
        return all(event.is_set() for event in self.start_events)

    def _run(self):
        while not self._stop_event.is_set():
            self.forward_event.wait()
            self.forward_event.clear()

            if self._stop_event.is_set():
                break

            with self._lock:
                if self._has_run or not self._ready():
                    continue
                self._has_run = True

            self.handler()
            self.end_event.set()
            for node in self.next_nodes:
                node.forward_event.set()

    def handler(self):
        raise NotImplementedError


class TaskNode(Node):
    def __init__(self, name: str, **task_kwargs):
        super().__init__(name)
        self.task_kwargs = task_kwargs
        self._inited = False

    def task_init(self, **kwargs):
        pass

    def task_step(self):
        pass

    def handler(self):
        if not self._inited:
            self.task_init(**self.task_kwargs)
            self._inited = True
        self.task_step()
