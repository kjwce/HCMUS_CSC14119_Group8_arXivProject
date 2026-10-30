"""Shared data contracts for the two Milestone 2 contributors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(slots=True)
class Node:
    """One logical element in a parsed LaTeX document.

    ``content`` contains only this node's own semantic value. Children hold the
    lower-level elements, so a section stores its title while its sentences are
    children. The class deliberately has no storage identifier; IDs are assigned
    only after cleanup by :mod:`src.store`.
    """

    kind: str
    content: str = ""
    children: list["Node"] = field(default_factory=list)

    def add(self, child: "Node") -> "Node":
        self.children.append(child)
        return child

    def walk(self, depth: int = 0) -> Iterator[tuple["Node", int]]:
        yield self, depth
        for child in self.children:
            yield from child.walk(depth + 1)

