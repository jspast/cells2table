from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from cells2table.datamodels import BoundingBox
from cells2table.models.tasks import DetectionResult


@dataclass
class Cell:
    bbox: BoundingBox
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1


def sort_cells_index_by_top(cells: list[Cell]) -> list[int]:
    return sorted(range(len(cells)), key=lambda i: cells[i].bbox.t)


def sort_cells_index_by_bottom(cells: list[Cell]) -> list[int]:
    return sorted(range(len(cells)), key=lambda i: cells[i].bbox.b)


def sort_cells_index_by_left(cells: list[Cell]) -> list[int]:
    return sorted(range(len(cells)), key=lambda i: cells[i].bbox.l)


def sort_cells_index_by_right(cells: list[Cell]) -> list[int]:
    return sorted(range(len(cells)), key=lambda i: cells[i].bbox.r)


@dataclass
class Table:
    cells: list[Cell] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0

    @classmethod
    def from_detections(cls, cells_det: Iterable[DetectionResult], tolerance: float = 10) -> Table:
        table = cls()

        for cell_det in cells_det:
            bbox = BoundingBox.from_array(cell_det.bbox)
            cell = Cell(bbox=bbox, row=0, col=0)
            table.cells.append(cell)

        table.compute_structure(tolerance)
        return table

    def compute_structure(self, tolerance: float) -> None:
        self.compute_cells_row(tolerance)
        self.compute_cells_col(tolerance)

    def compute_cells_row(self, tolerance: float) -> None:
        indices = sort_cells_index_by_top(self.cells)

        spos = None  # Spatial position
        lpos = 0  # Logical position
        spanning: set[int] = set()

        for i in indices:
            cell_spos = self.cells[i].bbox.t

            # First cell
            if spos is None:
                spos = cell_spos

            # New row
            elif cell_spos - spos > tolerance:
                lpos += 1
                spos = cell_spos

                # Check for the end of spanning cells
                for j in list(spanning):
                    if self.cells[j].bbox.b - spos < tolerance:
                        self.cells[j].row_span = lpos - self.cells[j].row
                        spanning.remove(j)

            self.cells[i].row = lpos
            spanning.add(i)

        # End all cells
        if spos is not None:
            for j in list(spanning):
                self.cells[j].row_span = 1 + lpos - self.cells[j].row
                spanning.remove(j)

        self.num_rows = lpos + 1

    def compute_cells_col(self, tolerance: float) -> None:
        indices = sort_cells_index_by_left(self.cells)

        spos = None  # Spatial position
        lpos = 0  # Logical position
        spanning: set[int] = set()

        for i in indices:
            cell_spos = self.cells[i].bbox.l

            # First cell
            if spos is None:
                spos = cell_spos

            # New col
            elif cell_spos - spos > tolerance:
                lpos += 1
                spos = cell_spos

                # Check for the end of spanning cells
                for j in list(spanning):
                    if self.cells[j].bbox.r - spos < tolerance:
                        self.cells[j].col_span = lpos - self.cells[j].col
                        spanning.remove(j)

            self.cells[i].col = lpos
            spanning.add(i)

        # End all cells
        if spos is not None:
            for j in list(spanning):
                self.cells[j].col_span = 1 + lpos - self.cells[j].col
                spanning.remove(j)

        self.num_cols = lpos + 1
