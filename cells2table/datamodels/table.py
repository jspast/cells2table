from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..models.tasks import DetectionResult
from .bbox import BoundingBox


@dataclass
class Cell:
    bbox: BoundingBox
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1


@dataclass
class Table:
    cells: list[Cell] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0

    @staticmethod
    def from_detections(cells_det: Iterable[DetectionResult], tolerance: float = 10) -> Table:
        table = Table()

        for cell_det in cells_det:
            bbox = BoundingBox.from_array(cell_det.bbox)
            cell = Cell(bbox=bbox, row=0, col=0)
            table.cells.append(cell)

        table.compute_rows_and_cols(tolerance)
        return table

    def compute_rows_and_cols(self, tolerance: float) -> None:
        self.compute_rows(tolerance)
        self.compute_cols(tolerance)

    def sort_cells_by_rows(self, cells: Optional[Iterable[Cell]] = None) -> list[Cell]:
        if cells is None:
            cells = self.cells

        return sorted(self.cells, key=lambda cell: cell.bbox.t)

    def sort_cells_by_cols(self, cells: Optional[Iterable[Cell]] = None) -> list[Cell]:
        if cells is None:
            cells = self.cells

        return sorted(self.cells, key=lambda cell: cell.bbox.l)

    def compute_rows(self, tolerance: float) -> None:
        self.cells = self.sort_cells_by_rows()

        row_y = None
        row_num = 0
        row_start_idx = 0
        row_end_idx = None
        check_span_indices: set[int] = set()

        for i in range(len(self.cells)):
            if row_y is None:
                row_y = self.cells[i].bbox.t

            elif abs(self.cells[i].bbox.t - row_y) > tolerance:
                row_end_idx = i
                for j in range(row_start_idx, row_end_idx):
                    self.cells[j].row = row_num
                    check_span_indices.add(j)

                row_y = self.cells[i].bbox.t
                row_start_idx = row_end_idx
                row_num += 1

                for j in list(check_span_indices):
                    if self.cells[j].bbox.b > row_y + tolerance:
                        self.cells[j].row_span += 1
                    else:
                        check_span_indices.remove(j)

        row_end_idx = len(self.cells)
        for j in range(row_start_idx, row_end_idx):
            self.cells[j].row = row_num
            check_span_indices.add(j)

        self.num_rows = row_num + 1

    def compute_cols(self, tolerance: float) -> None:
        self.cells = self.sort_cells_by_cols()

        col_x = None
        col_num = 0
        col_start_idx = 0
        col_end_idx = None
        check_span_indices: set[int] = set()

        for i in range(len(self.cells)):
            if col_x is None:
                col_x = self.cells[i].bbox.l

            elif abs(self.cells[i].bbox.l - col_x) > tolerance:
                col_end_idx = i
                for j in range(col_start_idx, col_end_idx):
                    self.cells[j].col = col_num
                    check_span_indices.add(j)

                col_x = self.cells[i].bbox.l
                col_start_idx = col_end_idx
                col_num += 1

                for j in list(check_span_indices):
                    if self.cells[j].bbox.r > col_x + tolerance:
                        self.cells[j].col_span += 1
                    else:
                        check_span_indices.remove(j)

        col_end_idx = len(self.cells)
        for j in range(col_start_idx, col_end_idx):
            self.cells[j].col = col_num
            check_span_indices.add(j)

        self.num_cols = col_num + 1
