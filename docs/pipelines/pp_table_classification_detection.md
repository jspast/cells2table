# PP Table Classification and Detection Pipeline

This pipeline uses deep learning models and lightweight heuristics for Table Structure Recognition.
It combines table classification with specialized cell detection:

1. **Classify** each table as wired or wireless
2. **Detect** cells in the table using appropriate set of weights for the class
3. **Reconstruct** row and column structure with positional heuristics

``` mermaid
graph LR
  A[Table image] --> B{PP-LCNet_x1_0_table_cls};
  B -->|wired| C[RT-DETR-L_wired_table_cell_det];
  B -->|wireless| D[RT-DETR-L_wireless_table_cell_det];
  C --> E[Cell detections];
  D --> E[Cell detections];
  E -->|Heuristics| F[Table structure]

```

This pipeline is provided as a Docling plugin for being part of a complete document parsing pipeline.
Check our documentation on [usage with Docling](../usage/docling.md).

??? info "The name cells2table"

    The name cells2table originated from this pipeline's concept of building tables from cell detections.
