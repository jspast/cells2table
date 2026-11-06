# cells2table

Efficient pipelines and models for document AI, with Docling support via plugins.

## Instalation

With [uv](https://docs.astral.sh/uv/), add to your project using:

``` sh
uv add cells2table
```

### Extras

| Optional          | Description                       |
| ----------------- | --------------------------------- |
| `huggingface`     | For downloading models from HF    |
| `onnxruntime`     | For using ONNX Runtime as runtime |
| `transformers`    | For using transformers as runtime |
| `opencv-headless` | For using ONNX models             |
| `opencv`          | For using ONNX models and the CLI |

## Usage

Check the [documentation](https://jspast.github.io/cells2table/).

## Citation

``` bibtex
@InProceedings{Pastorello2027TableStructureRecognition,
  author    = {Pastorello, Jo{\~a}o Speranza and
               Tamiosso, Gustavo Lopes and
               de Oliveira, Gabriel Bauer and
               Geiger, Albert and
               da Silveira, Evandro Dalbosco and
               Balreira, Dennis Giovani and
               Tavares, Anderson Rocha},
  editor    = {Recamonde-Mendoza, Mariana and Barros, Rodrigo C.},
  title     = {Efficient Table Structure Recognition via Deep Learning and Lightweight Heuristics},
  booktitle = {Intelligent Systems},
  year      = {2027},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {643--657},
  isbn      = {978-3-032-39895-6},
  doi       = {10.1007/978-3-032-39895-6_43}
}
```
