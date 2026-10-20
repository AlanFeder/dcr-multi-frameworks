# dcr-3-frameworks

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Comparison of Multiple Frammeworks

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         dcr_3_frameworks and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── dcr_3_frameworks   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes dcr_3_frameworks a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------



## Audited dependencies

Use Python 3.11 or 3.12. `requirements.in` contains the application dependencies
and security minimums; `requirements.txt` pins the complete resolved graph.
NumPy 1.26, pandas 2.x, and patched PyArrow 23.x keep the included data compatible.

```sh
pip install -r requirements.txt
python -m pip check
python -P -m unittest discover -s tests -v
uv pip compile requirements.in --python-version 3.11 --universal --upgrade --no-annotate --output-file requirements.txt
```

The compatibility tests exercise the committed data, Parquet, and app startup
without external API calls. CI runs them on both supported Python versions and
audits every locked package with pip-audit. It does not deploy the app.

Set `OPENAI_API_KEY` in your environment for the RAG app. The tests use a dummy
key and disable SDK HTTP requests. The old requirements snapshot now delegates
to the maintained dependency graph; Gradio and Shiny are included explicitly.
