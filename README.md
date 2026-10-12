# jailbreak-prompt-detection
Proyecto de NLP para la detección, clasificación y generación de jailbreak prompts


jailbreak-prompt-detection/
├── data/
│   ├── raw/          # copia sin tocar
│   ├── interim/      # pasos intermedios
│   └── processed/    # salida final (JSON/CSV)
├── notebooks/
│   ├── 01_auditoria.ipynb
│   ├── 02_limpieza.ipynb
│   └── ...           # prefijo numérico = orden
├── src/jailbreak_detection/
│   ├── data/         # carga, limpieza, deduplicación
│   ├── features/     # tokenización, TF-IDF, embeddings
│   └── models/       # (futuras entregas)
├── scripts/
│   └── run_e2_pipeline.py
├── docs/             # PDFs de las entregas
├── pyproject.toml
└── README.md