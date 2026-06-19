# Tradeoffs / what is NOT built (yet)

- Full experiment tracking system (MLflow/Weights&Biases): out of scope; can be integrated later.
- True safe expression parsing for slice rules: we use pandas query/eval with cautious logging. For hostile inputs, replace with a proper parser.
- Matching task specialized schemas: we treat matching as binary classification over engineered features.
- Advanced error clustering (embeddings/UMAP/HDBSCAN): baseline TF-IDF + KMeans is enough to find repeated modes; upgrade later.
- Bias/variance decomposition with repeated seeds: stubbed modules exist; add if needed.
- Interactive dashboards: we output JSON and PNG; plug into internal BI later.
