# Models Directory

This directory stores serialized model artifacts for the project.

### Generating the Model
To compile and generate the model file (`ngram_predictor.pkl`), run the root training script:

```bash
python train_model.py
```

### Artifact Specifications
* **File:** `models/ngram_predictor.pkl` (~90.7 MB)
* **Model Class:** `AmharicPredictor` (Unigram, Bigram, and Trigram models with Add-$k$ smoothing and candidate indexing)
* **Vocabulary ($V$):** 188,263 words
