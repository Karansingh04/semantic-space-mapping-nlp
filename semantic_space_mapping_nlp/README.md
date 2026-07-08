# Semantic Space Mapping in NLP

This is a Natural Language Processing final project based on **P15: Cartographers of the Invisible**.

## Research Question
Can text embeddings automatically organize concepts into meaningful semantic groups?

## Method
The project uses:
- a small concept dataset with semantic categories
- TF-IDF text embeddings based on concept descriptions
- K-Means clustering
- PCA visualization
- clustering evaluation metrics

## How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

The outputs will be saved in the `results/` folder:
- `clusters.csv`
- `semantic_map.png`
- `evaluation.txt`

## Repository Structure

```text
data/concepts.csv       input dataset
src/main.py             full experiment code
results/                generated outputs
docs/                   project report
slides/                 presentation slides
```

## AI Usage Disclaimer
Parts of the project structure and drafting were supported by OpenAI's ChatGPT. All content, code, and results should be reviewed and validated by me before submission.