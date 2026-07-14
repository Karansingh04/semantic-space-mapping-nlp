# Semantic Space Mapping in NLP

## Research Question
Can pretrained sentence embeddings automatically organize concepts into meaningful semantic groups?

## Method
- Dataset of 50 concepts in five semantic categories
- Pretrained Sentence-BERT (`all-MiniLM-L6-v2`)
- Normalized embeddings
- Pairwise cosine similarity
- K-Means clustering
- PCA visualization
- ARI, NMI and Silhouette evaluation

## Run
```bash
pip install -r requirements.txt
python src/main.py
```
Outputs are written to the results folder.
