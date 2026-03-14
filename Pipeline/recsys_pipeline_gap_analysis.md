# RecSys pipeline gap analysis and additions

## Existing in project

- ✅ Retrieval model training with **Two-Tower** (`Retrieval/train_retrieval.py`).
- ✅ Export user/item embeddings for downstream stages (`export_embeddings` in `Retrieval/Train/evaluation.py`).
- ✅ Build FAISS index from item embeddings (`Retrieval/Train/faiss_index.py`).

## Missing before this update

- ❌ No reusable utility to **query FAISS** and return Top-N candidates for a user embedding.
- ❌ No supervised **ranking model** implementation for reranking retrieved candidates.
- ❌ No training script implementing **Wide & Deep ranking** from retrieval outputs.

## Added in this update

1. `Retrieval/Train/candidate_retrieval.py`
   - `load_faiss_index(...)`: load index + item IDs.
   - `retrieve_topk(...)`: query Top-N candidate items from user embedding(s).

2. `Reranking/wide_deep_model.py`
   - `WideAndDeepRanker`: wide linear branch + deep MLP branch, output relevance logit.

3. `Reranking/train_wide_deep.py`
   - Build ranking dataset from:
     - positive interactions (`addtocart`, `transaction`),
     - retrieval candidates from FAISS,
     - exported user/item embeddings.
   - Feature design:
     - user embedding,
     - item embedding,
     - element-wise interaction (`user_emb * item_emb`).
   - Train with BCEWithLogits loss.
   - Evaluate with Recall@10 at query-group level.
   - Save best/last checkpoints and JSON summary.

## Suggested run order

1. Train retrieval + export embeddings + build FAISS:

```bash
python Retrieval/train_retrieval.py
```

2. Train ranking (Wide & Deep):

```bash
python Reranking/train_wide_deep.py
```

