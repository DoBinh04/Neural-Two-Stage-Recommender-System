import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def recall_at_k(ranked_items, relevant_items, k):
    top_k = ranked_items[:k]
    if not relevant_items:
        return 0.0
    hit_count = len(set(top_k) & relevant_items)
    return hit_count / len(relevant_items)


def ndcg_at_k(ranked_items, relevant_items, k):
    top_k = ranked_items[:k]
    dcg = 0.0
    for idx, item_id in enumerate(top_k, start=1):
        if item_id in relevant_items:
            dcg += 1.0 / np.log2(idx + 1)

    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def build_item_catalog_from_loader(model, loader, device):
    model.eval()
    item_ids = []
    item_vecs = []
    with torch.no_grad():
        for batch in loader:
            batch_device = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            vec = model.encode_item(batch_device)
            vec = F.normalize(vec, dim=1).cpu().numpy()

            raw_ids = batch["item_raw_id"]
            item_ids.extend([str(x) for x in raw_ids])
            item_vecs.append(vec)

    if not item_vecs:
        return np.empty((0, 0), dtype=np.float32), []

    item_vecs = np.concatenate(item_vecs, axis=0)

    unique = {}
    for item_id, item_vec in zip(item_ids, item_vecs):
        if item_id not in unique:
            unique[item_id] = item_vec

    uniq_ids = list(unique.keys())
    uniq_vecs = np.vstack([unique[i] for i in uniq_ids]).astype(np.float32)

    return uniq_vecs, uniq_ids


def evaluate_retrieval_metrics(model, loader, item_vecs, item_ids, device, ks=(20, 50)):
    model.eval()

    if item_vecs.shape[0] == 0:
        empty_metrics = {f"Recall@{k}": 0.0 for k in ks}
        empty_metrics.update({f"NDCG@{k}": 0.0 for k in ks})
        return empty_metrics

    item_tensor = torch.from_numpy(item_vecs).to(device)

    user_latest_vec = {}
    user_relevant_items = defaultdict(set)

    with torch.no_grad():
        for batch in loader:
            batch_device = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            user_vec = model.encode_user(batch_device)
            user_vec = F.normalize(user_vec, dim=1).cpu().numpy()

            user_raw_ids = [str(x) for x in batch["user_raw_id"]]
            item_raw_ids = [str(x) for x in batch["item_raw_id"]]
            relevance = batch["relevance"].cpu().numpy()

            for uid, iid, rel, uvec in zip(user_raw_ids, item_raw_ids, relevance, user_vec):
                user_latest_vec[uid] = uvec
                if rel > 0:
                    user_relevant_items[uid].add(iid)

    all_metrics = {f"Recall@{k}": [] for k in ks}
    all_metrics.update({f"NDCG@{k}": [] for k in ks})

    item_norm = F.normalize(item_tensor, dim=1)

    for uid, relevant in user_relevant_items.items():
        if not relevant:
            continue

        user_vec = torch.from_numpy(user_latest_vec[uid]).to(device).unsqueeze(0)
        user_vec = F.normalize(user_vec, dim=1)

        scores = torch.matmul(user_vec, item_norm.t()).squeeze(0)
        ranking = torch.argsort(scores, descending=True).cpu().numpy().tolist()
        ranked_items = [item_ids[idx] for idx in ranking]

        for k in ks:
            all_metrics[f"Recall@{k}"].append(recall_at_k(ranked_items, relevant, k))
            all_metrics[f"NDCG@{k}"].append(ndcg_at_k(ranked_items, relevant, k))

    summary = {}
    for name, values in all_metrics.items():
        summary[name] = float(np.mean(values)) if values else 0.0

    return summary


def export_embeddings(model, loader, output_dir, device, split_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    user_vectors = {}
    item_vectors = {}

    with torch.no_grad():
        for batch in loader:
            batch_device = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            user_vec = model.encode_user(batch_device)
            item_vec = model.encode_item(batch_device)

            user_vec = F.normalize(user_vec, dim=1).cpu().numpy()
            item_vec = F.normalize(item_vec, dim=1).cpu().numpy()

            user_ids = [str(x) for x in batch["user_raw_id"]]
            item_ids = [str(x) for x in batch["item_raw_id"]]

            for uid, vec in zip(user_ids, user_vec):
                user_vectors[uid] = vec

            for iid, vec in zip(item_ids, item_vec):
                if iid not in item_vectors:
                    item_vectors[iid] = vec

    user_ids = list(user_vectors.keys())
    user_matrix = np.vstack([user_vectors[u] for u in user_ids]).astype(np.float32) if user_ids else np.empty((0, 0), dtype=np.float32)

    item_ids = list(item_vectors.keys())
    item_matrix = np.vstack([item_vectors[i] for i in item_ids]).astype(np.float32) if item_ids else np.empty((0, 0), dtype=np.float32)

    user_emb_path = output_dir / f"user_embeddings_{split_name}.npz"
    item_emb_path = output_dir / f"item_embeddings_{split_name}.npz"

    np.savez(user_emb_path, ids=np.array(user_ids), vectors=user_matrix)
    np.savez(item_emb_path, ids=np.array(item_ids), vectors=item_matrix)

    mapping_path = output_dir / f"embedding_export_{split_name}.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "user_embeddings": str(user_emb_path),
                "item_embeddings": str(item_emb_path),
                "num_users": len(user_ids),
                "num_items": len(item_ids),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "user_embeddings": str(user_emb_path),
        "item_embeddings": str(item_emb_path),
        "num_users": len(user_ids),
        "num_items": len(item_ids),
    }
