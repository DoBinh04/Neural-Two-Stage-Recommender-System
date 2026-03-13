from pathlib import Path

import numpy as np


def build_faiss_index(item_vectors, item_ids, output_dir, index_name="item_faiss.index"):
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is not installed. Please install faiss-cpu or faiss-gpu before building ANN index."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if item_vectors.size == 0:
        raise ValueError("Cannot build FAISS index with empty item vectors.")

    dim = item_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_vectors.astype(np.float32))

    index_path = output_dir / index_name
    ids_path = output_dir / "item_ids.npy"

    faiss.write_index(index, str(index_path))
    np.save(ids_path, np.array(item_ids))

    return {
        "index_path": str(index_path),
        "ids_path": str(ids_path),
        "num_items": int(item_vectors.shape[0]),
        "dim": int(dim),
    }
