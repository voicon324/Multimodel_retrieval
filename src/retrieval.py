import logging
from pathlib import Path
from typing import List, Tuple

import torch

# Giả sử ColIdefics3Processor có phương thức score_multi_vector
try:
    from colpali_engine.models import ColIdefics3Processor
except ImportError:
     raise ImportError("Không thể import ColIdefics3Processor. "
                      "Hãy đảm bảo thư viện 'colpali-engine' được cài đặt đúng cách "
                      "hoặc code của nó nằm trong PYTHONPATH.")

logger = logging.getLogger(__name__)

def load_all_embeddings(
    embeddings_folder: Path,
    device: str # Thiết bị để tải tensor lên (ví dụ: 'cuda:0' hoặc 'cpu')
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Tải tất cả các file embedding (.pt) từ thư mục chỉ định.
    Trả về danh sách các tensor embedding và danh sách tên file gốc tương ứng.
    """
    all_embeddings = []
    embedding_filenames = []
    logger.info(f"Đang tải embeddings từ: {embeddings_folder} lên {device}")

    if not embeddings_folder.is_dir():
        logger.warning(f"Thư mục embeddings không tồn tại: {embeddings_folder}")
        return [], []

    count = 0
    for emb_path in embeddings_folder.glob('*.pt'):
        try:
            # Tải tensor và chuyển ngay lên device mong muốn
            embedding = torch.load(emb_path, map_location=device)
            all_embeddings.append(embedding)
            embedding_filenames.append(emb_path.name) # Chỉ lưu tên file (ví dụ: 'page_1.pt')
            count += 1
        except Exception as e:
            logger.error(f"Lỗi khi tải embedding file {emb_path}: {e}")

    logger.info(f"Đã tải thành công {count} embeddings.")
    return all_embeddings, embedding_filenames

def find_top_k(
    query_embedding: torch.Tensor,
    all_embeddings: List[torch.Tensor],
    embedding_filenames: List[str],
    processor: ColIdefics3Processor,
    k: int
) -> List[Tuple[float, str]]:
    """
    Tìm kiếm K embedding giống nhất với query_embedding.
    Trả về danh sách các cặp (score, embedding_filename).
    """
    if not all_embeddings:
        logger.warning("Không có embeddings nào được tải để thực hiện tìm kiếm.")
        return []

    # Đảm bảo query_embedding và all_embeddings cùng device
    # Giả định chúng đã được load/tính toán trên cùng device
    device = query_embedding.device
    try:
        # Chuyển list các tensor thành một batch tensor duy nhất
        # Đảm bảo tất cả tensor trong list đều ở trên cùng device
        # image_embeddings_tensor = torch.stack([emb.to(device) for emb in all_embeddings])

        # Tính score sử dụng processor
        # Query cần có batch dimension (unsqueeze(0))
        # score_multi_vector(query[1, D], corpus[N, D]) -> scores[1, N] hoặc [N]
        with torch.no_grad():
            scores = processor.score_multi_vector(query_embedding.unsqueeze(0), all_embeddings)

        # scores có thể là [1, N] hoặc [N], cần squeeze() để thành 1D tensor [N]
        scores = scores.squeeze()

        # Lấy top K scores và indices
        # Đảm bảo k không lớn hơn số lượng embeddings hiện có
        actual_k = min(k, len(scores))
        if actual_k == 0:
             return []

        top_k_scores, top_k_indices = torch.topk(scores, k=actual_k)

        # Chuyển sang CPU và thành list python để trả về
        top_k_scores_list = top_k_scores.cpu().tolist()
        top_k_indices_list = top_k_indices.cpu().tolist()

        # Tạo kết quả: list các (score, filename)
        results = [
            (top_k_scores_list[i], embedding_filenames[top_k_indices_list[i]])
            for i in range(actual_k)
        ]

        logger.info(f"Tìm thấy top {len(results)} kết quả.")
        return results

    except Exception as e:
        logger.error(f"Lỗi trong quá trình tìm kiếm top-k: {e}")
        # Có thể thêm chi tiết lỗi vào log, ví dụ: device của tensors
        logger.debug(f"Query embedding device: {query_embedding.device}")
        if all_embeddings:
             logger.debug(f"First image embedding device: {all_embeddings[0].device}")
        return []