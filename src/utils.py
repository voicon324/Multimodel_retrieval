import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir_exists(path: Path):
    """Kiểm tra và tạo thư mục nếu chưa tồn tại."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Đảm bảo thư mục tồn tại: {path}")
    except Exception as e:
        logger.error(f"Không thể tạo thư mục {path}: {e}")
        raise # Re-raise the exception after logging

def get_image_path_from_embedding(
    embedding_filename: str,
    processed_images_folder: Path
) -> Path | None:
    """
    Lấy đường dẫn file ảnh từ tên file embedding.
    Giả định tên file ảnh và embedding giống nhau (chỉ khác phần mở rộng).
    Ưu tiên tìm .png, sau đó là các định dạng phổ biến khác.
    """
    base_name = Path(embedding_filename).stem
    possible_extensions = ['.png', '.jpg', '.jpeg', '.webp'] # Ưu tiên .png

    for ext in possible_extensions:
        image_path = processed_images_folder / f"{base_name}{ext}"
        if image_path.is_file():
            return image_path

    logger.warning(f"Không tìm thấy file ảnh tương ứng cho embedding: {embedding_filename} trong {processed_images_folder}")
    return None

def get_embedding_path(
    image_filename: str, # Chỉ tên file, ví dụ 'page_1.png'
    embeddings_folder: Path
) -> Path:
    """Lấy đường dẫn file embedding (.pt) từ tên file ảnh."""
    base_name = Path(image_filename).stem
    return embeddings_folder / f"{base_name}.pt"

def sanitize_filename(filename: str) -> str:
    """Loại bỏ các ký tự không hợp lệ cho tên file."""
    # Giữ lại chữ cái, số, dấu gạch dưới, dấu gạch ngang, dấu chấm
    # Thay thế các ký tự khoảng trắng hoặc không hợp lệ khác bằng dấu gạch dưới
    import re
    sanitized = re.sub(r'[^\w\-.]', '_', filename)
    # Giảm các dấu gạch dưới liên tiếp thành một dấu
    sanitized = re.sub(r'_+', '_', sanitized)
    # Xóa dấu gạch dưới ở đầu hoặc cuối (nếu có)
    sanitized = sanitized.strip('_')
    return sanitized