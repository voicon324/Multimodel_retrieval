import logging
from pathlib import Path
from typing import List

import fitz # PyMuPDF
from PIL import Image

from .utils import sanitize_filename

logger = logging.getLogger(__name__)

def convert_pdf_to_images(pdf_path: Path, output_folder: Path) -> List[Path]:
    """
    Chuyển đổi mỗi trang PDF thành ảnh PNG và lưu vào output_folder.
    Trả về danh sách đường dẫn các ảnh đã tạo.
    """
    processed_image_paths = []
    sanitized_pdf_stem = sanitize_filename(pdf_path.stem)

    try:
        doc = fitz.open(pdf_path)
        logger.info(f"Đang xử lý PDF: {pdf_path.name} với {len(doc)} trang.")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render ở độ phân giải cao hơn (ví dụ: DPI=200) để chất lượng tốt hơn
            # Có thể điều chỉnh DPI tùy theo nhu cầu và hiệu năng
            zoom = 2 # zoom factor (2 = 144 dpi if base is 72 dpi)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Tạo tên file duy nhất cho mỗi trang
            image_filename = f"{sanitized_pdf_stem}_page_{page_num + 1}.png"
            output_path = output_folder / image_filename

            pix.save(str(output_path)) # Lưu ảnh PNG
            processed_image_paths.append(output_path)
            logger.debug(f"Đã lưu trang {page_num + 1} của PDF {pdf_path.name} vào {output_path}")

        doc.close()
        logger.info(f"Hoàn thành chuyển đổi PDF {pdf_path.name}. Đã tạo {len(processed_image_paths)} ảnh.")
        return processed_image_paths

    except Exception as e:
        logger.error(f"Lỗi khi xử lý PDF {pdf_path}: {e}")
        return [] # Trả về danh sách rỗng nếu có lỗi

def save_uploaded_image(
    uploaded_file, # Streamlit UploadedFile object
    output_folder: Path
) -> Path | None:
    """
    Lưu file ảnh được tải lên vào output_folder dưới dạng PNG.
    Trả về đường dẫn file đã lưu hoặc None nếu lỗi.
    """
    try:
        # Tạo tên file an toàn và chuẩn hóa thành .png
        original_filename = uploaded_file.name
        sanitized_stem = sanitize_filename(Path(original_filename).stem)
        output_filename = f"{sanitized_stem}.png"
        output_path = output_folder / output_filename

        # Đọc ảnh bằng PIL, đảm bảo là RGB và lưu lại dưới dạng PNG
        img = Image.open(uploaded_file).convert("RGB")
        img.save(output_path, format="PNG")
        logger.info(f"Đã lưu ảnh tải lên {original_filename} vào {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Lỗi khi lưu ảnh tải lên {uploaded_file.name}: {e}")
        return None