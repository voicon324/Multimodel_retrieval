import logging
from typing import List, Tuple

import torch
from PIL import Image

# Giả sử ColIdefics3 và ColIdefics3Processor nằm trong colpali_engine
# Nếu không, hãy điều chỉnh import cho phù hợp
try:
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor, ColQwen2, ColQwen2Processor
except ImportError:
    # Cung cấp hướng dẫn nếu không tìm thấy module
    raise ImportError("Không thể import ColIdefics3 hoặc ColIdefics3Processor. "
                      "Hãy đảm bảo thư viện 'colpali-engine' được cài đặt đúng cách "
                      "hoặc code của nó nằm trong PYTHONPATH.")


logger = logging.getLogger(__name__)

# Hàm này sẽ được cache trong app.py bằng @st.cache_resource
def load_model_processor(model_name: str, device: str) -> Tuple[ColIdefics3, ColIdefics3Processor]:
    """Tải model và processor ColIdefics3."""
    logger.info(f"Đang tải model và processor: {model_name} lên {device}")
    try:
        # Xác định device_map dựa trên device
        # device_map="auto" thường hoạt động tốt, nhưng chỉ định rõ ràng cũng được
        effective_device_map = device if "cuda" in device else "cpu"
        if effective_device_map == "cpu":
            logger.warning("Đang tải model lên CPU. Quá trình embedding có thể chậm.")
            model_dtype = torch.float32 # bfloat16 không được hỗ trợ tốt trên CPU
        else:
            model_dtype = torch.bfloat16 # Sử dụng bfloat16 cho GPU nếu hỗ trợ

        if model_name == "vidore/colSmol-256M":
            model = ColIdefics3.from_pretrained(
                model_name,
                torch_dtype=model_dtype,
                device_map=effective_device_map, # Hoặc "auto"
                attn_implementation="eager" # Sử dụng "flash_attention_2" nếu có và tương thích
            ).eval()

            processor = ColIdefics3Processor.from_pretrained(model_name)
            logger.info("Tải model và processor thành công.")
            return model, processor
        
        elif model_name == "vidore/colqwen2-v1.0":
            model = ColQwen2.from_pretrained(
                model_name,
                torch_dtype=model_dtype,
                device_map=effective_device_map, # Hoặc "auto"
                attn_implementation="eager" # Sử dụng "flash_attention_2" nếu có và tương thích
            ).eval()

            processor = ColQwen2Processor.from_pretrained(model_name)
            logger.info("Tải model và processor thành công.")
            return model, processor
        
        elif model_name == '5CD-AI/ColVintern-1B-v1':
            from transformers import AutoModel, AutoTokenizer, AutoProcessor

            processor =  AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval().cuda()

            return model, processor

        else:
            logger.error(f"Model {model_name} không được hỗ trợ. Vui lòng kiểm tra tên model.")
            raise ValueError(f"Model {model_name} không được hỗ trợ. Vui lòng kiểm tra tên model.")

    except Exception as e:
        logger.error(f"Lỗi khi tải model {model_name}: {e}")
        raise # Re-raise để Streamlit có thể hiển thị lỗi

def get_image_embedding(
    image: Image.Image,
    model: ColIdefics3,
    processor: ColIdefics3Processor,
    device: str
) -> torch.Tensor | None:
    """Tạo embedding cho một ảnh PIL."""
    try:
        # Đảm bảo ảnh là RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Xử lý ảnh (đặt trong list vì processor thường xử lý batch)
        batch_images = processor.process_images([image]).to(device)

        # Forward pass để lấy embedding
        with torch.no_grad():
            # Kiểm tra xem model cần input gì (có thể chỉ cần pixel_values)
            # Dựa vào code gốc, nó nhận trực tiếp dict từ processor
            embeddings = model(**batch_images) # Giả định model trả về tensor embeddings trực tiếp

        # Giả định output là tensor [batch_size, embedding_dim]
        if embeddings is not None and embeddings.ndim >= 2:
             # Lấy embedding của ảnh đầu tiên (và duy nhất trong batch này)
             # Chuyển về CPU trước khi trả về để giải phóng VRAM nếu cần
            return embeddings[0].cpu()
        else:
            logger.error(f"Output không mong đợi từ model khi embedding ảnh: {type(embeddings)}")
            return None

    except Exception as e:
        logger.error(f"Lỗi khi tạo image embedding: {e}")
        return None

def get_text_embedding(
    text: str,
    model: ColIdefics3,
    processor: ColIdefics3Processor,
    device: str
) -> torch.Tensor | None:
    """Tạo embedding cho một đoạn văn bản."""
    try:
        # Xử lý query (đặt trong list)
        batch_queries = processor.process_queries([text]).to(device)

        # Forward pass
        with torch.no_grad():
            embeddings = model(**batch_queries)

        # Giả định output là tensor [batch_size, embedding_dim]
        if embeddings is not None and embeddings.ndim >= 2:
             # Lấy embedding của query đầu tiên và chuyển về CPU
            return embeddings[0].cpu()
        else:
            logger.error(f"Output không mong đợi từ model khi embedding text: {type(embeddings)}")
            return None

    except Exception as e:
        logger.error(f"Lỗi khi tạo text embedding: {e}")
        return None

# (Optional) Batch embedding nếu cần tối ưu cho nhiều ảnh cùng lúc
# def get_batch_image_embeddings(...)