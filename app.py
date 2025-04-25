import streamlit as st
from PIL import Image
import torch
from pathlib import Path
import time
import logging

# Cấu hình logging cơ bản cho Streamlit app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import các module từ src
# Giả sử các file này tồn tại trong thư mục src cùng cấp với app.py
# Nếu cấu trúc khác, bạn cần điều chỉnh import
try:
    from src import utils, preprocessing, embedding, retrieval
except ImportError as e:
    st.error(f"Lỗi import module từ src: {e}. Đảm bảo các file utils.py, preprocessing.py, embedding.py, retrieval.py tồn tại trong thư mục 'src'.")
    st.stop()


# --- Cấu hình ---
# MODEL_NAME = "vidore/colSmol-256M"  # Thay đổi nếu cần
# MODEL_NAME = "vidore/colqwen2-v1.0"  # Thay đổi nếu cần
MODEL_NAME = "5CD-AI/ColVintern-1B-v1"  # Thay đổi nếu cần
# Cố gắng sử dụng GPU nếu có
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
    logger.warning("Không tìm thấy GPU CUDA. Sử dụng CPU. Quá trình xử lý có thể chậm.")

# Xác định các đường dẫn thư mục dữ liệu
APP_DIR = Path(__file__).parent # Thư mục chứa app.py
DATA_DIR = APP_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_IMAGES_DIR = DATA_DIR / "processed_images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# --- Khởi tạo ---
st.set_page_config(layout="wide", page_title="Document Retrieval App")

# Tạo các thư mục dữ liệu nếu chưa có
utils.ensure_dir_exists(UPLOADS_DIR)
utils.ensure_dir_exists(PROCESSED_IMAGES_DIR)
utils.ensure_dir_exists(EMBEDDINGS_DIR)

# --- Tải Model (Cached) ---
# Sử dụng cache_resource để model chỉ tải một lần mỗi session
@st.cache_resource
def load_model_cached(model_name, device):
    logger.info("Bắt đầu tải model (có thể mất vài phút lần đầu)...")
    start_time = time.time()
    try:
        model, processor = embedding.load_model_processor(model_name, device)
        end_time = time.time()
        logger.info(f"Tải model hoàn tất trong {end_time - start_time:.2f} giây.")
        return model, processor
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi tải model: {e}. Không thể tiếp tục. Vui lòng kiểm tra cài đặt và log.")
        logger.critical(f"Lỗi nghiêm trọng khi tải model: {e}", exc_info=True)
        st.stop() # Dừng ứng dụng nếu không tải được model

model, processor = load_model_cached(MODEL_NAME, DEVICE)

# --- Giao diện người dùng (UI) ---
st.title("Ứng dụng Tìm kiếm Tài liệu bằng Hình ảnh/Văn bản")
st.markdown(f"Sử dụng model `{MODEL_NAME}` trên `{DEVICE}`.")
st.markdown("---")

# --- Phần Sidebar ---
with st.sidebar:
    # --- Phần điều khiển và Upload ---
    st.header("Điều khiển & Upload")
    # button delete all embeddings
    if st.button("Xóa tất cả embeddings và ảnh đã xử lý"):
        deleted_embeddings = 0
        deleted_images = 0
        if EMBEDDINGS_DIR.is_dir():
            for emb_file in EMBEDDINGS_DIR.glob('*.pt'):
                try:
                    emb_file.unlink(missing_ok=True)
                    deleted_embeddings += 1
                except Exception as e:
                    st.warning(f"Không thể xóa embedding: {emb_file.name} - {e}")
        if PROCESSED_IMAGES_DIR.is_dir():
             for img_file in PROCESSED_IMAGES_DIR.iterdir():
                 if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    try:
                        img_file.unlink(missing_ok=True)
                        deleted_images += 1
                    except Exception as e:
                         st.warning(f"Không thể xóa ảnh: {img_file.name} - {e}")

        st.success(f"Đã xóa {deleted_embeddings} embeddings và {deleted_images} ảnh đã xử lý.")
        # Force rerun để cập nhật hiển thị ảnh
        st.rerun()


    st.subheader("1. Upload & Xử lý Tài liệu")
    uploaded_files = st.file_uploader(
        "Tải lên file PDF hoặc Ảnh (PNG, JPG, WEBP)",
        type=['pdf', 'png', 'jpg', 'jpeg', 'webp'],
        accept_multiple_files=True
    )

    process_button = st.button("Xử lý Files Đã Tải Lên")

    # --- Logic xử lý file (như cũ) ---
    if process_button and uploaded_files:
        total_files = len(uploaded_files)
        st.info(f"Bắt đầu xử lý {total_files} file...")
        processed_count = 0
        errored_files = []
        start_process_time = time.time()

        progress_bar = st.progress(0)
        status_text = st.empty()

        all_processed_image_paths_this_run = [] # Lưu lại path ảnh được xử lý trong lần chạy này

        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            status_text.text(f"Đang xử lý file {i+1}/{total_files}: {file_name}...")
            logger.info(f"Đang xử lý file: {file_name}")

            try:
                # 1. Lưu file gốc
                upload_path = UPLOADS_DIR / utils.sanitize_filename(file_name)
                with open(upload_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                logger.info(f"Đã lưu file gốc vào: {upload_path}")

                processed_image_paths_for_file = [] # Path ảnh từ file hiện tại
                # 2. Tiền xử lý
                if Path(file_name).suffix.lower() == ".pdf":
                    pdf_paths = preprocessing.convert_pdf_to_images(upload_path, PROCESSED_IMAGES_DIR)
                    if pdf_paths:
                        processed_image_paths_for_file.extend(pdf_paths)
                    else:
                        logger.warning(f"Không thể xử lý PDF: {file_name}. Bỏ qua.")
                        errored_files.append(file_name + " (PDF processing failed)")

                elif Path(file_name).suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    img_path = preprocessing.save_uploaded_image(uploaded_file, PROCESSED_IMAGES_DIR)
                    if img_path:
                        processed_image_paths_for_file.append(img_path)
                    else:
                        logger.warning(f"Không thể lưu ảnh: {file_name}. Bỏ qua.")
                        errored_files.append(file_name + " (Image saving failed)")
                else:
                    logger.warning(f"Định dạng file không hỗ trợ: {file_name}. Bỏ qua.")
                    errored_files.append(file_name + " (Unsupported format)")

                all_processed_image_paths_this_run.extend(processed_image_paths_for_file) # Thêm vào list tổng

                # 3. Tạo và Lưu Embeddings
                num_images_to_embed = len(processed_image_paths_for_file)
                for j, img_path in enumerate(processed_image_paths_for_file):
                    status_text.text(f"File {i+1}/{total_files}: {file_name} - Tạo embedding cho ảnh {j+1}/{num_images_to_embed} ({img_path.name})...")
                    try:
                        pil_image = Image.open(img_path).convert("RGB") # Đảm bảo RGB
                        img_embedding = embedding.get_image_embedding(pil_image, model, processor, DEVICE, MODEL_NAME)

                        if img_embedding is not None:
                            emb_path = utils.get_embedding_path(img_path.name, EMBEDDINGS_DIR)
                            torch.save(img_embedding.cpu(), emb_path)
                            logger.info(f"Đã lưu embedding cho {img_path.name} vào {emb_path}")
                            processed_count += 1 # Chỉ tăng khi tạo embedding thành công
                        else:
                            logger.error(f"Không thể tạo embedding cho ảnh: {img_path.name}. Bỏ qua.")
                            errored_files.append(img_path.name + " (Embedding failed)")

                    except Exception as emb_err:
                        logger.error(f"Lỗi khi tạo embedding cho {img_path.name}: {emb_err}", exc_info=True)
                        errored_files.append(img_path.name + f" (Embedding error: {emb_err})")

            except Exception as file_err:
                logger.error(f"Lỗi nghiêm trọng khi xử lý file {file_name}: {file_err}", exc_info=True)
                errored_files.append(file_name + f" (Processing error: {file_err})")

            # Cập nhật progress bar sau mỗi file gốc
            progress_bar.progress((i + 1) / total_files)

        end_process_time = time.time()
        status_text.text("Hoàn tất xử lý!")
        st.success(f"Xử lý hoàn tất {total_files} files trong {end_process_time - start_process_time:.2f} giây. Đã tạo {processed_count} embeddings.")
        if errored_files:
            st.warning("Một số files/ảnh gặp lỗi trong quá trình xử lý:")
            st.json(errored_files) # Hiển thị danh sách lỗi

        # --- CHANGE START: Rerun để cập nhật danh sách ảnh hiển thị ---
        # Sau khi xử lý xong, chạy lại script để phần hiển thị ảnh được cập nhật
        if all_processed_image_paths_this_run: # Chỉ rerun nếu có xử lý ảnh mới
            st.rerun()
        # --- CHANGE END ---

    elif process_button and not uploaded_files:
        st.warning("Vui lòng tải lên ít nhất một file.")

    # --- CHANGE START: Phần hiển thị ảnh đã xử lý ---
    st.markdown("---")
    st.header("Ảnh Đã Xử Lý")
    st.caption(f"Hiển thị ảnh từ: `{PROCESSED_IMAGES_DIR}`")

    # Quét thư mục processed_images
    processed_image_files = []
    if PROCESSED_IMAGES_DIR.is_dir():
        processed_image_files = sorted(
            [f for f in PROCESSED_IMAGES_DIR.iterdir()
             if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']],
            key=lambda x: x.name # Sắp xếp theo tên file
        )

    if processed_image_files:
        # Sử dụng expander để không chiếm quá nhiều không gian ban đầu
        with st.expander(f"Hiện có {len(processed_image_files)} ảnh đã xử lý", expanded=False):
             # Giới hạn số lượng ảnh hiển thị trực tiếp để tránh làm chậm sidebar
            max_images_to_show_in_sidebar = 20
            for i, img_path in enumerate(processed_image_files):
                if i >= max_images_to_show_in_sidebar:
                    st.caption(f"... và {len(processed_image_files) - max_images_to_show_in_sidebar} ảnh khác.")
                    break
                try:
                    # Hiển thị ảnh với chiều rộng vừa với container sidebar
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
                except Exception as e:
                    st.warning(f"Lỗi khi hiển thị ảnh {img_path.name}: {e}")
                st.markdown("---") # Thêm dòng kẻ phân cách giữa các ảnh (tùy chọn)
    else:
        st.info("Chưa có ảnh nào trong thư mục xử lý.")
    # --- CHANGE END ---


# --- Phần Retrieval (Main Area - giữ nguyên như cũ) ---
st.header("2. Tìm kiếm Tài liệu")

search_type = st.radio("Tìm kiếm bằng:", ("Văn bản", "Hình ảnh"), key="search_type")

query_text = None
query_image_file = None
query_image_pil = None

if search_type == "Văn bản":
    query_text = st.text_input("Nhập câu hỏi hoặc mô tả:")
else:
    query_image_file = st.file_uploader("Tải lên ảnh để tìm kiếm:", type=['png', 'jpg', 'jpeg', 'webp'], key="query_image")
    if query_image_file:
        try:
            query_image_pil = Image.open(query_image_file).convert("RGB")
            st.image(query_image_pil, caption="Ảnh dùng để tìm kiếm", width=200)
        except Exception as e:
            st.error(f"Không thể đọc ảnh query: {e}")
            query_image_pil = None # Đảm bảo không tìm kiếm nếu ảnh lỗi


k_value = st.number_input("Số lượng kết quả (K):", min_value=1, max_value=100, value=5, step=1, key="k_value")

search_button = st.button("Tìm kiếm", key="search_button")

st.markdown("---")
st.subheader("Kết quả tìm kiếm:")

if search_button:
    query_embedding = None
    valid_query = False

    # --- Kiểm tra và Tạo Query Embedding ---
    if search_type == "Văn bản" and query_text:
        valid_query = True
        with st.spinner("Đang tạo embedding cho câu hỏi..."):
            query_embedding = embedding.get_text_embedding(query_text, model, processor, DEVICE, MODEL_NAME)
        if query_embedding is None:
            st.error("Không thể tạo embedding cho câu hỏi. Vui lòng thử lại.")
            valid_query = False

    elif search_type == "Hình ảnh" and query_image_pil:
        valid_query = True
        with st.spinner("Đang tạo embedding cho ảnh query..."):
            query_embedding = embedding.get_image_embedding(query_image_pil, model, processor, DEVICE, MODEL_NAME)
        if query_embedding is None:
            st.error("Không thể tạo embedding cho ảnh query. Vui lòng thử lại.")
            valid_query = False

    elif (search_type == "Văn bản" and not query_text) or \
         (search_type == "Hình ảnh" and not query_image_pil):
        st.warning("Vui lòng nhập câu hỏi hoặc tải lên ảnh để tìm kiếm.")

    # --- Thực hiện Tìm kiếm nếu Query hợp lệ ---
    if valid_query and query_embedding is not None:
        with st.spinner(f"Đang tìm kiếm {k_value} kết quả phù hợp nhất..."):
            search_start_time = time.time()

            # 1. Tải tất cả embeddings đã lưu
            all_embeddings, embedding_filenames = retrieval.load_all_embeddings(EMBEDDINGS_DIR, DEVICE)

            if not all_embeddings or not embedding_filenames: # Kiểm tra cả hai
                st.warning("Chưa có tài liệu nào được xử lý và lưu embedding. Vui lòng upload và xử lý tài liệu trước.")
            else:
                # 2. Tìm kiếm Top-K
                # Chuyển query_embedding lên đúng device trước khi tìm kiếm
                results = retrieval.find_top_k(
                    query_embedding.to(DEVICE), # Đảm bảo query embedding trên đúng device
                    all_embeddings,       # Đã được load lên device
                    embedding_filenames,
                    k_value # Truyền K value vào hàm find_top_k (Cần đảm bảo hàm này nhận K)
                            # Lưu ý: Hàm find_top_k gốc có thể không cần processor, kiểm tra lại định nghĩa hàm
                )
                search_end_time = time.time()
                logger.info(f"Tìm kiếm hoàn tất trong {search_end_time - search_start_time:.2f} giây.")

                # 3. Hiển thị kết quả
                if not results:
                    st.info("Không tìm thấy kết quả nào phù hợp.")
                else:
                    st.success(f"Tìm thấy {len(results)} kết quả:")
                    # Chia thành các cột để hiển thị đẹp hơn
                    num_columns = 3 # Số cột hiển thị kết quả
                    cols = st.columns(num_columns)
                    for i, (score, emb_filename) in enumerate(results):
                        # Lấy tên file ảnh gốc từ tên file embedding
                        # (Giả sử hàm utils.get_image_path_from_embedding làm điều này)
                        image_path = utils.get_image_path_from_embedding(emb_filename, PROCESSED_IMAGES_DIR)
                        if image_path and image_path.is_file():
                            col_index = i % num_columns
                            with cols[col_index]:
                                st.image(str(image_path), use_container_width=True)
                                st.caption(f"Tên file: {image_path.name}")
                                st.caption(f"Score: {score:.4f}")
                                st.markdown("---") # Ngăn cách giữa các kết quả trong cùng cột
                        else:
                             # Tìm lại tên ảnh gốc nếu hàm get_image_path_from_embedding không hoạt động đúng
                            possible_image_name = emb_filename.replace(".pt", "") # Tên file ảnh thường là tên embedding bỏ .pt
                            potential_image_path = PROCESSED_IMAGES_DIR / possible_image_name
                            if potential_image_path.is_file():
                                 col_index = i % num_columns
                                 with cols[col_index]:
                                    st.image(str(potential_image_path), use_container_width=True)
                                    st.caption(f"Tên file: {potential_image_path.name}")
                                    st.caption(f"Score: {score:.4f}")
                                    st.markdown("---")
                            else:
                                logger.warning(f"Không tìm thấy file ảnh gốc cho embedding: {emb_filename} tại đường dẫn {potential_image_path} hoặc {image_path}")
                                st.warning(f"Không tìm thấy file ảnh cho embedding: {emb_filename}")