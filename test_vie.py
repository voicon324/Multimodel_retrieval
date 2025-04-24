import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import matplotlib.pyplot as plt

model_name = "5CD-AI/ColVintern-1B-v1"

processor =  AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().cuda()

#!wget https://huggingface.co/5CD-AI/ColVintern-1B-v1/resolve/main/ex1.jpg
#!wget https://huggingface.co/5CD-AI/ColVintern-1B-v1/resolve/main/ex2.jpg

images = [Image.open("ex1.jpg"),Image.open("ex2.jpg")]
batch_images = processor.process_images(images)

queries = [
    "Cảng Hải Phòng thông báo gì ?",
    "Phí giao hàng bao nhiêu ?",
]

batch_queries = processor.process_queries(queries) 

batch_images["pixel_values"] =  batch_images["pixel_values"].cuda().bfloat16()
batch_images["input_ids"] = batch_images["input_ids"].cuda() 
batch_images["attention_mask"] = batch_images["attention_mask"].cuda().bfloat16()
batch_queries["input_ids"] = batch_queries["input_ids"].cuda() 
batch_queries["attention_mask"] = batch_queries["attention_mask"].cuda().bfloat16()

with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)

max_scores, max_indices = torch.max(scores, dim=1)
# In ra kết quả cho mỗi câu hỏi
for i, query in enumerate(queries):
    print(f"Câu hỏi: '{query}'")
    print(f"Điểm số: {max_scores[i].item()}\n")
    plt.figure(figsize=(5,5))
    plt.imshow(images[max_indices[i]])
    plt.show()

from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("5CD-AI/ColVintern-1B-v1", trust_remote_code=True)
mdl = AutoModel.from_pretrained("5CD-AI/ColVintern-1B-v1", trust_remote_code=True)
print("Loaded OK ✅")