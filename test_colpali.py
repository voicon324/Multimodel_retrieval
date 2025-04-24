import torch
from PIL import Image

from colpali_engine.models import ColIdefics3, ColIdefics3Processor

model = ColIdefics3.from_pretrained(
        "vidore/colSmol-256M",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager" # or eager
    ).eval()
processor = ColIdefics3Processor.from_pretrained("vidore/colSmol-256M")

# Your inputs
images = [
    Image.open("chart.png").convert("RGB"),
    # Image.open("creative_process.webp").convert("RGB"),
    Image.new("RGB", (32, 32), color="black"),
    # Image.new("RGB", (32, 32), color="black"),
]
queries = [
    "How many of IT's employees are in the office?",
    # "What is the creative process?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)