#Test using a small VLM (moondream2 ~2B param)
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True
   
)
#caption image

with Image.open("images/image.jpg") as image:
    print("Caption:")
    print(model.caption(image, length="normal")["caption"])
#query on the image
with Image.open("images/image2.jpg") as image2:
    print("How many people are in the image? Give a precise number")
    print(model.query(image2, "How many people are in the image? Give a precise number.")["answer"])