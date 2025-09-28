import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
import string
from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig

# root_image_dir = "D:/fzh/15-EEG/Neural-MCRL-main/test_images" 
root_image_dir = "C:/fzh/MEG/THINGSfMRI/THINGS-fMRI/images"
output_pkl = "fMRI_4_item_text_train_test_mix.pkl"  
local_model_path = "D:/fzh/15-EEG/Neural-MCRL-main/BLIP_checkpoint"

prompts = [
    "this is a picture of",
    "Question: Describe this picture in three words and only words. Answer:",
    "Question: Describe the mental feeling of this picture in one word. Answer:",
    "Question: Describe the location of this picture in one word. Answer:"
]

def clean_text(generated_text, prompt):
    if generated_text.startswith(prompt):
        cleaned = generated_text[len(prompt):].strip()
    else:
        cleaned = generated_text.strip()
    translator = str.maketrans('', '', string.punctuation)
    cleaned = cleaned.translate(translator)
    return cleaned.lower()
processor = AutoProcessor.from_pretrained(local_model_path, use_fast=True)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    local_model_path,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)
all_results = []
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
image_files = [
    f for f in os.listdir(root_image_dir)
    if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(root_image_dir, f))
]
image_files.sort() 
for img_file in image_files:
    img_path = os.path.join(root_image_dir, img_file)
    image = Image.open(img_path).convert("RGB")
    img_results = [] 
    prefixes = [
        "this is ",
        "this is ",
        "It's ",
        "It's in "
    ]
    for i, prompt in enumerate(prompts):
        inputs = processor(
            image,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True
        )
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        cleaned_text = clean_text(generated_text, prompt)
        prefixed_text = prefixes[i] + cleaned_text
        print(prefixed_text)
        img_results.append(prefixed_text)

    all_results.append([img_results])
with open(output_pkl, 'wb') as f:
    pickle.dump(all_results, f)
    