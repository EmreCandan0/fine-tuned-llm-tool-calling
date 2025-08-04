# -*- coding: utf-8 -*-

Original file is located at
    https://colab.research.google.com/drive/1CbHv3SRTTfvC9bH0DOux15Vt9WrH_VfL
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     # Do this only in Colab notebooks! Otherwise use pip install unsloth
#     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" huggingface_hub hf_transfer
#     !pip install --no-deps unsloth
# !pip install protobuf==3.20.3 # required
# !pip install --no-deps transformers-cfg

from unsloth import FastQwen2Model
import torch

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastQwen2Model.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length=None,
    dtype=None,
    load_in_4bit=False,
    fix_tokenizer=False
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastQwen2Model.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from datasets import load_dataset

dataset = load_dataset('json', data_files='/content/toolcalling_dataset.jsonl', split='train')
print(dataset[0])

def convert_to_sharegpt_format(dataset):
    """Convert dataset with 'messages' field to ShareGPT format"""
    new_dataset = []
    for example in dataset:
        # Use the existing messages structure
        conversations = example["messages"]
        new_dataset.append({
            "conversations": conversations
        })
    return new_dataset

from datasets import Dataset
dataset_sharegpt = convert_to_sharegpt_format(dataset)
hf_dataset = Dataset.from_list(dataset_sharegpt)

print("ShareGPT format sample:")
print(hf_dataset[0])

from unsloth import standardize_sharegpt
dataset_standardized = standardize_sharegpt(hf_dataset)

print("Standardized dataset sample:")
print(dataset_standardized[0])

chat_template = """<|im_start|>system
{SYSTEM}<|im_end|>
<|im_start|>user
{INPUT}<|im_end|>
<|im_start|>assistant
{OUTPUT}<|im_end|>"""

# Apply chat template
from unsloth import apply_chat_template

dataset_final = apply_chat_template(
    dataset_standardized,
    tokenizer=tokenizer,
    chat_template=chat_template,
    # default_system_message = "You are a helpful assistant", # [OPTIONAL]
)

print("Final dataset ready for training!")
print(f"Dataset size: {len(dataset_final)}")
print("Sample text:")
print(dataset_final[0]['text'])

print(dataset_final[213]['text'])  #test if the format changed

from trl import SFTTrainer, SFTConfig #trainer configs
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_final,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 50,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer_stats = trainer.train()  #train the model

from transformers import TextStreamer


model = FastQwen2Model.for_inference(model)

# Messages must be in ChatML/ShareGPT format
messages = [
    {
        'role': 'user',
        'content': 'Analyze the file C:/Users/emre/Desktop/abc.tif'
    }
]

# ChatML tokenizer to create the prompt
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,      # "assistant" adds <|im_start|>assistant to the end of the prompt
    return_tensors="pt"
).to(model.device)                   # "cuda" or "cpu"

streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids,
    streamer=streamer,
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id
)
##The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
##TOOL_NEEDED: analyze_tiff
##PARAMS: {"filepath": "C:/Users/emre/Desktop/abc.tif"}<|im_end|> that was the output

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")

from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_id = "unsloth/Qwen2.5-Coder-1.5B-Instruct"   #Or the model you used
adapter_path = "/content/lora_model"         # Adapter path

# 1. Base model
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="cpu")

# 2. Adapter and merge
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

# 3.New merged model
model.save_pretrained("merged_model")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained("merged_model")

!zip -r /content/adapter.zip /content/lora_model

!cp /content/adapter.zip /content/drive/MyDrive/

!zip -r /content/merged_model.zip /content/merged_model

!cp /content/merged_model.zip /content/drive/MyDrive/
