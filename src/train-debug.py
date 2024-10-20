from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration,BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
import sys
import time
from peft import get_peft_model, LoraConfig


# --------- Training Parameters ---------
batch_size = 1
grad_accum_steps = 16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_epochs = 1
num_warmup_steps = 0
initial_lr = 5e-5
# -----------------------------------------

# LORa config
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
)

lora_config = LoraConfig(
    r=8, 
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# -----------------------------------------

hf_token = ""
model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id, token=hf_token)
dataset = load_dataset("nickhobbs09/screen-ai-annotation", token=hf_token)
dtype = torch.bfloat16

# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, token=hf_token, device_map={"":0},  quantization_config=bnb_config)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)

# # Freeze vision tower
for param in model.vision_tower.parameters():
    param.requires_grad = False

# Freeze multi_modal_projector
for param in model.multi_modal_projector.parameters():
    param.requires_grad = True
    
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

train_ds = dataset['train']
val_ds = dataset['test']
PROMPT = "Summarize"

# # FOR DEBUGGING
NUM_SAMPLES = 16*64*2
train_ds = train_ds.select(range(NUM_SAMPLES))
# val_ds = val_ds.select(range(NUM_SAMPLES))
# -----------------------------------------

def collate_fn(examples):
    texts = [PROMPT for example in examples]
    labels = [example["screen_annotation"] for example in examples]
    images = [example["image"] for example in examples]

    tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")
    tokens = tokens.to(dtype).to(device)
    # inputs = {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": tokens.labels}
    return tokens

train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
# Optimizer and scheduler
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=initial_lr)

from transformers import get_scheduler

num_training_steps = num_epochs * (len(train_ds)//batch_size)
print("Number of training steps: ", num_training_steps)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

iter_dataloader = iter(train_dataloader)

image1 = train_ds[0]['image']
annotation1 = train_ds[0]['screen_annotation']

model.train()

# # Simple training loop
for i, tokens in enumerate(train_dataloader):
    optimizer.zero_grad()
    outputs = model(**tokens)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    if i % 256 == 0:
        print("Pushing to hub...")
        model.push_to_hub("nickhobbs09/screen-ai-annotation-finetuned-paligemma-3b-pt-224", private=True, token=hf_token)
    print(f'step {i: 5d} | loss: {loss.item():.6f}')

print("Pushing to hub...")
model.push_to_hub("nickhobbs09/screen-ai-annotation-finetuned-paligemma-3b-pt-224", private=True)

# # Make a prediction
# model.eval()
# for i, data in enumerate(val_ds.select(range(3))):
#     image = data['image']
#     image.save(f"./val_data/val_image_{i}.png")
#     annotation = data['screen_annotation']
#     input = processor(images=image, text="Summarize", return_tensors="pt").to(device)
#     output = model.generate(**input, max_new_tokens=256)
#     print(processor.decode(output[0], skip_special_tokens=True))


# import sys; sys.exit()

# # Example training loop
# for i, tokens in enumerate(train_dataloader):
#   t0 = time.time()
#   # Model Optimization
#   model.train()
#   optimizer.zero_grad()
#   loss_accum = 0
#   num_tokens = 0
#   for j in range(grad_accum_steps):
#     outputs = model(**tokens)
#     loss = outputs.loss
#     loss = loss / grad_accum_steps
#     loss_accum += loss.detach()
#     num_tokens += tokens.input_ids.ne(processor.tokenizer.pad_token_id).sum().item()
#     loss.backward()
#   norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#   optimizer.step()  
#   lr = lr_scheduler.get_lr() # Figure out how to print learning rate
#   lr_scheduler.step()
#   if device == "cuda":
#     torch.cuda.synchronize() # wait for the GPU to finish work
#   t1 = time.time()
#   dt = (t1-t0) * 1000
#   print(f'step {i: 5d} | loss: {loss_accum.item():.6f} | time: {dt:.2f}ms | tokens_per_sec: {num_tokens / dt:.2f} | lr: {lr}')

# print("Pushing to hub...")
# model.push_to_hub("nickhobbs09/screen-ai-annotation-finetuned-paligemma-3b-pt-224", private=True)
# model.eval()
# input = processor(images=image1, text="Summarize", return_tensors="pt").to(device)
# output = model.generate(**input, max_new_tokens=256)
# print(processor.decode(output[0], skip_special_tokens=True))

# model.push_to_hub("nickhobbs09/screen-ai-annotation-finetuned-paligemma-3b-pt-224", use_auth_token=hf_token)

# args=TrainingArguments(
#         num_train_epochs=1,
#         remove_unused_columns=False,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=16,
#         warmup_steps=2,
#         learning_rate=2e-5,
#         weight_decay=1e-6,
#         adam_beta2=0.999,
#         logging_steps=100,
#         optim="adamw_hf",
#         save_strategy="steps",
#         save_steps=1000,
#         push_to_hub=False,
#         save_total_limit=1,
#         bf16=True,
#         report_to=["tensorboard"],
#         dataloader_pin_memory=False,
#         output_dir='./output'
#         )

# trainer = Trainer(
#         model=model,
#         train_dataset=dataset["train"],
#         eval_dataset=dataset["test"],
#         data_collator=collate_fn,
#         args=args
#         )
# trainer.train()
