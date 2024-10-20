from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration,TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
import sys
import time

# --------- Training Parameters ---------
batch_size = 2
grad_accum_steps = 16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_epochs = 1
num_warmup_steps = 0
initial_lr = 5e-5
# -----------------------------------------

model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)
dataset = load_dataset("nickhobbs09/screen-ai-annotation")

train_ds = dataset['train']
val_ds = dataset['test']
PROMPT = "Summarize"

def collate_fn(examples):
    texts = [PROMPT for example in examples]
    labels = [example["screen_annotation"] for example in examples]
    images = [example["image"] for example in examples]

    tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")
    tokens = tokens.to(torch.bfloat16).to(device)
    inputs = {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": tokens.labels}
    return tokens

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

# Freeze vision tower
for param in model.vision_tower.parameters():
    param.requires_grad = False

# Unfreeze multi_modal_projector
for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

# Optimizer and scheduler
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=initial_lr)

from transformers import get_scheduler

num_training_steps = num_epochs * (len(train_ds)/batch_size)
print("Number of training steps: ", num_training_steps)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)


iter_dataloader = iter(train_dataloader)
model.train()
# Example training loop
for i in range(1):
  t0 = time.time()
  # Model Optimization
  model.train()
  optimizer.zero_grad()
  loss_accum = 0
  num_tokens = 0
  for j in range(grad_accum_steps):
    if not iter_dataloader.has_next():
        print("Resetting dataloader")
        # Checkpoint the model
        iter_dataloader = iter(train_dataloader)
    tokens = next(iter_dataloader)
    tokens = tokens.to(device)
    outputs = model(**tokens)
    loss = outputs.loss
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    num_tokens += tokens.input_ids.ne(processor.tokenizer.pad_token_id).sum().item()
    loss.backward()
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  optimizer.step()  
  lr_scheduler.step()
  if device == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
  t1 = time.time()
  dt = (t1-t0) * 1000
  print(f'step {i: 5d} | loss: {loss_accum.item():.6f} | time: {dt:.2f}ms | tokens_per_sec: {num_tokens / dt:.2f}')

model.eval()

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
#         push_to_hub=True,
#         save_total_limit=1,
#         bf16=True,
#         report_to=["tensorboard"],
#         dataloader_pin_memory=False,
#         output_dir='./output'
#     )

# trainer = Trainer(
#         model=model,
#         train_dataset=dataset["train"],
#         eval_dataset=dataset["test"],
#         data_collator=collate_fn,
#         args=args
#         )
# trainer.train()
