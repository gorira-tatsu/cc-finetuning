from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
import wandb

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", use_fast=False)
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

# ① pad_token を eos_token に設定
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
model.config.pad_token_id = tokenizer.pad_token_id

# ② データセット読み込み & 10% を validation に分割
raw_dataset = load_dataset("text", data_files={"train": "jp.txt"})["train"]
split_datasets = raw_dataset.train_test_split(test_size=0.1)

# トークナイズ関数（長文対応のため truncation/max_length を外す）
def tokenize_function(examples):
    return tokenizer(examples["text"])

# 長文を block_size 単位に分割するヘルパー
block_size = 1024
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# トークナイズ → チャンク分割（train/validation）
tokenized_train = split_datasets["train"].map(
    tokenize_function, batched=True, remove_columns=["text"]
).map(group_texts, batched=True)
tokenized_validation = split_datasets["test"].map(
    tokenize_function, batched=True, remove_columns=["text"]
).map(group_texts, batched=True)

# ④ DataCollator に pad_to_multiple_of を指定
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# Wandb 初期化（API キーは環境変数か事前ログインで設定）
wandb.login()
wandb.init(project="cc-finetuning", config={
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8
})

# 学習設定
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    do_eval=True,
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    report_to="wandb",
)

# Trainer 初期化 & 学習実行
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    data_collator=data_collator,
)
trainer.train()
