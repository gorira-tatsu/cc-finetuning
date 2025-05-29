from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
import wandb

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", use_fast=False)
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

# データセット読み込み（train.txt のみ）＆10%を validation に分割
raw_dataset = load_dataset("text", data_files={"train": "jp.txt"})["train"]
split_datasets = raw_dataset.train_test_split(test_size=0.1)

# トークナイズ関数
def tokenize_function(examples):
    return tokenizer(examples["text"])

# トークナイズ & 不要カラム削除（train/validation）
tokenized_train = split_datasets["train"].map(
    tokenize_function, batched=True, remove_columns=["text"]
)
tokenized_validation = split_datasets["test"].map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# データコラレータ（causal LM 用）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
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
