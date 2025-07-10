from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
import evaluate
import numpy as np
import glob
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--train_dir",type=str,default="train")
parser.add_argument("--val_dir",type=str,default="val")
args = parser.parse_args()
for k, v in vars(args).items():
    print(f"{k}:{v}")

# モデル指定（日本語BERT）
model_name = "llm-jp/llm-jp-modernbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 🔹 複数CSVファイルをまとめて読み込む関数
def load_multiple_csv(path_pattern):
    files = glob.glob(path_pattern)
    dataset_list = [load_dataset("csv", data_files=file)["train"] for file in files]
    return concatenate_datasets(dataset_list)

# 🔸 train/val の複数ファイルをロード
train_dataset = load_multiple_csv(f"{args.train_dir}/*.csv")
val_dataset = load_multiple_csv(f"{args.val_dir}/*.csv")

# 🔸 トークナイズ処理
def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

tokenized_train = train_dataset.map(preprocess, batched=True)
tokenized_val = val_dataset.map(preprocess, batched=True)

# 🔸 評価指標（accuracy）
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 🔸 学習設定
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
)

# 🔸 Trainerで学習
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# 🔸 ファインチューニング実行
trainer.train()

model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

metrics = trainer.evaluate()

with open("eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
