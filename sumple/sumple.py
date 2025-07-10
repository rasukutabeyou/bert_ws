from transformers import AutoTokenizer, AutoModel
import torch

# 1. モデルとトークナイザーの読み込み
model_name = "llm-jp/llm-jp-modernbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 入力する日本語テキスト
text = "こんにちは。"

# 3. テキストをトークン化 → モデル用のテンソルに変換
inputs = tokenizer(text, return_tensors="pt")

# 4. モデルに入力して出力を得る（勾配不要モード）
with torch.no_grad():
    outputs = model(**inputs)

# 5. 出力（last_hidden_state）はトークンごとのベクトル
last_hidden_state = outputs.last_hidden_state  # shape: [1, トークン数, 768]

# 6. [CLS]トークンのベクトルだけを取り出す（文全体の特徴）
cls_vector = last_hidden_state[:, 0, :]  # shape: [1, 768]

# 結果を確認
print(cls_vector)