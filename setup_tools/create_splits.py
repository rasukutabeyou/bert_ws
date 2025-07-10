import os
import shutil
import glob
import random
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--input_dir", type=str, default="/home/sakai/sakai_dev/bert_ws/dataset/text_csv")
parser.add_argument("--output_dir", type=str, default="dataset")
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.1)
args = parser.parse_args()

# 引数表示
for k, v in vars(args).items():
    print(f"{k}: {v}")

# ファイルリストの取得とシャッフル
files = glob.glob(f"{args.input_dir}/*.csv")
files = [f for f in files if os.path.isfile(f)]
random.shuffle(files)  # ランダムに並び替え

file_count = len(files)

# 各スプリットのサイズを計算
train_count = int(file_count * args.train_ratio)
val_count = int(file_count * args.val_ratio)
test_count = file_count - train_count - val_count  # 残り

# ディレクトリ作成とコピー処理
split_dirs = {
    "train": files[:train_count],
    "val": files[train_count:train_count + val_count],
    "test": files[train_count + val_count:]
}

for split_name, split_files in split_dirs.items():
    split_path = os.path.join(args.output_dir, split_name)
    os.makedirs(split_path, exist_ok=True)
    for f in split_files:
        shutil.copy(f, split_path)

# 確認用出力
print(f"train: {len(split_dirs['train'])}, val: {len(split_dirs['val'])}, test: {len(split_dirs['test'])}")
