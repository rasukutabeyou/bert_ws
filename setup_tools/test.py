import pandas as pd
import re
import glob
from argparse import ArgumentParser

# 引数のパーサーを設定
parser = ArgumentParser()
parser.add_argument("--input_dir", type=str, default="hoge")
args = parser.parse_args()
for k, v in vars(args).items():
        print(f"{k}: {v}")


# チェック対象のファイル（複数可）
file_list = glob.glob(f"{args.input_dir}/*.csv")

# テキストが「数字だけ」or「空」or「None」or「タグだらけ」などを判定する関数

def is_problematic(text):
    if pd.isnull(text):  # None or NaN
        return True
    text = str(text).strip()

    if text == "":
        return True
    #if text.isdigit():  # 数字だけ
        #return True
    if re.fullmatch(r"[<>＜＞]", text):  # タグだけ
        return True
    if re.fullmatch(r"。+", text):
        return True
    return False

# 各ファイルごとにチェック
for filepath in file_list:
    #print(f"🔍 ファイル: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8")

    for i, row in df.iterrows():
        if "text" not in row:
            print(f"🔍 ファイル: {filepath}")
            print(f"  ❌ {i+1}行目：text列が存在しません")
            continue

        if is_problematic(row["text"]):
            print(f"🔍 ファイル: {filepath}")
            print(f"  ⚠ {i+1}行目に問題あり: 「{row['text']}」")

    #print("✅ 完了\n")