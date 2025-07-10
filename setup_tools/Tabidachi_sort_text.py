import os
import csv
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--output_dir", type=str, default="piyo_csv")
args = parser.parse_args()
for k, v in vars(args).items():
        print(f"{k}: {v}")

# ====== 関数定義 ======

# 時間を秒に変換
def time_str_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

# 話者を0,1に変換
def convert_speaker_to_fig(speaker_fig):
    if speaker_fig == "オペレータ":
        return 0
    elif speaker_fig == "オペレーター":
        return 0
    elif speaker_fig == "カスタマ―":
        return 1
    elif speaker_fig == "カスタマー":
        return 1
    else:
        return "error"
    
def arrange_text(text_data):
        try:
            text_data = re.sub(r"[<>＜＞]","",text_data)
            text_data = re.sub(r"\b[a-zA-Z]\b","",text_data)
            return text_data
        except:
            pass


# ====== ディレクトリの探索と処理 ======

# データの親ディレクトリ
parent_dir = '/autofs/diamond3/share/corpus/Tabidachi/extracted/Tabidachi2109-3'

# すべてのディレクトリを取得
all_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

# 先頭2つの番号でグループ化
grouped_dirs = defaultdict(list)
for d in all_dirs:
    parts = d.split('_')
    if len(parts) >= 2:
        key = f"{parts[0]}_{parts[1]}"
        grouped_dirs[key].append(d)

# 各グループごとに処理
for key, dirs in grouped_dirs.items():
    #print(f"\n=== Processing group: {key} ===")
    
    for d in dirs:
        tsv_file = os.path.join(parent_dir, d, f"{d}_zoom.tsv")
        if not os.path.isfile(tsv_file):
            print(f"  [!] File not found: {tsv_file}")
            continue

        try:
            df = pd.read_csv(tsv_file, sep='\t')
        except Exception as e:
            print(f"  [!] Failed to read {tsv_file}: {e}")
            continue

        # データ変換
        try:
            name = df['発話者名'].apply(convert_speaker_to_fig)
            start_time = df['発話開始時間'].apply(time_str_to_seconds)
            end_time = df['発話終了時間'].apply(time_str_to_seconds)
            text = df['書き起こしテキスト'].apply(arrange_text)
        except KeyError as e:
            print(f"  [!] Missing column in {tsv_file}: {e}")
            continue

        if len(df) < 2:
            print("  [!] Too few rows in file")
            continue

        # 処理本体
        all_allay = [[f"{text[1]}", name[1]]]
        start_time_max = start_time[1]
        end_time_max = end_time[1]

        for index, row in df.iloc[2:].iterrows():
            if (end_time_max < start_time[index] or end_time_max < end_time[index]) and (name[index] == 0 or name[index] == 1) and text[index] and str(text[index]).strip() != "。":
                end_time_max = end_time[index]
                hoge = [f"{text[index]}", name[index]]
                all_allay.append(hoge)
                
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_file_name = d + ".csv"
        df = pd.DataFrame(all_allay,columns = ["text","label"])
        df.to_csv(f"{args.output_dir}/{output_file_name}",index = False,encoding="utf-8")