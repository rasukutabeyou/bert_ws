import pandas as pd
import os 
import csv
import re
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()

parser.add_argument("--input_file",type=str,default="~/sakai_dev/hoge/119_6_5_zoom.tsv")
parser.add_argument("--output_dir",type=str,default="piyo_csv")
parser.add_argument("--file_name",type=str,default="hoge_file")
args = parser.parse_args()
for k, v in vars(args).items():
        print(f"{k}: {v}")


df = pd.read_csv(args.input_file, sep='\t')

#時間を秒に変換
def time_str_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

#話者を0,1に変換
def convert_speaker_to_fig(speaker):
    if speaker == "オペレーター":
        return int(0)
    elif speaker == "オペレータ":
         return int(0)
    elif speaker == "カスタマー":
        return int(1)
    else:
        return "hoge"

def arrange_text(text_data):
    # try:
         text_data = re.sub(r"[<>＜＞]","",text_data)
         text_data = re.sub(r"\b[a-zA-Z]\b","",text_data)
         return text_data
    # except:
          #pass
     

# 各列を変数に格納
name = df['発話者名'].apply(convert_speaker_to_fig)
start_time = df['発話開始時間'].apply(time_str_to_seconds)
end_time = df['発話終了時間'].apply(time_str_to_seconds)
text = df['書き起こしテキスト'].apply(arrange_text)

# 確認（例：最初の5件を表示）
#print(name.head())
#print(start_time.head())
#print(end_time.head())
#print(text.head())
all_array= [[f"{text[1]}",name[1]]]
start_time_max = start_time[1]
end_time_max = end_time[1]

for index, row in df.iloc[2:].iterrows():
    if (end_time_max < start_time[index] or end_time_max < end_time[index]) and (name[index] == 0 or name[index] == 1) and text[index] and str(text[index]).strip() != "。":
        end_time_max = end_time[index]
        hoge = [f"{text[index]}",name[index]]
        all_array.append(hoge)

Path(args.output_dir).mkdir(parents=True,exist_ok=True)
output_filename = f"{args.file_name}" + ".csv"
df = pd.DataFrame(all_array,columns=["text","label"])
df.to_csv(f"{args.output_dir}/{output_filename}" ,index = False,encoding = "utf-8")