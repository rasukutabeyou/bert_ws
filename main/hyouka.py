import os
import glob
import pandas as pd
from argparse import ArgumentParser
from Fchuning_model import model_predict
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--test",type=str,default="dataset/test")

args = parser.parse_args()

files = glob.glob(f"{args.input}/*.csv")
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
count = len(df["label"])
label_0 = 0
label_1 = 0
result_0 = 0
result_1 = 0

for i in tqdm(range(count), desc="推論中"):
    if df["label"][i] == 1:
        label_1 += 1
    if df["label"][i] == 0:
        label_0 += 1
    text = df["text"][i]
    pred , probs = model_predict(text)
    if pred == df["label"][i]:
        if pred == 0:
            result_0 += 1
        if pred == 1:
            result_1 += 1

print(f"話者0の推論成功率：{result_0/label_0*100:.2f}%({result_0}/{label_0})")
print(f"話者1の推論成功率：{result_1/label_1*100:.2f}%({result_1}/{label_1})")
