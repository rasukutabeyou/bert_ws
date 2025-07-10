# 言語モデル"BERT"による話者の分類

## 概要
BERTのファインチューニングを行い、話者の分類を行う。

- `main`:ファインチューニングを行う`bert_Fchuning.py`、ファインチューニングで得られたモデルを使って推論を行う`hyouka.py`、モデル実行用の`Fchuning_model.py`が存在。
- `setup_tools`:データの整列を行う`Tabidachi_sort_text.py `、イレギュラー対応用の`Tabidachi_sort_part_text.py`データのエラー確認用の`test.py`、train,val,testへデータを分割する`create_splits.py`
- `dataset`:整理済みのデータが格納されている。

## 実行方法
以下の手順で環境構築し、実行してください。
```
$conda create -n bert python=3.10
$conda activate bert
$pip install install transformers datasets evaluate accelerate numpy scikit-learn
$cd ~/bert_ws/main
$python bert_Fchuning.py --train ~/bert_ws/dataset/train --val ~/bert_ws/dataset/val
$python Fchuning_model.py 
```
###　実行例
```
$python Fchuning_model.py
発言内容：あの、季節、季節的にはどれくらいの季節がよろしいですか？
予測ラベル:0
```

## 必要ソフト
- python
    - 動作確認済み　python3.10

## 環境
- ubuntu-22.04

## 仮想環境
- Anaconda
    - conda-24.9.2

## その他
- このソフトウェアは、3条項BSDライセンスの下、再頒布および使用が許可されます。
- © 2025 Kouta Sakai