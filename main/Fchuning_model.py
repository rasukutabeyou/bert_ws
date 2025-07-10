from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained("/home/sakai/sakai_dev/bert_ws/history/epoch1/my_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("/home/sakai/sakai_dev/bert_ws/history/epoch1/my_model")

#text = "あの、季節、季節的にはどれくらいの季節がよろしいですか？"

def model_predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs[0].tolist()

#pred, hoge = model_predict(text)

#print(f"発言内容: {text}")
#print(f"予測ラベル: {pred}")
