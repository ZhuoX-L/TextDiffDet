import json
import jieba
from bert_pytorch.dataset import WordVocab

TRAIN_JSON = "/root/autodl-tmp/DiffusionDet/coco/train/annotations.json"

with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

texts = []
for img in coco["images"]:
    report = img.get("report", "").strip()
    if report:
        words = jieba.lcut(report)
        texts.append(" ".join(words))

print(f"[INFO] Collected {len(texts)} training reports")

vocab = WordVocab(
    texts,
    max_size=30000,
    min_freq=2   # 医学文本建议 >=2，过滤噪声
)

VOCAB_PATH = "/root/autodl-tmp/DiffusionDet/medical_vocab.pkl"
vocab.save_vocab(VOCAB_PATH)

print(f"[INFO] Vocab size = {len(vocab)}")

