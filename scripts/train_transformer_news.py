from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fairy_core.news_intelligence import CONFIG_PATH, LABELS, MODEL_DIR, MODEL_PATH, VOCAB_PATH, CharTokenizer, TransformerNewsClassifier


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "news_train.jsonl")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen = set()

    for path in paths:
        if not os.path.exists(path):
            print(f"跳过不存在的数据文件: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                text = row.get("text", "").strip()
                label = row.get("label", "").strip()
                key = (text, label)
                if not text or not label or key in seen:
                    continue

                rows.append({"text": text, "label": label})
                seen.add(key)
    return rows


def build_vocab(texts: list[str], min_freq: int = 1) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(list(text.strip()))

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


class NewsDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer: CharTokenizer, label_to_id: dict[str, int]) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        input_ids, attention_mask = self.tokenizer.encode(row["text"])
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": torch.tensor(self.label_to_id[row["label"]], dtype=torch.long),
        }


def evaluate(model: TransformerNewsClassifier, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data_paths = args.data_path or [DATA_PATH]
    rows = load_dataset(data_paths)
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}

    if len(rows) < 8:
        print(f"训练样本不足，当前仅有 {len(rows)} 条，至少需要 8 条。")
        return

    label_counts = Counter(row["label"] for row in rows)
    available_labels = [label for label, count in label_counts.items() if count > 0]
    if len(available_labels) < 2:
        print("训练样本类别不足，至少需要两个类别。")
        return

    train_rows, val_rows = train_test_split(
        rows,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[row["label"] for row in rows] if min(label_counts.values()) >= 2 else None,
    )

    vocab = build_vocab([row["text"] for row in train_rows], min_freq=1)
    tokenizer = CharTokenizer(vocab=vocab, max_len=args.max_len)

    train_loader = DataLoader(
        NewsDataset(train_rows, tokenizer, label_to_id),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        NewsDataset(val_rows, tokenizer, label_to_id),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNewsClassifier(
        vocab_size=len(vocab),
        num_classes=len(LABELS),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / max(len(train_rows), 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            with open(VOCAB_PATH, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "labels": LABELS,
                        "vocab_size": len(vocab),
                        "embed_dim": args.embed_dim,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "max_len": args.max_len,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    print(f"训练完成，最佳验证准确率：{best_acc:.4f}")
    print(f"模型文件已保存到：{MODEL_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练简易 Transformer 新闻分类模型")
    parser.add_argument("--data-path", action="append", dest="data_path", help="可重复传入多个训练数据文件路径")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
