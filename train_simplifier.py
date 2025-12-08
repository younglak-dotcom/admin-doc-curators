# train_simplifier.py
"""
행정문서 -> 쉬운 문장 순화 모델 파인튜닝 스크립트
"""

import pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from config import (
    PARALLEL_DATA_PATH,
    SIMPLIFIER_BASE_MODEL,
    SIMPLIFIER_SAVE_DIR,
    MAX_SOURCE_LEN,
    MAX_TARGET_LEN,
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
)

def load_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)[["source", "target"]]
    return Dataset.from_pandas(df)

def preprocess(batch, tokenizer):
    inputs = ["행정문서 순화: " + s for s in batch["source"]]
    targets = batch["target"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LEN,
        truncation=True,
    )
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    print("📘 순화 모델 파인튜닝 시작")

    raw = load_dataset(PARALLEL_DATA_PATH)
    split = raw.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    tokenizer = T5Tokenizer.from_pretrained(SIMPLIFIER_BASE_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(SIMPLIFIER_BASE_MODEL)

    train_ds = train_ds.map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = TrainingArguments(
        output_dir="runs/simplifier",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=2,
        predict_with_generate=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print("✅ 학습 완료, 모델 저장 중...")
    model.save_pretrained(SIMPLIFIER_SAVE_DIR)
    tokenizer.save_pretrained(SIMPLIFIER_SAVE_DIR)
    print(f"✅ 저장 완료: {SIMPLIFIER_SAVE_DIR}")

if __name__ == "__main__":
    main()
