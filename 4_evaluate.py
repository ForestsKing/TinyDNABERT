import argparse
import os

import numpy as np
import pandas as pd
import swanlab
from datasets import load_dataset, Dataset
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

swanlab.init(mode="disabled")


def process_data(train_dataset, test_dataset, tokenizer):
    train_data, train_label = train_dataset["sequence"], train_dataset["label"]
    test_data, test_label = test_dataset["sequence"], test_dataset["label"]

    train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.1)

    train_dataset = Dataset.from_dict({"data": train_data, "label": train_label})
    valid_dataset = Dataset.from_dict({"data": valid_data, "label": valid_label})
    test_dataset = Dataset.from_dict({"data": test_data, "label": test_label})

    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenizer(x["data"]), batched=True, num_proc=8, remove_columns=["data"])
    tokenized_valid_dataset = valid_dataset.map(
        lambda x: tokenizer(x["data"]), batched=True, num_proc=8, remove_columns=["data"])
    tokenized_test_dataset = test_dataset.map(
        lambda x: tokenizer(x["data"]), batched=True, num_proc=8, remove_columns=["data"])

    return tokenized_train_dataset, tokenized_valid_dataset, tokenized_test_dataset


def compute_mcc_score(output):
    predictions = np.argmax(output.predictions, axis=-1)
    references = output.label_ids
    result = {"mcc_score": matthews_corrcoef(references, predictions)}

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--flag", type=str, default="final", choices=["init", "final"])

    parser.add_argument("--init_path", type=str, default="./Result/TinyDNABERT-20M-V1-Init/")
    parser.add_argument("--final_path", type=str, default="./Result/TinyDNABERT-20M-V1-Final/")
    parser.add_argument("--data_path", type=str, default="InstaDeepAl/nucleotide_transformer_downstream_tasks_revised/")

    parser.add_argument("--max_step", type=int, default=5_000)
    parser.add_argument("--save_step", type=int, default=200)
    parser.add_argument("--warmup_step", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()

    score_list = []
    df = pd.read_csv(f"./NT_Benchmark.csv")

    for i in range(18):
        task_name = df["Task"].values[i]
        data_name = df["DataName"].values[i]
        label_num = df["LabelNum"].values[i]

        # load model
        if args.flag == "init":
            tokenizer = AutoTokenizer.from_pretrained(args.init_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                args.init_path, num_labels=label_num, device_map="auto")
        elif args.flag == "final":
            tokenizer = AutoTokenizer.from_pretrained(args.final_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                args.final_path, num_labels=label_num, device_map="auto")
        else:
            raise ValueError("Invalid flag. Choose either 'init' or 'final'.")

        # load data
        train_dataset = load_dataset(args.data_path, split="train", streaming=False).filter(
            lambda example: example["task"] == data_name)
        test_dataset = load_dataset(args.data_path, split="test", streaming=False).filter(
            lambda example: example["task"] == data_name)
        tokenized_train_dataset, tokenized_valid_dataset, tokenized_test_dataset = process_data(
            train_dataset, test_dataset, tokenizer)

        # train
        train_args = TrainingArguments(
            output_dir=f"./checkpoint/eval/{i + 1:02d}_{task_name}_{args.flag}",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            max_steps=args.max_step,

            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            eval_steps=args.save_step,
            save_steps=args.save_step,
            logging_steps=args.save_step,

            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_step,
            lr_scheduler_type="linear",

            metric_for_best_model="mcc_score",
            save_total_limit=2,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            dataloader_drop_last=True,
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_valid_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_mcc_score,
        )
        trainer.train()

        # evaluate
        score = trainer.predict(tokenized_test_dataset).metrics["test_mcc_score"]
        score_list.append(score)
        print(f"Task {i + 1:02d} {task_name} || MCC Score: {score:.4f}")

    res = df[["Task"]].copy()
    res["MCC"] = score_list
    res.to_csv(f"./result_{args.flag}.csv", index=False)
