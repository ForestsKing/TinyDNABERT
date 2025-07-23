import argparse
import os
from itertools import chain

import swanlab
from accelerate.test_utils.testing import get_backend
from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback
from swanlab.plugin.notification import LarkCallback
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

lark_callback = LarkCallback(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/xxxx",
    secret="xxxx",
)


def group_texts(examples, block_size=512):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_file", type=str, default="./Result/TinyDNABERT-PretrainData-V1/genes.txt")
    parser.add_argument("--init_path", type=str, default="./Result/TinyDNABERT-20M-V1-Init/")
    parser.add_argument("--final_path", type=str, default="./Result/TinyDNABERT-20M-V1-Final/")

    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    parser.add_argument("--max_step", type=int, default=200_000)
    parser.add_argument("--save_step", type=int, default=1_000)
    parser.add_argument("--warmup_step", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()

    swanlab.init(
        project="TinyDNABERT-20M-V1",
        experiment_name="pre-training",
        config=args,
        callbacks=[lark_callback],
    )

    # load BPE tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.final_path)

    # init Roberta model
    device, _, _ = get_backend()
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_len + 2,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
    )
    model = RobertaForMaskedLM(config=config).to(device)
    model.save_pretrained(args.init_path)
    print(f"\nParameters: {model.num_parameters():,}")

    # load text file
    train_dataset, valid_dataset = load_dataset("text", data_files=args.text_file, split=['train[:90%]', 'train[90%:]'])
    print(f"\nTrain Dataset Size: {len(train_dataset)}, Valid Dataset Size: {len(valid_dataset)}")

    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenizer(x["text"]), batched=True, num_proc=8, remove_columns=["text"])
    tokenized_valid_dataset = valid_dataset.map(
        lambda x: tokenizer(x["text"]), batched=True, num_proc=8, remove_columns=["text"])
    grouped_train_dataset = tokenized_train_dataset.map(
        lambda x: group_texts(x, block_size=args.max_len), batched=True, num_proc=8)
    grouped_valid_dataset = tokenized_valid_dataset.map(
        lambda x: group_texts(x, block_size=args.max_len), batched=True, num_proc=8)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)
    print()

    # train Roberta model
    train_args = TrainingArguments(
        output_dir=f"./checkpoint/train/",
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
        train_dataset=grouped_train_dataset,
        eval_dataset=grouped_valid_dataset,
        data_collator=data_collator,
        callbacks=[SwanLabCallback()]
    )
    trainer.train()

    print()
    model.save_pretrained(args.final_path)
    swanlab.finish()
