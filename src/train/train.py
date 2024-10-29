import torch
import os
import wandb
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


def get_trainer(training_args, model, train_set, eval_set, data_collator, compute_metrics, processor):
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    return trainer


def get_train_args(args):

    training_args = Seq2SeqTrainingArguments(
        output_dir= args.output_path,  # change to a repo name of your choice
        per_device_train_batch_size= args.per_device_train_batch_size,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate= args.lr,
        warmup_steps=500,
        do_eval=True,
        # max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        num_train_epochs=args.epochs,
    )

    return training_args

def compute_metrics(pred, tokenizer):
    metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}