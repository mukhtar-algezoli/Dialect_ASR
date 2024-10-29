from __future__ import print_function
import argparse
import torch
from src.models.model import load_whisper
from src.data.make_dataset import get_dataset, DataCollatorSpeechSeq2SeqWithPadding
from src.train.train import get_trainer, get_train_args, compute_metrics
from dotenv import load_dotenv
import wandb
import huggingface_hub
import os 
import logging


load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HG_API_KEY = os.getenv("HG_API_KEY")  # for huggingface







def main():
    # Training settings
    parser = argparse.ArgumentParser(description='ASR 4M training')

    parser.add_argument('--wandb_token', type=str ,help='Weights and Biases Token')
    parser.add_argument('--hg_token', type=str ,help='Huggingface Token')

    parser.add_argument('--per_device_train_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='lwarmup steps')

    parser.add_argument('--model_path', type=str, default="openai/whisper-large-v3", 
                        help='Whisper model')
    
    parser.add_argument('--data_path', type=str, default="UBC-NLP/masc_cv15_asc_fleurs_mgb5_mgb2_qasr_100K" ,help='path to the dataset in Huggingface')
    
    parser.add_argument('--output_path', type=str, default="./models/whisper-V3-finetuned" ,help='path to the output dir')

    parser.add_argument("--push_to_hub", type=bool, default=True, help="Whether or not to push the model to the Hub.")

    parser.add_argument('--wandb_project_name', type=str, default="Dysarthria Classification", help='wandb project name')

    parser.add_argument('--wandb_run_name', type=str, default="Test Run",  help='current wandb run name')

    parser.add_argument('--EXP_name', type=str , default="Test Run", help='the name of the experiment examples: EXP1, EXP2, ...')

    parser.add_argument('--seed', type=int, default=101, metavar='S',
                        help='random seed (default: 101)')

    parser.add_argument('--checkpoint_path', type=str, default="./models" ,help='path to the checkpoint')

    args = parser.parse_args()


    torch.manual_seed(args.seed)

    logging.info("logging to HG and wandb...")
    huggingface_hub.login(token=HG_API_KEY)
    wandb.login(key=WANDB_API_KEY)
    os.environ["WANDB_PROJECT"]="dialects_ASR"

    logging.info("checking available device...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"device: {device}\n device count: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        logging.info(f"device: cpu")


    logging.info("load the model...")
    feature_extractor, tokenizer, processor, model = load_whisper(path=args.model_path)

    logging.info("prepare dataset...")
    data = get_dataset(path=args.data_path)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )


    logging.info("define training...")    
    training_args = get_train_args(args)
    trainer = get_trainer(training_args, 
                          model, 
                          data["train"], 
                          data["validation"], 
                          data_collator, 
                          compute_metrics, 
                          processor)
    
    logging.info("start training...")
    trainer.train()

    logging.info("save model...")
    if args.push_to_hub:
        kwargs = {
            "dataset_tags": args.data_path,
            "dataset_args": "config: ar, split: test",
            "language": "ar",
            "model_name": "Whisper V3 Finetuned",  # a 'pretty' name for our model
            "finetuned_from": args.data_path,
            "tasks": "automatic-speech-recognition",
            }
        trainer.push_to_hub()
    else:
        trainer.save_model(args.output_path)


    




if __name__ == '__main__':
    main()