import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration


def load_whisper(path: str, language="Arabic", task="transcribe") -> tuple:
    feature_extractor = WhisperFeatureExtractor.from_pretrained(path)
    tokenizer = WhisperTokenizer.from_pretrained(path, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(path, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

    return feature_extractor, tokenizer, processor, model



