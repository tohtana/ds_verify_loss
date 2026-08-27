"""
Data utilities for loading datasets, tokenizers, and data loaders.
This module handles all dataset-related functionality for the loss verification script.
"""

import os
from typing import Tuple, Any
import torch
from datasets import load_dataset, DownloadConfig
from datasets.utils.logging import disable_progress_bar
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class SyntheticTokenDataset(Dataset):
    """Deterministic fixed-token dataset for benchmark reproduction runs."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int, seed: int = 1234):
        self.num_samples = num_samples
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.input_ids = torch.randint(
            low=0,
            high=max(int(vocab_size), 8),
            size=(num_samples, seq_length),
            dtype=torch.long,
            generator=generator,
        )
        self.attention_mask = torch.ones((num_samples, seq_length), dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }


def get_tokenizer(model_name: str, trust_remote_code: bool = True) -> Any:
    """
    Load and configure the tokenizer for the given model.
    
    Args:
        model_name: Name of the model to load tokenizer for
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback for models without eos_token
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(2)
    
    return tokenizer


def load_and_prepare_dataset(
    dataset_name: str,
    dataset_percentage: float,
    tokenizer: Any,
    seq_length: int,
    accelerator: Any,
    batch_size: int,
    is_main_process: bool = True,
    dataset_samples: int = 1024,
) -> Tuple[Any, Any, str]:
    """
    Load and prepare the dataset with tokenization and data loader creation.
    
    Args:
        dataset_name: Name of the dataset to load
        dataset_percentage: Percentage of dataset to use (0.0-1.0)
        tokenizer: Tokenizer to use for text processing
        seq_length: Maximum sequence length for tokenization
        accelerator: Accelerator object for distributed training
        batch_size: Batch size for the data loader
        is_main_process: Whether this is the main process
        
    Returns:
        Tuple of (dataset, data_loader, text_column_name)
    """
    # Disable progress bar for non-main processes
    if not is_main_process:
        disable_progress_bar()
    
    if is_main_process:
        print(f"Loading dataset: {dataset_name} ({dataset_percentage*100:.1f}% of data)...")

    if dataset_name in {"synthetic", "synthetic_tokens", "synthetic-token"}:
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        sample_count = max(dataset_samples, accelerator.num_processes * batch_size)
        dataset = SyntheticTokenDataset(sample_count, seq_length, vocab_size)
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        if is_main_process:
            print(
                f"Synthetic dataset ready: {len(dataset)} examples, "
                f"seq_length={seq_length}, vocab_size={vocab_size}"
            )
        return dataset, data_loader, "synthetic_tokens"
    
    # Calculate split string based on percentage
    if dataset_percentage >= 1.0:
        split_str = "train"
    else:
        # Convert to integer percentage to avoid decimal points
        percentage_int = int(dataset_percentage * 100)
        split_str = f"train[:{percentage_int}%]"
    
    # Load the specified dataset
    if dataset_name == "wikitext":
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split_str, download_config=DownloadConfig(disable_tqdm=True))
        text_column = 'text'
    elif dataset_name == "openwebtext":
        # For openwebtext, use a smaller percentage since it's very large
        if dataset_percentage >= 1.0:
            split_str = "train[:1%]"  # Cap at 1% for full dataset requests
        dataset = load_dataset('openwebtext', split=split_str, download_config=DownloadConfig(disable_tqdm=True))
        text_column = 'text'
    elif dataset_name == "c4":
        # For C4, use even smaller percentage since it's massive
        if dataset_percentage >= 1.0:
            split_str = "train[:0.1%]"  # Cap at 0.1% for full dataset requests
        dataset = load_dataset('c4', 'en', split=split_str, download_config=DownloadConfig(disable_tqdm=True))
        text_column = 'text'
    elif dataset_name == "ag_news":
        dataset = load_dataset('ag_news', split=split_str, download_config=DownloadConfig(disable_tqdm=True))
        text_column = 'text'
    else:
        # Try to load as custom dataset
        try:
            dataset = load_dataset(dataset_name, split=split_str, download_config=DownloadConfig(disable_tqdm=True))
            # Try common text column names
            if 'text' in dataset.column_names:
                text_column = 'text'
            elif 'content' in dataset.column_names:
                text_column = 'content'
            elif 'body' in dataset.column_names:
                text_column = 'body'
            else:
                text_column = dataset.column_names[0]
                if is_main_process:
                    print(f"Warning: Using column '{text_column}' as text column. Available columns: {dataset.column_names}")
        except Exception as e:
            if is_main_process:
                print(f"Error loading dataset '{dataset_name}': {e}")
                print("Falling back to wikitext dataset...")
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split_str, download_config=DownloadConfig(disable_tqdm=True))
            text_column = 'text'
    
    if is_main_process:
        print(f"Dataset loaded: {len(dataset)} examples using column '{text_column}'")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding='max_length', max_length=seq_length, truncation=True)
    
    # Tokenize the dataset
    if is_main_process:
        print("Tokenizing dataset...")
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=1, keep_in_memory=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    if is_main_process:
        print(f"Tokenization complete. Dataset ready with {len(tokenized_dataset)} examples.")
    
    # Create data loader
    sampler = DistributedSampler(
        tokenized_dataset, 
        num_replicas=accelerator.num_processes, 
        rank=accelerator.process_index
    )
    data_loader = DataLoader(
        tokenized_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4
    )
    
    return dataset, data_loader, text_column


def get_dataset_info(dataset_name: str, dataset_percentage: float) -> dict:
    """
    Get metadata about the dataset configuration for logging purposes.
    
    Args:
        dataset_name: Name of the dataset
        dataset_percentage: Percentage of dataset being used
        
    Returns:
        Dictionary with dataset metadata
    """
    return {
        "dataset_name": dataset_name,
        "dataset_percentage": dataset_percentage,
        "effective_dataset_size": f"{dataset_percentage*100:.1f}%"
    }
