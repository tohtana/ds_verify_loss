import os
import argparse
import json
import platform
import socket
import time
import traceback
from datetime import datetime
from contextlib import nullcontext, contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, enable_full_determinism, set_seed
from accelerate import Accelerator
import wandb

from data_utils import get_tokenizer, load_and_prepare_dataset

@contextmanager
def use_default_device(device):
    prev_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(prev_device)


def chunked_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_tokens: int,
    empty_cache_before_chunks: bool = False,
    loss_device: str = "cuda",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute the same shifted causal-LM CE loss while upcasting logits by chunks."""
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if logits.ndim != 3:
        raise ValueError(f"expected logits with shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"expected labels with shape [batch, seq], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits/labels shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
    if logits.shape[1] < 2:
        raise ValueError("causal LM loss requires sequence length >= 2")
    if loss_device not in {"cuda", "cpu"}:
        raise ValueError(f"loss_device must be 'cuda' or 'cpu', got {loss_device!r}")

    vocab_size = logits.shape[-1]
    reduction_device = torch.device("cpu") if loss_device == "cpu" else logits.device
    total_loss = torch.zeros((), dtype=torch.float32, device=reduction_device)
    total_items = torch.zeros((), dtype=torch.float32, device=reduction_device)

    # Match Transformers' shifted ForCausalLMLoss while avoiding one full
    # [batch, seq, vocab] fp32 allocation.
    if empty_cache_before_chunks and logits.is_cuda:
        torch.cuda.empty_cache()

    for start in range(0, logits.shape[1] - 1, chunk_tokens):
        end = min(start + chunk_tokens, logits.shape[1] - 1)
        chunk_logits = logits[:, start:end, :]
        if loss_device == "cpu":
            chunk_logits = chunk_logits.to(device="cpu", dtype=torch.float32)
            chunk_labels = labels[:, start + 1 : end + 1].to("cpu")
        else:
            chunk_logits = chunk_logits.float()
            chunk_labels = labels[:, start + 1 : end + 1].to(chunk_logits.device)
        flat_labels = chunk_labels.reshape(-1)
        total_loss = total_loss + F.cross_entropy(
            chunk_logits.reshape(-1, vocab_size),
            flat_labels,
            ignore_index=ignore_index,
            reduction="sum",
        )
        total_items = total_items + (flat_labels != ignore_index).sum().to(total_items.dtype)

    return total_loss / total_items.clamp_min(1.0)


def chunked_lm_head_loss_backward(
    model: torch.nn.Module,
    accelerator: Accelerator,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_tokens: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Backprop causal-LM CE in sequence chunks without full logits gradients."""
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")

    module = getattr(model, "module", model)
    transformer = getattr(module, "model", None)
    lm_head = getattr(module, "lm_head", None)
    if transformer is None or lm_head is None:
        raise ValueError(
            "chunked LM-head loss requires a Hugging Face causal LM with "
            "`.model` and `.lm_head` attributes"
        )

    outputs = transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    if hidden_states.shape[:2] != input_ids.shape:
        raise ValueError(
            "hidden/input shape mismatch: "
            f"{tuple(hidden_states.shape[:2])} vs {tuple(input_ids.shape)}"
        )
    if hidden_states.shape[1] < 2:
        raise ValueError("causal LM loss requires sequence length >= 2")

    shift_labels = input_ids[:, 1:]
    total_items = (shift_labels != ignore_index).sum().clamp_min(1).to(
        device=hidden_states.device,
        dtype=torch.float32,
    )
    total_loss_value = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    last_start = hidden_states.shape[1] - 1

    for start in range(0, last_start, chunk_tokens):
        end = min(start + chunk_tokens, last_start)
        chunk_hidden = hidden_states[:, start:end, :]
        chunk_logits = lm_head(chunk_hidden).float()
        chunk_labels = input_ids[:, start + 1 : end + 1].to(chunk_logits.device)
        loss_sum = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.shape[-1]),
            chunk_labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
        chunk_loss = loss_sum / total_items
        total_loss_value = total_loss_value + chunk_loss.detach()
        accelerator.backward(chunk_loss, retain_graph=end < last_start)
        del chunk_hidden, chunk_logits, chunk_labels, loss_sum, chunk_loss

    return total_loss_value

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name for pretraining evaluation")
    parser.add_argument("--dataset_percentage", type=float, default=10.0, help="Percentage of dataset to use (e.g., 10.0 for 10 percent)")
    parser.add_argument("--dataset_samples", type=int, default=1024, help="Synthetic dataset sample count for dataset_name=synthetic")
    parser.add_argument("--num_layers", type=int, default=0)
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--passes", type=str, default=None)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--offload_opt_states", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--profile_dir", type=str, default=None)
    parser.add_argument("--profile_summary_output", type=str, default=None)
    parser.add_argument(
        "--chunked_causal_lm_loss_tokens",
        type=int,
        default=0,
        help="If >0, compute shifted causal-LM cross entropy in token chunks of this size.",
    )
    parser.add_argument(
        "--chunked_causal_lm_loss_empty_cache",
        action="store_true",
        help="Call torch.cuda.empty_cache() before chunked causal-LM loss upcasts.",
    )
    parser.add_argument(
        "--chunked_causal_lm_loss_device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device used for chunked causal-LM cross entropy.",
    )
    parser.add_argument(
        "--chunked_lm_head_loss_tokens",
        type=int,
        default=0,
        help=(
            "If >0, run the transformer once, then apply lm_head and backward "
            "causal-LM CE in token chunks to avoid full logits/grad tensors."
        ),
    )
    parser.add_argument("--profile_wait_steps", type=int, default=0)
    parser.add_argument("--profile_warmup_steps", type=int, default=10)
    parser.add_argument("--profile_active_steps", type=int, default=3)
    parser.add_argument("--metrics_output", type=str, default=None)
    parser.add_argument("--bench_step", type=int, default=100)
    parser.add_argument("--warmup_step", type=int, default=15)
    parser.add_argument("--zero_stage", type=int, default=3)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument("--load_weights", action="store_true")
    
    # WandB logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="ds-verify-loss", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="WandB tags for the run")

    return parser.parse_args()


def write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(output)


def rank_from_env() -> int:
    for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


def exception_summary(exc: BaseException) -> Dict[str, Any]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    lines = [line.rstrip() for line in tb.splitlines() if line.strip()]
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": "\n".join(lines[-24:]),
    }


def classify_fraction(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value >= 0.25:
        return "high"
    if value >= 0.10:
        return "medium"
    return "low"


def classify_imbalance(rank_step_times: List[Dict[str, float]]) -> str:
    values = [
        item["avg_step_time_sec"]
        for item in rank_step_times
        if item.get("measured_steps", 0) > 0 and item.get("avg_step_time_sec", 0) > 0
    ]
    if len(values) < 2:
        return "unknown"
    ratio = max(values) / max(min(values), 1e-9)
    if ratio >= 1.30:
        return "high"
    if ratio >= 1.10:
        return "medium"
    return "low"


def summarize_profile(
    profiler: Any,
    active_step_times: List[float],
    rank_step_times: List[Dict[str, float]],
    peak_allocated: int,
    peak_reserved: int,
    compile_time_sum: float,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "gpu_idle": "unknown",
        "rank_imbalance": classify_imbalance(rank_step_times),
        "visible_nccl_collective_cost": "unknown",
        "input_h2d_wait": "unknown",
        "optimizer_tail": "unknown",
        "allocator_or_memory_pressure": "unknown",
        "compile_graph_break_signal": "not-measured",
        "notes": [],
    }

    if peak_reserved > 0 and peak_allocated > 0:
        reserved_gap = max(peak_reserved - peak_allocated, 0) / peak_reserved
        summary["allocator_or_memory_pressure"] = classify_fraction(reserved_gap)
        summary["allocator_reserved_gap_fraction"] = reserved_gap

    if compile_time_sum:
        summary["compile_graph_break_signal"] = "unknown"

    if profiler is None:
        summary["notes"].append("PyTorch profiler was not enabled for this run.")
        return summary

    try:
        events = profiler.key_averages()
    except Exception as exc:  # profiler summarization must not fail training
        summary["notes"].append(f"Profiler key_averages failed: {type(exc).__name__}: {exc}")
        return summary

    total_cuda_us = 0.0
    nccl_us = 0.0
    copy_us = 0.0
    optimizer_us = 0.0
    for event in events:
        key = event.key.lower()
        cuda_us = float(getattr(event, "self_cuda_time_total", 0.0) or 0.0)
        cpu_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
        total_cuda_us += cuda_us
        if "nccl" in key or "all_reduce" in key or "all_gather" in key or "reduce_scatter" in key:
            nccl_us += cuda_us
        if "memcpy" in key or "copy_" in key or "aten::to" in key or "_to_copy" in key:
            copy_us += max(cuda_us, cpu_us)
        if "adam" in key or "optimizer" in key or "optim" in key:
            optimizer_us += max(cuda_us, cpu_us)

    wall_us = sum(active_step_times) * 1_000_000.0
    if wall_us > 0 and total_cuda_us > 0:
        cuda_wall_fraction = min(total_cuda_us / wall_us, 1.0)
        summary["cuda_wall_fraction_rank0"] = cuda_wall_fraction
        if cuda_wall_fraction < 0.35:
            summary["gpu_idle"] = "high"
        elif cuda_wall_fraction < 0.65:
            summary["gpu_idle"] = "medium"
        else:
            summary["gpu_idle"] = "low"

    if total_cuda_us > 0:
        summary["visible_nccl_collective_cost"] = classify_fraction(nccl_us / total_cuda_us)
        summary["input_h2d_wait"] = classify_fraction(copy_us / total_cuda_us)
        summary["optimizer_tail"] = classify_fraction(optimizer_us / total_cuda_us)
        summary["profile_cuda_time_us"] = total_cuda_us
        summary["profile_nccl_time_us"] = nccl_us
        summary["profile_copy_time_us"] = copy_us
        summary["profile_optimizer_time_us"] = optimizer_us
    else:
        summary["notes"].append("Profiler collected no CUDA self time on rank 0.")

    return summary


def make_schedule(passes: List[str], warmup):
    from deepspeed.compile.passes import zero3_compile, prefetch, selective_gather, offload_adam_states

    schedule = []

    if "offload_adam_states" in passes:
        assert len(passes) == 1, "offload_adam_states should be the only pass"
        schedule.append((0, [offload_adam_states.offload_adam_states_for_init, zero3_compile.add_z3_gather_release, offload_adam_states.move_opt_states_sync]))
        schedule.append((5, [offload_adam_states.offload_adam_states_for_init, zero3_compile.add_z3_gather_release, offload_adam_states.move_opt_states]))
    elif "offload_adam_states_sync" in passes:
        assert len(passes) == 1, "offload_adam_states_sync should be the only pass"
        schedule.append((0, [zero3_compile.add_z3_gather_release, offload_adam_states.move_opt_states_sync]))
    else:
        schedule.append((0, [zero3_compile.add_z3_gather_release]))
        second_opt = [zero3_compile.add_z3_gather_release]
        if "prefetch" in passes:
            second_opt.append(prefetch.schedule_prefetch)
        if "selective_gather" in passes:
            second_opt.append(selective_gather.selective_gather)
        schedule.append((warmup, second_opt))
    return schedule


def run_training(args):
    print(args)
    if args.chunked_causal_lm_loss_tokens > 0 and args.chunked_lm_head_loss_tokens > 0:
        raise ValueError(
            "--chunked_causal_lm_loss_tokens and --chunked_lm_head_loss_tokens "
            "are mutually exclusive"
        )

    if args.passes is not None and "offload_adam_states" in args.passes:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    if args.deterministic:
        enable_full_determinism(args.seed)
        from torch._inductor import config
        config.fallback_random = True
    else:
        set_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    is_deepspeed = accelerator.state.deepspeed_plugin is not None
    print(f"Running on device: {device} is_deepspeed: {is_deepspeed}")

    # Load model and tokenizer
    if accelerator.is_main_process:
        print("Loading model and tokenizer...")

    model_name = args.model_name

    if args.load_weights:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        model_config = AutoConfig.from_pretrained(model_name, attn_implementation=args.attn_impl, trust_remote_code=True)
        with use_default_device(device):
            if args.num_layers > 0:
                print(f"num_hidden_layers: {model_config.num_hidden_layers} -> {args.num_layers}")
                model_config.num_hidden_layers = args.num_layers
            model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)

    # Load tokenizer
    tokenizer = get_tokenizer(model_name, trust_remote_code=True)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    # Load and prepare dataset
    _, data_loader, _ = load_and_prepare_dataset(
        dataset_name=args.dataset_name,
        dataset_percentage=args.dataset_percentage / 100.0,  # Convert percentage to fraction
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        accelerator=accelerator,
        batch_size=args.batch_size,
        is_main_process=accelerator.is_main_process,
        dataset_samples=args.dataset_samples,
    )

    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Prepare everything with accelerator
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    print(f"Model prepared: {model.__class__} optimizer: {optimizer.__class__}")

    # Determine experimental settings for logging
    is_deepcompile = is_deepspeed and hasattr(model, '_config') and hasattr(model._config, 'compile_config') and model._config.compile_config.deepcompile

    # Create experiment name (used for wandb and file logging)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name_short = args.model_name.split("/")[-1]
    exp_name = f"{model_name_short}_np{accelerator.num_processes}ds{1 if is_deepspeed else 0}" \
               f"B{args.backend}z{args.zero_stage}" \
               f"L{0 if args.num_layers is None else args.num_layers}" \
               f"bs{args.batch_size}seq{args.seq_length}acc{args.gradient_accumulation_steps}ac{1 if args.activation_checkpointing else 0}" \
               f"c{1 if args.compile else 0}" \
               f"dc{1 if is_deepcompile else 0}" \
               f"pass_{'none' if args.passes is None else args.passes.replace(',', '_')}_" \
               f"os{1 if args.offload_opt_states else 0}" \
               f"T{timestamp}"

    # Initialize wandb logging
    if args.use_wandb and accelerator.is_main_process:
        wandb_run_name = args.wandb_run_name if args.wandb_run_name else exp_name
        
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            tags=args.wandb_tags,
            config={
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "seq_length": args.seq_length,
                "learning_rate": args.learning_rate,
                "max_grad_norm": args.max_grad_norm,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "activation_checkpointing": args.activation_checkpointing,
                "dataset_name": args.dataset_name,
                "dataset_percentage": args.dataset_percentage,
                "dataset_samples": args.dataset_samples,
                "num_layers": args.num_layers,
                "attn_impl": args.attn_impl,
                "compile": args.compile,
                "passes": args.passes,
                "backend": args.backend,
                "offload_opt_states": args.offload_opt_states,
                "chunked_causal_lm_loss_tokens": args.chunked_causal_lm_loss_tokens,
                "chunked_causal_lm_loss_empty_cache": args.chunked_causal_lm_loss_empty_cache,
                "chunked_causal_lm_loss_device": args.chunked_causal_lm_loss_device,
                "chunked_lm_head_loss_tokens": args.chunked_lm_head_loss_tokens,
                "zero_stage": args.zero_stage,
                "is_deepspeed": is_deepspeed,
                "is_deepcompile": is_deepcompile,  # Experimental setting
                "num_processes": accelerator.num_processes,
                "deterministic": args.deterministic,
            }
        )

    if "Mixtral" in model_name:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True

    if is_deepspeed:
        if args.compile:
            schedule = make_schedule(args.passes.split(","), warmup=5) if args.passes else None
            model.compile(backend=args.backend, schedule=schedule)
    else:
        if args.compile:
            model = torch.compile(model, backend=args.backend)

    prof_dir = None
    if args.profile:
        base_profile_dir = Path(args.profile_dir or "profiles")
        if accelerator.is_main_process:
            base_profile_dir.mkdir(parents=True, exist_ok=True)
            prof_dir = str(base_profile_dir / exp_name)
            os.makedirs(prof_dir, exist_ok=True)
        accelerator.wait_for_everyone()

    do_profile = args.profile and accelerator.is_main_process
    prof_context = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=args.profile_wait_steps * args.gradient_accumulation_steps,
            warmup=args.profile_warmup_steps * args.gradient_accumulation_steps,
            active=args.profile_active_steps * args.gradient_accumulation_steps,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
    ) if do_profile else nullcontext()

    # Training 
    if args.eval:
        model.eval()
    else:
        model.train()

    global_step = 0
    iter_times = []

    if args.chunked_causal_lm_loss_tokens > 0 and accelerator.is_main_process:
        print(
            "Using chunked causal LM loss with "
            f"chunk_tokens={args.chunked_causal_lm_loss_tokens} "
            f"empty_cache={args.chunked_causal_lm_loss_empty_cache} "
            f"loss_device={args.chunked_causal_lm_loss_device}"
        )
    if args.chunked_lm_head_loss_tokens > 0 and accelerator.is_main_process:
        print(
            "Using chunked LM-head loss/backward with "
            f"chunk_tokens={args.chunked_lm_head_loss_tokens}"
        )
    
    # Loss averaging for logging
    losses = []

    # See https://github.com/microsoft/DeepSpeed/issues/6793
    acc_context = nullcontext if is_deepspeed else accelerator.accumulate

    stop = False
    with prof_context as prof:
        for epoch in range(args.num_epochs):
            start_iter = time.time()

            for step, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                with acc_context(model):
                    update_step = (is_deepspeed and model.is_gradient_accumulation_boundary()) \
                        or (not is_deepspeed and accelerator.sync_gradients)
                    if args.chunked_lm_head_loss_tokens > 0:
                        loss = chunked_lm_head_loss_backward(
                            model,
                            accelerator,
                            input_ids,
                            attention_mask,
                            args.chunked_lm_head_loss_tokens,
                        )
                    elif args.chunked_causal_lm_loss_tokens > 0:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                        loss = chunked_causal_lm_loss(
                            outputs.logits,
                            input_ids,
                            args.chunked_causal_lm_loss_tokens,
                            empty_cache_before_chunks=args.chunked_causal_lm_loss_empty_cache,
                            loss_device=args.chunked_causal_lm_loss_device,
                        )
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
                        loss = outputs.loss

                    if args.chunked_lm_head_loss_tokens <= 0:
                        accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Accumulate loss for averaging
                    if update_step:
                        losses.append(loss.item())

                    if update_step:
                        # Calculate average loss for logging
                        avg_loss = sum(losses) / len(losses) if losses else loss.item()
                        
                        if accelerator.is_main_process and global_step % (args.log_interval * args.gradient_accumulation_steps) == 0:
                            print(f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.6f} sync: {accelerator.sync_gradients} time: {time.time() - start_iter} alloc_mem: {torch.cuda.memory_allocated()} peak_mem: {torch.cuda.max_memory_allocated()}")

                        iter_times.append(time.time() - start_iter)
                        
                        # Log timing information to wandb at specified interval
                        if args.use_wandb and accelerator.is_main_process and global_step % (args.log_interval * args.gradient_accumulation_steps) == 0:
                            wandb.log({
                                "timing/iteration_time": time.time() - start_iter,
                                "train/loss": avg_loss,
                                "train/epoch": epoch + 1,
                                "train/global_step": global_step,
                                "train/learning_rate": args.learning_rate,
                                "system/cuda_memory_allocated": torch.cuda.memory_allocated(),
                                "system/cuda_memory_peak": torch.cuda.max_memory_allocated(),
                            }, step=global_step)
                            
                            # Reset loss list after logging
                            losses = []
                        
                        start_iter = time.time()

                if do_profile:
                    prof.step()

                stop = global_step >= args.bench_step * args.gradient_accumulation_steps
                if stop:
                    break
            if stop:
                break

    iter_times = iter_times[args.warmup_step:]

    local_avg_iter_time = sum(iter_times) / len(iter_times) if iter_times else 0.0
    local_step_stats = torch.tensor(
        [float(local_avg_iter_time), float(len(iter_times))],
        dtype=torch.float64,
        device=device,
    )
    try:
        gathered_step_stats = accelerator.gather(local_step_stats).detach().cpu().reshape(-1, 2).tolist()
        rank_step_times = [
            {
                "rank": index,
                "avg_step_time_sec": float(values[0]),
                "measured_steps": int(values[1]),
            }
            for index, values in enumerate(gathered_step_stats)
        ]
    except Exception as exc:
        rank_step_times = []
        if accelerator.is_main_process:
            print(f"Warning: failed to gather rank timing stats: {type(exc).__name__}: {exc}")

    if accelerator.is_main_process:
        compile_time_sum = 0
        compile_time = 0
        if args.compile and hasattr(model, "get_compile_time"):
            compile_time = model.get_compile_time()
            compile_time_sum = sum(t for _, _, _, t in compile_time)

        avg_iter_time = local_avg_iter_time
        samples_per_step = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        tokens_per_step = samples_per_step * args.seq_length
        samples_per_second = samples_per_step / avg_iter_time if avg_iter_time > 0 else None
        tokens_per_second = tokens_per_step / avg_iter_time if avg_iter_time > 0 else None
        cuda_max_allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        cuda_allocated = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        cuda_reserved = int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
        cuda_max_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        
        msg = f"{args.model_name} ds={is_deepspeed} np={accelerator.num_processes} batch_size={args.batch_size} seq={args.seq_length} zero_stage={args.zero_stage} acc={args.gradient_accumulation_steps} ac={args.activation_checkpointing} compile={args.compile} backend={args.backend} deepcompile={is_deepcompile} passes={args.passes} compile_time={compile_time_sum} iteration time: {avg_iter_time:.4f} samples/s: {samples_per_second} tokens/s: {tokens_per_second} alloc_mem: {cuda_allocated} peak_mem: {cuda_max_allocated} reserved_mem: {cuda_reserved} peak_reserved_mem: {cuda_max_reserved}"
        print(msg)

        profile_summary = summarize_profile(
            prof if do_profile else None,
            iter_times[: args.profile_active_steps],
            rank_step_times,
            cuda_max_allocated,
            cuda_max_reserved,
            compile_time_sum,
        )

        metrics_payload = {
            "status": "success",
            "success": True,
            "error_summary": None,
            "generated_at": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "model_name": args.model_name,
            "backend": "deepspeed" if is_deepspeed else "accelerate",
            "zero_stage": args.zero_stage,
            "num_processes": accelerator.num_processes,
            "batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seq_length": args.seq_length,
            "chunked_causal_lm_loss_tokens": args.chunked_causal_lm_loss_tokens,
            "chunked_causal_lm_loss_empty_cache": args.chunked_causal_lm_loss_empty_cache,
            "chunked_causal_lm_loss_device": args.chunked_causal_lm_loss_device,
            "dataset_name": args.dataset_name,
            "dataset_percentage": args.dataset_percentage,
            "dataset_samples": args.dataset_samples,
            "global_samples_per_step": samples_per_step,
            "global_tokens_per_step": tokens_per_step,
            "bench_step": args.bench_step,
            "warmup_step": args.warmup_step,
            "measured_steps": len(iter_times),
            "avg_step_time_sec": avg_iter_time,
            "samples_per_second": samples_per_second,
            "tokens_per_second": tokens_per_second,
            "cuda_memory_allocated_bytes": cuda_allocated,
            "cuda_max_memory_allocated_bytes": cuda_max_allocated,
            "cuda_memory_reserved_bytes": cuda_reserved,
            "cuda_max_memory_reserved_bytes": cuda_max_reserved,
            "compile_time_sum_sec": compile_time_sum,
            "rank_step_times": rank_step_times,
            "profile_summary": profile_summary,
        }
        write_json(args.metrics_output, metrics_payload)
        write_json(args.profile_summary_output, profile_summary)

        # Log final summary metrics to wandb
        if args.use_wandb:
            wandb.log({
                "summary/average_iteration_time": avg_iter_time,
                "summary/compile_time_sum": compile_time_sum,
                "summary/total_steps": global_step,
                "summary/final_cuda_memory_allocated": torch.cuda.memory_allocated(),
                "summary/final_cuda_memory_peak": torch.cuda.max_memory_allocated(),
            })

        if args.profile_dir:
            # Create timestamp if it wasn't created earlier
            if 'timestamp' not in locals():
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            filepath = Path(args.profile_dir) / f"result.txt"
            with open(filepath, "a") as f:
                f.write(f"{timestamp} {msg}" + "\n")

            if args.compile:
                filepath = Path(args.profile_dir) / f"compile_time.txt"
                with open(filepath, "a") as f:
                    msg_compile =  f"{msg} compile_time={compile_time_sum} {compile_time}"
                    f.write(f"{timestamp} {msg_compile}" + "\n")

    # Close wandb run
    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()

    # # Save the model
    # if accelerator.is_main_process:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained("fine_tuned_model", save_function=accelerator.save)
    #     tokenizer.save_pretrained("fine_tuned_model")


def main():
    args = get_args()
    try:
        run_training(args)
    except Exception as exc:
        if rank_from_env() == 0:
            write_json(args.metrics_output, {
                "status": "failure",
                "success": False,
                "generated_at": datetime.now().isoformat(),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "model_name": args.model_name,
                "zero_stage": args.zero_stage,
                "bench_step": args.bench_step,
                "warmup_step": args.warmup_step,
                "chunked_causal_lm_loss_tokens": args.chunked_causal_lm_loss_tokens,
                "chunked_causal_lm_loss_empty_cache": args.chunked_causal_lm_loss_empty_cache,
                "chunked_causal_lm_loss_device": args.chunked_causal_lm_loss_device,
                "error_summary": exception_summary(exc),
            })
        raise

if __name__ == "__main__":
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    main()
