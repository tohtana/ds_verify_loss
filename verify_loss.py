import os
import argparse
import time
from datetime import datetime
from contextlib import nullcontext
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, enable_full_determinism, set_seed
from accelerate import Accelerator
import wandb

from data_utils import get_tokenizer, load_and_prepare_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name for pretraining evaluation")
    parser.add_argument("--dataset_percentage", type=float, default=10.0, help="Percentage of dataset to use (e.g., 10.0 for 10 percent)")
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


def main():
    args = get_args()
    print(args)

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
        is_main_process=accelerator.is_main_process
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
                "num_layers": args.num_layers,
                "attn_impl": args.attn_impl,
                "compile": args.compile,
                "passes": args.passes,
                "backend": args.backend,
                "offload_opt_states": args.offload_opt_states,
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

    if args.profile_dir:
        if accelerator.is_main_process and args.profile_dir:
            os.makedirs(args.profile_dir, exist_ok=True)
            if args.profile:
                prof_dir = f"{args.profile_dir}/{exp_name}"
                os.makedirs(prof_dir, exist_ok=True)
        accelerator.wait_for_everyone()        
        
    do_profile = args.profile and accelerator.is_main_process
    prof_context = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=10*args.gradient_accumulation_steps, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
    ) if do_profile else nullcontext()

    # Training 
    if args.eval:
        model.eval()
    else:
        model.train()

    global_step = 0
    iter_times = []
    
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
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
                    loss = outputs.loss

                    update_step = (is_deepspeed and model.is_gradient_accumulation_boundary()) \
                        or (not is_deepspeed and accelerator.sync_gradients)
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

    if accelerator.is_main_process:
        compile_time_sum = 0
        compile_time = 0
        if args.compile and hasattr(model, "get_compile_time"):
            compile_time = model.get_compile_time()
            compile_time_sum = sum(t for _, _, _, t in compile_time)

        avg_iter_time = sum(iter_times) / len(iter_times) if iter_times else 0
        
        msg = f"{args.model_name} ds={is_deepspeed} np={accelerator.num_processes} batch_size={args.batch_size} seq={args.seq_length} zero_stage={args.zero_stage} acc={args.gradient_accumulation_steps} ac={args.activation_checkpointing} compile={args.compile} backend={args.backend} deepcompile={is_deepcompile} passes={args.passes} compile_time={compile_time_sum} iteration time: {avg_iter_time:.4f} alloc_mem: {torch.cuda.memory_allocated()} peak_mem: {torch.cuda.max_memory_allocated()}"
        print(msg)

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
            from pathlib import Path
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

if __name__ == "__main__":
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    main()
