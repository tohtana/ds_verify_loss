# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
from jinja2 import Template
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='Config generation')

    parser.add_argument('--machine_rank', type=int, help='machine_rank')
    parser.add_argument('--num_machines', type=int, help='num_machines')
    parser.add_argument('--num_processes', type=int, help='num_processes')
    parser.add_argument('--zero_stage', type=int, choices=[0, 1, 2, 3], help='ZeRO stage')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--deepcompile', action='store_true', help='Use deepcompile')
    parser.add_argument('--debug_log', action='store_true', help='Debug log')
    parser.add_argument('--sync_before_reduce', action='store_true', help='Sync before reduce')
    parser.add_argument('--sync_after_reduce', action='store_true', help='Sync after reduce')
    parser.add_argument('--sync_before_allgather', action='store_true', help='Sync before allgather')
    parser.add_argument('--sync_after_allgather', action='store_true', help='Sync after allgather')
    parser.add_argument('--zero_overlap_comm', type=str, choices=['true', 'false'], default='true')
    parser.add_argument('--zero_contiguous_gradients', type=str, choices=['true', 'false'], default='')
    parser.add_argument('--zero_reduce_scatter', type=str, choices=['true', 'false'], default='')
    parser.add_argument('--zero_allgather_partitions', type=str, choices=['true', 'false'], default='')
    parser.add_argument('--zero_reduce_bucket_size', type=int, default=0)
    parser.add_argument('--zero_allgather_bucket_size', type=int, default=0)
    parser.add_argument('--zero_sub_group_size', type=int, default=0)
    parser.add_argument('--zero_stage3_prefetch_bucket_size', type=int, default=0)
    parser.add_argument('--zero_stage3_param_persistence_threshold', type=int, default=-1)
    parser.add_argument('--zero_stage3_max_live_parameters', type=int, default=0)
    parser.add_argument('--zero_stage3_max_reuse_distance', type=int, default=0)
                        
    parser.add_argument('--template_file', type=Path, help='Template file')
    parser.add_argument('--output_file', type=Path, help='Output file')

    return parser.parse_args()


def main(args):
    with open(args.template_file, 'r') as f:
        template = Template(f.read())

    with open(args.output_file, 'w') as f:
        f.write(template.render(machine_rank=args.machine_rank,
                                num_machines=args.num_machines,
                                num_processes=args.num_processes,
                                zero_stage=args.zero_stage,
                                fp16=args.fp16,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                deepcompile=str(args.deepcompile).lower(),
                                debug_log=str(args.debug_log).lower(),
                                sync_before_reduce=str(args.sync_before_reduce).lower(),
                                sync_after_reduce=str(args.sync_after_reduce).lower(),
                                sync_before_allgather=str(args.sync_before_allgather).lower(),
                                sync_after_allgather=str(args.sync_after_allgather).lower(),
                                zero_overlap_comm=args.zero_overlap_comm,
                                zero_contiguous_gradients=args.zero_contiguous_gradients,
                                zero_reduce_scatter=args.zero_reduce_scatter,
                                zero_allgather_partitions=args.zero_allgather_partitions,
                                zero_reduce_bucket_size=args.zero_reduce_bucket_size,
                                zero_allgather_bucket_size=args.zero_allgather_bucket_size,
                                zero_sub_group_size=args.zero_sub_group_size,
                                zero_stage3_prefetch_bucket_size=args.zero_stage3_prefetch_bucket_size,
                                zero_stage3_param_persistence_threshold=args.zero_stage3_param_persistence_threshold,
                                zero_stage3_max_live_parameters=args.zero_stage3_max_live_parameters,
                                zero_stage3_max_reuse_distance=args.zero_stage3_max_reuse_distance))

if __name__ == '__main__':
    args = get_args()
    main(args)
