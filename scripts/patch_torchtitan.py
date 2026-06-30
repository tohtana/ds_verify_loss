#!/usr/bin/env python3
"""
Idempotent compatibility patch for the pinned torchtitan submodule.

torchtitan main calls `create_block_mask(separate_full_blocks=...)` (FlexAttention),
a kwarg present only in a narrow torch nightly window — not in stable torch 2.6 nor
the latest. We make `create_attention_mask` drop kwargs the installed torch's
`create_block_mask` doesn't accept (it's a perf hint; dropping it is functionally
fine). Re-run after a fresh submodule checkout. Called by setup_torchtitan_env.sh.
"""
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "third_party/torchtitan/torchtitan/models/common/attention.py"

OLD = (
    'def create_attention_mask(*args, **kwargs):\n'
    '    """Create an attention mask using compiled create_block_mask."""\n'
    '    return _compiled_create_block_mask(*args, **kwargs)'
)
NEW = (
    'import inspect as _inspect\n'
    '_CBM_PARAMS = set(_inspect.signature(create_block_mask).parameters)\n\n\n'
    'def create_attention_mask(*args, **kwargs):\n'
    '    """Create an attention mask using compiled create_block_mask."""\n'
    '    # bench patch: drop kwargs the installed torch\'s create_block_mask does not\n'
    '    # accept (e.g. separate_full_blocks, only in a narrow torch nightly window).\n'
    '    kwargs = {k: v for k, v in kwargs.items() if k in _CBM_PARAMS}\n'
    '    return _compiled_create_block_mask(*args, **kwargs)'
)


def main() -> None:
    src = TARGET.read_text()
    if "_CBM_PARAMS" in src:
        print("patch_torchtitan: already patched")
    elif OLD in src:
        TARGET.write_text(src.replace(OLD, NEW))
        print("patch_torchtitan: patched create_attention_mask")
    else:
        print("patch_torchtitan: WARN anchor not found (torchtitan source changed) — review manually")


if __name__ == "__main__":
    main()
