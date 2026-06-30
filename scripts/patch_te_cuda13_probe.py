#!/usr/bin/env python3
"""Make TransformerEngine usable on a box that has BOTH CUDA 12 and CUDA 13 visible.

TE's core (libtransformer_engine.so) detects the CUDA runtime version by dlopen-probing
a hard-coded SONAME list ["libcudart.so.12", "libcudart.so.13"] (and the same for nvrtc).
On this cluster /etc/ld.so.conf.d/cuda-13-0.conf puts cuda-13.0 in the ldconfig cache, so
`dlopen("libcudart.so.13")` succeeds even though we run a cu12 torch + cu12 TE. TE then sees
two libcudart majors and aborts every kernel call with:

    RuntimeError: Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13

LD_LIBRARY_PATH can't fix it (the loader always falls through to the ldconfig cache after it),
there's no NVTE env knob for the runtime version, and we don't have sudo to edit ldconfig.
So we patch the probe: rewrite the ".so.13" SONAMEs in the vendored TE core to ".so.99"
(a major that doesn't exist) -> that dlopen fails -> TE finds only the cu12 runtime. The
replacement is exactly the same length, so the ELF stays byte-for-byte valid. Idempotent;
keeps a .cuda13probe.bak the first time. Re-run after any TE reinstall/upgrade.
"""
import os
import shutil
import sys
import sysconfig

REPLS = [
    (b"libcudart.so.13\x00", b"libcudart.so.99\x00"),
    (b"libnvrtc.so.13\x00", b"libnvrtc.so.99\x00"),
]


def find_core(py_prefix=None):
    base = py_prefix or sysconfig.get_path("purelib")
    cand = os.path.join(base, "transformer_engine", "wheel_lib", "libtransformer_engine.so")
    if os.path.exists(cand):
        return cand
    # fall back: search under the TE package for the core .so
    te = os.path.join(base, "transformer_engine")
    for root, _dirs, files in os.walk(te):
        if "libtransformer_engine.so" in files:
            return os.path.join(root, "libtransformer_engine.so")
    return None


def main():
    so = sys.argv[1] if len(sys.argv) > 1 else find_core()
    if not so or not os.path.exists(so):
        print("ERROR: libtransformer_engine.so not found "
              "(pass its path, or run with the megatron venv's python)", file=sys.stderr)
        return 2

    data = bytearray(open(so, "rb").read())
    if data.count(b"libcudart.so.13\x00") == 0 and data.count(b"libcudart.so.99\x00") >= 1:
        print(f"already patched: {so}")
        return 0

    bak = so + ".cuda13probe.bak"
    if not os.path.exists(bak):
        shutil.copy2(so, bak)
        print(f"backup -> {bak}")

    total = 0
    for old, new in REPLS:
        assert len(old) == len(new)
        n = data.count(old)
        if n:
            data = bytearray(bytes(data).replace(old, new))
            total += n
        print(f"  {old.rstrip(chr(0).encode()).decode():18s} -> "
              f"{new.rstrip(chr(0).encode()).decode():18s}  ({n})")
    open(so, "wb").write(data)
    assert os.path.getsize(so) == os.path.getsize(bak), "size changed!"
    print(f"patched {total} SONAME(s) in {so}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
