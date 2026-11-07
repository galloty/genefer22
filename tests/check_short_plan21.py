"""Compare the two GFN-21 dispatch sequences on synthetic data using native Clang kernels."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
from native_opencl.kernel import compile_kernels, parameters, twiddles
from native_opencl.polynomial import expected_product, sparse


N = 1 << 21
V = 4
GROUP = {"forward64_0": 64, "forward64_9": 64, "backward64_0": 64, "backward64_9": 64,
         "square512": 64, "mul512": 64, "fwd512p": 64, "forward1024_0": 256,
         "backward1024_0": 256, "square2048": 128, "mul2048": 128, "fwd2048p": 128}
PLANS = [(("forward64_0", "forward64_9"), "512", ("backward64_9", "backward64_0")),
         (("forward1024_0",), "2048", ("backward1024_0",))]


def stage(lib, name, data, w, operand=None):
    kernel = getattr(lib, name)
    args = (kernel, None) if operand is None else (None, kernel)
    lib.run_groups(*args, data.ctypes.data, None if operand is None else operand.ctypes.data,
                   w.ctypes.data, 0, data.size // (4*V), GROUP[name])


def calculate(lib, plan, left, right, w):
    outer, tail, inverse = plan
    out = left.copy()
    for name in outer:
        stage(lib, name, out, w)
    if right is None:
        stage(lib, "square" + tail, out, w)
    else:
        operand = right.copy()
        for name in outer:
            stage(lib, name, operand, w)
        stage(lib, "fwd" + tail + "p", operand, w)
        stage(lib, "mul" + tail, out, w, operand)
    for name in inverse:
        stage(lib, name, out, w)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rns", choices=(2, 3), type=int, default=2)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--use-wi", action="store_true")
    args = parser.parse_args()
    defs, primes, generators = parameters(N, args.wide, args.rns, args.use_wi)
    w = twiddles(N, primes, generators)
    a = {0: -3, 15: 19, 2047: 7, 2048: -11, N-1: 5}
    b = {1: 9, 512: -3, N-2: 23}
    rng = np.random.default_rng(210512)
    dense = np.stack([rng.integers(0, p, N, dtype=np.uint32) for p in primes])
    with tempfile.TemporaryDirectory() as tmp:
        lib = compile_kernels(Path(tmp), defs)
        for label, left, right, expected in (
                ("dense square", dense, None, None),
                ("sparse square", sparse(N, primes, a), None, expected_product(N, primes, a, a)),
                ("sparse multiply", sparse(N, primes, a), sparse(N, primes, b), expected_product(N, primes, a, b))):
            base, candidate = (calculate(lib, plan, left, right, w) for plan in PLANS)
            np.testing.assert_array_equal(candidate, base)
            if expected is not None:
                np.testing.assert_array_equal(candidate, expected)
            print(f"{label}: PASS ({candidate.size} raw residues)", flush=True)


if __name__ == "__main__":
    main()
