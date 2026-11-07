"""Linux/x86-64 CPU-only execution of synthetic OpenCL kernel inputs."""
import ctypes
import os
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def parameters(n, wide, rns, use_wi):
    # Reproduce the production preamble from the host constants and the selected size.
    kernel = (ROOT / "ocl/kernel.cl").read_text()
    fixture = kernel.split("#if !defined(N_SZ)\n", 1)[1].split("#endif", 1)[0]
    defs = dict(re.findall(r"^#define\s+(\w+)\s+(\S+)", fixture, re.M))
    header = (ROOT / "src/transformGPU.h").read_text()
    suffix = "U" if wide else "S"
    primes = []
    for i in range(1, 4):
        mul, shift = re.search(rf"#define\s+P{i}{suffix}\s+\((\d+) \* \(uint32\(1\) << (\d+)\) \+ 1\)", header).groups()
        p = (int(mul) << int(shift)) + 1
        primes.append(p)
        defs[f"P{i}"] = f"{p}u"
        for stem in ("Q", "RSQ", "IM", "MFIM", "SQRTI", "ISQRTI"):
            defs[f"{stem}{i}"] = re.search(rf"#define\s+{stem}{i}{suffix}\s+(\d+u)", header)[1]
    exponent = n.bit_length() - 1
    defs.update(N_SZ=f"{n}u", LN_SZ=exponent, W_SHFT=f"{n}u", WI_SHFT=f"{n//2}u", RNS_SZ=rns,
                NORM_WG_SZ=1 << max(5, exponent//2 - 3))
    defs.pop("ALL_FUNC", None)
    for i, p in enumerate(primes, 1):
        defs[f"NORM{i}"] = f"{p - (p-1)//(n//2)}u"
    for k in ("INVP2_P1", "INVP3_P1", "INVP3_P2", "P1P2P3L", "P1P2P3H", "P1P2P3_2L", "P1P2P3_2H"):
        defs[k] = re.search(rf"#define\s+{k}{suffix}\s+(\d+u(?:l)?)", header)[1]
    if wide:
        defs["IS32"] = 1
    if use_wi:
        defs["USE_WI"] = 1
    return defs, primes[:rns], ([3, 7, 3] if wide else [3, 5, 31])[:rns]


def compile_kernels(directory, defs):
    kernel = ROOT / "ocl/kernel.cl"
    obj = directory / "kernels.o"
    library = directory / "kernels.so"
    subprocess.run([os.environ.get("CC_CL", "clang"), "-x", "cl", "-cl-std=CL1.2",
                    "-D__OPENCL_VERSION__=120", "-O2", "-fPIC",
                    *(f"-D{k}={v}" for k, v in defs.items()), "-c", str(kernel), "-o", str(obj)], check=True)
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-shared", "-fPIC",
                    "-Wl,-Bsymbolic", "-Wl,-z,defs", str(HERE / "workgroup.cpp"), str(obj), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    lib.run_groups.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_size_t] * 3
    lib.run_groups.restype = None
    return lib


def twiddles(n, primes, generators):
    tables = np.zeros((len(primes), n), dtype=np.uint32)
    for table, p, generator in zip(tables, primes, generators):
        for level in range(n.bit_length() - 2):
            s = 1 << level
            r = pow(generator, (p - 1) // (4*s), p)
            entries = np.empty(s, dtype=np.uint64)
            entries[0] = (r << 32) % p
            for k in range(level):
                width = 1 << k
                entries[width:2*width] = entries[:width] * pow(r, s//width, p) % p
            table[s:2*s] = entries
            table[n//2+s:n//2+2*s] = entries[::-1]
    return tables
