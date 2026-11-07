"""Independent signed negacyclic polynomial oracle for raw NTT output."""
import numpy as np


def sparse(n, primes, terms):
    data = np.zeros((len(primes), n), dtype=np.uint32)
    for i, p in enumerate(primes):
        for j, value in terms.items():
            data[i, j] = value % p
    return data


def expected_product(n, primes, a, b):
    terms = {}
    for i, x in a.items():
        for j, y in b.items():
            index = (i + j) % n
            terms[index] = terms.get(index, 0) + (x*y if i+j < n else -x*y)
    # Raw inverse state precedes normalize1's Montgomery divide by N/2.
    return sparse(n, primes, {i: v * (n//2) * (1 << 32) for i, v in terms.items()})
