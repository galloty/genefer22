#!/usr/bin/env python3

import random

MASK = (1 << 32) - 1

PRIMES = (
    (2130706433, 2164260865),
    (2113929217, 2181038081),
    (2013265921, 2281701377),
)


def i32(x):
    x &= MASK
    return x if x < (1 << 31) else x - (1 << 32)


def submod(lhs, rhs, p):
    t = (lhs - rhs) & MASK
    return (t + (p if i32(t) < 0 else 0)) & MASK


def addmod(lhs, rhs, p):
    t = lhs + rhs
    return t - (p if t >= p else 0)


def mulmod(lhs, rhs, p, q):
    t = lhs * rhs
    lo = t & MASK
    hi = (t >> 32) & MASK

    mq = (lo * q) & MASK
    mp = ((mq * p) >> 32) & MASK

    return submod(hi, mp, p)


def reduce_lazy(x, p):
    return x - (p if x >= p else 0)


def canonical_forward(z0, z1, w, p, q):
    t = mulmod(z1, w, p, q)
    return (
        addmod(z0, t, p),
        submod(z0, t, p),
    )


def lazy_forward(z0, z1, w, p, q):
    a = reduce_lazy(z0, p)
    t = mulmod(z1, w, p, q)

    out0 = a + t
    out1 = a + p - t

    assert 0 <= out0 < 2 * p
    assert 0 <= out1 < 2 * p

    return out0, out1


def check_prime(p, q):
    assert 2 * p < (1 << 32)

    rng = random.Random(p ^ q ^ 0x47A9DFAD)

    edges = (
        0,
        1,
        p - 1,
        p,
        p + 1,
        2 * p - 2,
        2 * p - 1,
    )

    ws = (
        0,
        1,
        p // 2,
        p - 2,
        p - 1,
    )

    cases = 0

    for x in edges:
        for w in ws:
            got = mulmod(x, w, p, q)
            ref = mulmod(x % p, w, p, q)

            assert got == ref
            cases += 1

    for _ in range(100000):
        x = rng.randrange(0, 2 * p)
        w = rng.randrange(0, p)

        got = mulmod(x, w, p, q)
        ref = mulmod(x % p, w, p, q)

        assert got == ref
        cases += 1

    butterflies = 0

    for _ in range(200000):
        z0 = rng.randrange(0, 2 * p)
        z1 = rng.randrange(0, 2 * p)
        w = rng.randrange(0, p)

        lazy0, lazy1 = lazy_forward(
            z0,
            z1,
            w,
            p,
            q,
        )

        ref0, ref1 = canonical_forward(
            z0 % p,
            z1 % p,
            w,
            p,
            q,
        )

        assert lazy0 % p == ref0
        assert lazy1 % p == ref1

        butterflies += 1

    print(
        f"P={p} "
        f"LAZY_MONTGOMERY_CASES={cases} "
        f"LAZY_BUTTERFLY_CASES={butterflies} "
        "PASS"
    )


def main():
    for p, q in PRIMES:
        check_prime(p, q)

    print("SIGNED_PRIME_2P_LT_2POW32=PASS")
    print("LAZY_MONTGOMERY_INPUT_BOUND=PASS")
    print("LAZY_FORWARD_EQUIVALENCE=PASS")


if __name__ == "__main__":
    main()
