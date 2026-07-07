"""
Sanity checks for the Roseidon implementation.
"""

load("roseidon.sage")

ros = Roseidon()
ring = ros.ring
Rq = ring.quotient_ring
set_random_seed(0)


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


# The splitting structure required: 4 slots of degree 4.
check("Frog ring splits into 4 slots of degree 4",
      ring.tau == 4 and ring.delta == 4
      and all(f.degree() == 4 and f.is_irreducible() for f in ring.slot_moduli))

# S-box is invertible on R_q ~ (F_{q^4})^4: x -> x^alpha undone by x -> x^(alpha^-1 mod q^4 - 1).
alpha_inv = inverse_mod(ALPHA, Q ^ DELTA - 1)
z = ring.random_element()
check("S-box invertible", (ros.sbox(z)) ^ alpha_inv == z)

# Each slot matrix is MDS: isAllInvertible from the GRS21 tool checks all proper square submatrices (sizes 2..t-1); 1x1 entries are nonzero by the Cauchy construction, and we check full-size invertibility directly.
check("slot matrices are MDS",
      all(isAllInvertible(Mk, T) and Mk.is_invertible()
          for Mk in ros.slot_matrices))

# GRS21 subspace-trail tests (minpoly condition + algorithms 1-3 from grs21.sage)
check("slot matrices pass the GRS21 tests",
      all(passes_grs_tests(Mk, K)
          for Mk, K in zip(ros.slot_matrices, ring.slot_fields)))

# MixColumns equivalence
state = [ring.random_element() for _ in range(T)]
mixed = [sum((ros.M[i][j] * state[j] for j in range(T)), Rq(0)) for i in range(T)]
state_ntt = [ring.element_to_crt(s) for s in state]
mixed_ntt = [ring.element_to_crt(m) for m in mixed]
mix_ok = all(
    vector([mixed_ntt[i][k] for i in range(T)])
    == ros.slot_matrices[k] * vector([state_ntt[i][k] for i in range(T)])
    for k in range(TAU)
)
check("MixColumns = slot-wise Cauchy matrices", mix_ok)


# Two states differing in a single NTT slot must differ in all t x tau slots
one_slot = [ring.slot_fields[0](1)] + [K(0) for K in ring.slot_fields[1:]]
delta_elt = ring.crt_to_element(one_slot)   # difference in one slot of one row
state_a = [ring.random_element() for _ in range(T)]
state_b = [state_a[0] + delta_elt] + state_a[1:]
for rnd in range(2):
    state_a = ros._round(state_a, rnd, full=True)
    state_b = ros._round(state_b, rnd, full=True)
diff_ok = all(x != y for x, y in zip(
    (v for s in state_a for v in ring.element_to_crt(s)),
    (v for s in state_b for v in ring.element_to_crt(s))))
check("full diffusion after 2 rounds", diff_ok)

m1 = [Rq(123), Rq(456)]
m2 = [Rq(123), Rq(457)]


# Padding 
check("padding: m vs m || 0", ros.hash(m1) != ros.hash(m1 + [Rq(0)]))

# Multi-block
long_msg = [Rq(i) for i in range(10)]
out = ros.hash(long_msg, out_len=5)
check("long message, 5 output elements", len(out) == 5)

print()
print("All tests passed.")
print()
print("Example digest (regression vector):")
print(f"  message: [123, 456] as constants in R_q")
print(f"  hash   : {ros.hash(m1)[0].lift()}")
