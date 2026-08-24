"""
security_check.py

Checks that a Roseidon parameter set provides lambda-bit security against the
algebraic attacks analysed in algebraic_attacks.tex, namely the interpolation
attack and the Groebner basis attack on CICO-k of Section A.2, whose cost is
governed by the ideal degree D_I.  The older Groebner estimates (full
permutation, subspace trails, single S-boxes) are superseded by that analysis
but are still evaluated here for reference.

Each attack is implemented as a function returning its estimated cost in bits
(log2 of the number of field operations).  The parameter set is considered
secure when every attack costs at least `lam` bits.  Where an attack is more
naturally framed as a lower bound on the number of rounds (interpolation), the
check compares the chosen round count against that bound instead.

CICO granularity.  The sponge absorbs and squeezes whole R_q elements, and each
R_q element is tau NTT words of FF_{q^delta}.  Any CICO instance reachable from
the sponge therefore constrains at least chi_min = c * tau output words, so
CICO-1 does not arise (Lemma 1 assumes tau > 2, hence tau >= 4). 
Cf. Bak et al., ToSC/CiC "Algebraic CheapLunch", Section 1: CICO-1 is the
relevant problem only when the capacity is a single field element, and CICO-k
with k > 1 is the relevant one otherwise.
"""

from math import log2, ceil, sqrt

# ====================  PARAMETERS  ====================
# Round structure of the permutation pi.
R_f   = 6              # full rounds on *each* side of the partial-round block
R_F   = 2 * R_f        # total number of full rounds
R_P   = 22             # number of partial rounds
R     = R_F + R_P      # total number of rounds

# Sponge layout.
r     = 3              # rate     (number of ring words absorbed/squeezed)
c     = 1              # capacity (number of ring words)
t     = r + c          # state size (number of ring words)

# S-box and target security.
alpha = 7              # S-box exponent: x -> x^alpha
lam   = 128            # target security level in bits

# --- auxiliary parameters required by the bounds (see algebraic_attacks.tex) ---
# These are not part of the round/sponge layout above but are needed to evaluate
# the attack complexities; adjust them to match the chosen ring R_q.
# The following are for the frog ring R_q := Z_q[X]/(X^16 + 1) with q = 15912092521325583641.
tau   = 4                       # number of NTT factors of R_q  (R_q ~ FF_{q^delta}^tau)
delta = 4                       # extension degree of each NTT factor (each factor is FF_{q^delta})
q     = 15912092521325583641    # prime modulus
N     = 1                       # round delay for full diffusion: deg_pi(R) >= alpha^{R-N}
R_pp  = N+1                     # R'': extra rounds accounting for the partial S-box layer

# Smallest number of FF_{q^delta} output words a CICO instance reachable from the
# sponge can constrain.  The capacity is c whole R_q elements and each R_q element
# is tau NTT words, so chi >= c * tau; see the module docstring. 
CHI_MIN = c * tau


# ====================  ALGEBRAIC DEGREE  ====================
def deg_pi(rounds):
    """Algebraic degree of `rounds` rounds of pi.

    With the power-map S-box x -> x^alpha and full diffusion after N rounds,
    deg_pi(R) = alpha^floor(R - N)  (cf. algebraic_attacks.tex)."""
    return alpha ** (rounds - N)


def log2_deg_pi(rounds):
    """log2(deg_pi(rounds)), computed directly to avoid huge intermediates."""
    return (rounds - N) * log2(alpha)


# ====================  INTERPOLATION ATTACK  ====================
# Ref: Jakobsen-Knudsen [JK97], analysed over the NTT
# representation (over FF_{q^delta}).  The attack is infeasible once the
# interpolation polynomial has 2^128 non-zero monomials.
# We force the monomials X^d to be present by requiring log_{deg_pi(2)}(d)+1
# rounds, to which we add log_{deg_pi(2)}(t*tau)+1 rounds to make all
# multivariate monomials of individual degree d appear.
# Finally, we add N rounds to account for the partial S-box layer.
# This totals *at least* 2^128 monomials in the interpolation polynomial,
# which is a lower bound on the attack cost.
def interpolation_min_rounds():
    """Lower bound on R for the interpolation polynomial to be sufficiently dense.

    Pick d so that d^(t*tau) >= 2^lambda, i.e. log2(d) = lambda/(t*tau).  Then
    require enough rounds for all multivariate monomials of individual degree d
    to appear, plus N rounds for the partial S-box layer:
        R >= (log_{deg_pi(2)}(d) + 1) + (log_{deg_pi(2)}(t*tau) + 1) + N."""
    base = log2(alpha)                              # log2(deg_pi(2)) = log2(alpha)
    log2_d = lam / (t * tau)                        # d = 2^(lambda/(t*tau))
    term_d = ceil(log2_d / base) + 1                # log_{deg_pi(2)}(d) + 1
    term_state = ceil(log2(t * tau) / base) + 1     # log_{deg_pi(2)}(t*tau) + 1
    return term_d + term_state + N



# ====================  GROEBNER BASIS ATTACKS  ====================
# The primary estimate is groebner_cico_k_bits() below, which follows the
# ideal-degree analysis of [BBL+24, BBB+25] (Section A.2, Eq. (11)-(12)) and
# supersedes the older estimates of [BCD+20, GKRS22, ABM24].  The latter are
# retained below for reference; they are looser and do not bind.
#
# Older estimates: cost of computing the basis ~ binom(m + Dreg, Dreg)^omega
# with omega = 2, lower-bounded (see algebraic_attacks.tex) by deg^(2 * #vars).

def log2_ideal_degree(k=None):
    """log2 of the weighted Bezout bound on the ideal degree, Section A.2 Eq. (11).

        D_I <= alpha^(k*R_F + tau*R_P)

    obtained from the CICO modeling with k input variables of weight alpha and,
    per partial round, tau variables of weight alpha^(R_f) -- one per NTT slot,
    since a partial round of Roseidon applies the power map to a whole R_q
    element.  `k` is the number of constrained output words; the default is the
    smallest value reachable from the sponge, k = CHI_MIN = c*tau.

    NB: this bound is fixed by the *weights* of the modeling and deliberately
    does not involve deg_pi() or N.  The alpha^(R-1) degree lag from the
    preamble of Section A.2 plays no role here: full rounds apply the S-box to
    every word, so the weighted degree multiplies by exactly alpha per full
    round, and the partial rounds carry their degree in the fresh y-variables
    rather than accumulating it.  Do not substitute log2_deg_pi() here."""
    if k is None:
        k = CHI_MIN
    return (k * R_F + tau * R_P) * log2(alpha)


def groebner_cico_k_bits(k=None):
    """Groebner basis attack on CICO-k, Section A.2 Eq. (12).
    """
    return 2 * log2_ideal_degree(k) - R_F * log2(alpha)


def groebner_full_permutation_bits():
    """Groebner attack on the full R-round permutation over the NTT state.

    chi = number of unknown inputs = number of known outputs.  Only
    CHI_MIN <= chi <= t*tau is reachable from the sponge (see module docstring).
    The cost ~ deg_pi(R)^(2*chi) is increasing in chi, so the attacker's best
    reachable choice is chi = CHI_MIN.

    The chi = 1 case, which admits the cheaper univariate treatment of
    [ToSC:GKRS22], is handled separately by groebner_cico1_permutation_bits()
    and is not a pass/fail criterion here."""
    log_d = log2_deg_pi(R)
    if CHI_MIN <= 1:
        return groebner_cico1_permutation_bits()
    return 2 * CHI_MIN * log_d


def groebner_cico1_permutation_bits():
    """Permutation-level CICO-1: cost of solving one univariate equation.

    It is NOT an attack on the sponge: a solution zeroes a single output word,
    whereas a collision or preimage requires the whole c-element capacity, i.e.
    CHI_MIN = c*tau words, to be controlled simultaneously.  
    The same test applied to the Poseidon instance
    benchmarked in Section 5 (R_F = 8, R_P = 22, alpha = 7, capacity 4 field
    elements) returns ~97 bits, which likewise reflects the scope of the test
    rather than a weakness."""
    log_d = log2_deg_pi(R)
    return log_d + 2 * log2(log_d) if log_d >= 2 else log_d


def groebner_subspace_bits():
    """Groebner attack exploiting subspace trails [C:BCDELL20].

    A coset of the subspace S^{r_trail} stays affine through r_trail partial
    rounds, turning non-linear equations into linear ones.  With an MDS linear
    layer the longest non-trivial trail has r_trail = t - 1.  Cost
    ~ deg_pi((R_F + R_P - r_trail)/2)^(2*(t + chi)), cheapest at chi = 1.

    NB: r_trail is the *trail length* (<= t-1), not the sponge rate r."""
    r_trail = t - 1
    mid = (R_F + R_P - r_trail) // 2
    chi = 1
    return 2 * (t + chi) * log2_deg_pi(mid)


def groebner_single_sbox_bits():
    """Groebner attack relating each S-box input to its output [acisp/AshurBM24].

    System of n = t*(R_F - 1) + R_P + r equations of degree deg_pi(2) = alpha,
    cost ~ deg_pi(2)^(2*n).  Here r is the sponge rate.  Pessimistic: ignores
    the number of S-boxes active in the NTT representation."""
    n_eq = t * (R_F - 1) + R_P + r
    return 2 * n_eq * log2_deg_pi(2)


# ====================  STATISTICAL ATTACKS  ====================

# ====================  DIFFERENTIAL ATTACKS  ====================
# Cost of running existing differential attacks on AES and related constructions

def differential_bits():
    """Bits of security against general differential attack

    Computes the single max differential probability as single_prob and considers
    that an MDS matrix activates t+1 S-boxes per pair of rounds.
    """
    round_pairs = 2 * (R_f//2)
    d = delta * tau
    single_prob = (alpha - 1)**tau / (q**d)
    return -1 * (round_pairs * (t + 1)) * log2(single_prob)

def jnp_bits():
    """Cost of (cheaper) outbound phase of [JNP14] 8 round rebound attack on AES

    This ensures that the two tau to 1 truncated differentials do not occur with
    good probability for the outbound phase rendering the attack inefficient.
    This is a very conservative underestimate of the attack.
    """
    exponent = 2 * delta * (tau - 1)
    return log2(q ** exponent)

def bl_bits():
    """Cost of (cheaper) outbound phase of [BL23] 6 round rebound attack on AES
    """
    exponent = 2 * delta * (tau - 1)
    return 0.5 * log2(2 * q ** exponent)

def subspace_trail_min_rounds():
    """This tests whether there are enough rounds for the subspace trail attacks on
    AES-like structures [GRR16] works

    """
    min_rounds = {
        "partial": 4,
        "full": 5,
    }
    return min_rounds
    
# ====================  LINEAR ATTACKS  ====================
# Cost of running linear attacks on Roseidon

def linear_bits():
    """This costs the averaging linear attack of [BSV07] over non-binary fields that
    optimizes the attack of [GLP06]

    It computes the linear potential under a dominant trail/Markov cipher assumption, costs
    the attack of [GLP06] and then applies the improvement factor of q^2/2 due to [BSV07]
    """
    single_LP = ((alpha - 1) / q)**2
    improvement = q ** 2 / 2
    return -1 * (log2(improvement) + R_F * log2(single_LP))


# ====================  REPORT  ====================
def _report_bits(name, bits):
    ok = bits >= lam
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<34} {bits:8.1f} bits  (target {lam})")
    
    if name == "rebound" and not ok:
        print(f'There may be an attack on R_F <= 7 rounds')
    if name == "boomerang" and not ok:
        print(f'There may be an attack on R_F <= 6 rounds')
        
    return ok


def _report_rounds(name, achieved, required, var_name):
    ok = achieved >= required
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<34} {var_name}={achieved} >= {required} rounds")
    return ok


def check_security():
    """Run every algebraic-attack check; return True iff all pass."""
    print(f"Roseidon parameter check (lambda = {lam}, alpha = {alpha}, "
          f"r = {r}, c = {c}, R = {R} = {R_F} full + {R_P} partial)")
    print(f"  smallest CICO instance reachable from the sponge: chi >= "
          f"c*tau = {CHI_MIN} words of FF_(q^{delta})")
    results = [
        _report_rounds("interpolation", R, interpolation_min_rounds(), "R"),
        _report_bits(f"groebner (CICO-k, k={CHI_MIN})", groebner_cico_k_bits()),
        _report_bits(f"groebner (full perm., chi={CHI_MIN})", groebner_full_permutation_bits()),
        _report_bits("groebner (subspace trails)", groebner_subspace_bits()),
        _report_bits("groebner (single S-boxes)", groebner_single_sbox_bits()),

        _report_bits("differential", differential_bits()),
        _report_bits("rebound", jnp_bits()),
        _report_bits("boomerang", bl_bits()),
        _report_rounds("partial subspace trail", R_P, subspace_trail_min_rounds()["partial"], "R_P"),
        _report_rounds("full subspace trail", R_F, subspace_trail_min_rounds()["full"], "R_F"),
        _report_bits("linear", linear_bits()),
    ]
    secure = all(results)
    print(f"\n=> parameters are {'SECURE' if secure else 'NOT secure'} "
          f"against the implemented algebraic attacks.")
    return secure


if __name__ == "__main__":
    check_security()

