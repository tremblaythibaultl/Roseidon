"""
Roseidon: a sponge-based hash function over cyclotomic rings R_q = Z_q[X]/(X^d + 1).

The permutation follows the HADES design strategy: RF full rounds and RP partial rounds, where each round comprises:
1. add round constants
2. S-box x -> x^alpha (on the whole state in full rounds, on state[0] only in partial ones)
3. linear layer
    a. MixColumns: state <- M * state  with M in R_q^{t x t} an MDS matrix 
    b. ShiftRows: state[i] <- sigma_{2i+1}(state[i])

The sponge has state R_q^t with t = r + c, absorbs/squeezes r ring elements each permutation.

Instantiation: the "Frog" ring, i.e. d = 16 with the 64-bit prime q below.  Since q = 9 (mod 16), the ring splits into tau = 4 NTT slots of degree delta = 4.
The state has t = tau = 4 ring elements (the square NTT state matrix required by Section 4.1.2), and (RF, RP, alpha, r, c) = (12, 22, 7, 3, 1).  The S-box is invertible on every slot since gcd(alpha, q^delta - 1) = 1, and no automorphism in the schedule inverts the power map.

Usage (from this directory):
    sage: load("roseidon.sage")
    sage: ros = Roseidon()
    sage: digest = ros.hash([ros.ring.quotient_ring(123)])
"""

import hashlib

load("ring.sage")

# Parameters 
Q = 15912092521325583641   # 64-bit "Frog" prime; q = 9 (mod 16)
D = 16                     # ring degree d of the Frog ring
TAU = 4                    # number of NTT slots
DELTA = 4                  # slot degree: each slot is a copy of F_{q^4}
T = 4                      # state size t in ring elements; the design needs t = tau
RATE = 3                   # r: ring elements absorbed/squeezed per permutation call
CAPACITY = 1               # c: hidden part of the state (t = r + c)
ALPHA = 7                  # S-box exponent; invertible since gcd(7, q^4 - 1) = 1
R_FULL = 12                # RF: total full rounds (RF/2 at the start, RF/2 at the end)
R_PARTIAL = 22             # RP: partial rounds (middle of the permutation)

# grs21.sage is the original tool released with the [GRS21] paper
with open("grs21.sage") as _grs_file:
    _grs_source = _grs_file.read()
exec(preparse(_grs_source[_grs_source.index("def print_matrix_format"):
                          _grs_source.index("test_algorithms = [")]))

# Globals read by the grs21.sage functions.
t = T   # state size
s = 1   # S-boxes per partial round: one, at state position 0

def passes_grs_tests(M, field):
    # [GRS21] tests for a candidate linear-layer matrix M over `field`: check_minpoly_condition is the paper's sufficient condition, and algorithms 1-3 are ran as cross-check.
    global F
    F = field   # the grs21.sage functions read the field from the global F
    return (check_minpoly_condition(M)
            and algorithm_1(M)[0] and algorithm_2(M)[0] and algorithm_3(M)[0])


class Roseidon:
    def __init__(self, seed="Roseidon-v1"):
        assert T == RATE + CAPACITY
        self.ring = CyclotomicRingAutomorphisms(Q, D)
        assert self.ring.tau == TAU and self.ring.delta == DELTA
        assert T == TAU                                  # t = tau for a square NTT state matrix
        assert gcd(ALPHA, Q ^ DELTA - 1) == 1            # S-box invertibility field-wise 
        assert all((ALPHA * (2 * i + 1)) % (2 * D) != 1 for i in range(T)) # S-box non-invertibility by automorphisms

        # All public constants (round constants, MDS matrices) are derived deterministically from a seed via the SHAKE-128 XOF.
        self._xof = hashlib.shake_128(seed.encode()).digest(1 << 20)
        self._xof_pos = 0
        self._generate_round_constants()
        self._generate_mds_matrix()

    def _next_field_element(self):
        # Uniform from rejection sampling
        while True:
            chunk = self._xof[self._xof_pos:self._xof_pos + 8]
            self._xof_pos += 8
            if len(chunk) < 8:
                raise RuntimeError("XOF stream exhausted")
            v = int.from_bytes(chunk, 'big')  # q > 2^63, so accept ~86% of draws
            if v < Q:
                return self.ring.Zp(v)

    def _next_ring_element(self):
        X = self.ring.R.gen()
        return self.ring.quotient_ring(
            sum(self._next_field_element() * X ^ k for k in range(D))
        )

    def _next_slot_element(self, K):
        return sum(K(self._next_field_element()) * K.gen() ^ k for k in range(DELTA))

    def _generate_round_constants(self):
        # The constants must not be constant polynomials and must not all lie in a proper subring of R_q; uniformly random polynomials satisfy this with overwhelming probability, and we assert the first condition explicitly.
        self.round_constants = [
            [self._next_ring_element() for _ in range(T)]
            for _ in range(R_FULL + R_PARTIAL)
        ]
        assert any(rc.lift().degree() > 0 for rcs in self.round_constants for rc in rcs)

    def _generate_mds_matrix(self):
        # In the NTT domain the state is a t x tau matrix over F_{q^delta}, and step 1 multiplies column j by its own t x t MDS matrix M_j.  We draw the M_j as independent random Cauchy matrices over the slot fields F_{q^4} and combine them into a single ring matrix M via the inverse NTT applied entrywise: NTT(M[i][j]) = ((M_1)[i][j], ..., (M_tau)[i][j]).  
        # Random F_{q^4} entries also ensure the entries of M do not all lie in the subfield F_q, or any other subring of R_q.

        # We additionally require the M_j to pass the subspace-trail tests of [GRS21]. This is implemented with rejection sampling where a candidate Cauchy matrix is kept only if it satisfies the minimal-polynomial condition and passes algorithms 1-3 of the GRS21 tool.
        self.slot_matrices = []
        for j in range(TAU):  # one Cauchy matrix per NTT slot
            K = self.ring.slot_fields[j]
            while True:
                xs, ys = [], []
                while len(xs) < T:
                    v = self._next_slot_element(K)
                    if v not in xs:
                        xs.append(v)
                while len(ys) < T:
                    v = self._next_slot_element(K)
                    if v not in ys and all(x + v != 0 for x in xs):
                        ys.append(v)
                candidate = generate_mds_matrix(xs, ys, K)
                if passes_grs_tests(candidate, K):
                    break
            self.slot_matrices.append(candidate)

        # Inverse NTT
        self.M = [
            [
                self.ring.crt_to_element(
                    [self.slot_matrices[k][i, j] for k in range(TAU)]
                )
                for j in range(T)
            ]
            for i in range(T)
        ]

    def sbox(self, x):
        return x ^ ALPHA

    def linear_layer(self, state):
        # MixColumns
        mixed = [
            sum((self.M[i][j] * state[j] for j in range(T)),
                self.ring.quotient_ring(0))
            for i in range(T)
        ]
        # ShiftRows
        return [
            self.ring.apply_automorphism(mixed[i], 2 * i + 1)
            for i in range(T)
        ]

    def _round(self, state, round_index, full):
        # Add round constants
        state = [state[i] + self.round_constants[round_index][i] for i in range(T)]
        # Apply S-boxes
        if full:
            state = [self.sbox(x) for x in state]
        else:
            state = [self.sbox(state[0])] + state[1:]
        # Apply linear layer
        return self.linear_layer(state)

    def permutation(self, state):
        state = list(state)
        r = 0
        for _ in range(R_FULL // 2):
            state = self._round(state, r, full=True)
            r += 1
        for _ in range(R_PARTIAL):
            state = self._round(state, r, full=False)
            r += 1
        for _ in range(R_FULL // 2):
            state = self._round(state, r, full=True)
            r += 1
        return state


    def pad(self, message):
        padded = [self.ring.quotient_ring(m) for m in message]
        padded.append(self.ring.quotient_ring(1))
        while len(padded) % RATE != 0:
            padded.append(self.ring.quotient_ring(0))
        return padded

    def hash(self, message, out_len=1):
        padded = self.pad(message)
        state = [self.ring.quotient_ring(0)] * T

        for block_start in range(0, len(padded), RATE):
            for i in range(RATE):
                state[i] += padded[block_start + i]
            state = self.permutation(state)

        output = []
        while True:
            output.extend(state[:RATE])
            if len(output) >= out_len:
                return output[:out_len]
            state = self.permutation(state)
