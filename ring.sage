"""
Boiler plate code for power-of-two cyclotomic rings
"""


class CyclotomicRingAutomorphisms:
    def __init__(self, p, N):
        if not is_prime(p):
            raise ValueError(f"p = {p} must be prime")
        if not (N > 0 and (N & (N - 1)) == 0):
            raise ValueError(f"N = {N} must be a power of 2")

        self.p = p
        self.N = N
        self.Zp = GF(p)

        # The polynomial ring Z_p[X] and the quotient R_q = Z_p[X]/(X^N + 1).
        self.R = PolynomialRing(self.Zp, 'X')
        X = self.R.gen()
        self.cyclotomic_poly = X ^ N + 1
        self.quotient_ring = self.R.quotient(self.cyclotomic_poly)

        self._compute_components()
        self._compute_automorphism_group()

    def _compute_components(self):
        self.delta = Mod(self.p, 2 * self.N).multiplicative_order()
        self.tau = self.N // self.delta

        # Deterministic primitive 2*tau-th root of unity in Z_p.
        g = self.Zp.multiplicative_generator()
        self.omega = g ^ ((self.p - 1) // (2 * self.tau))
        if self.omega.multiplicative_order() != 2 * self.tau:
            raise RuntimeError("Computed root is not a primitive 2*tau-th root of unity")

        X = self.R.gen()
        self.slot_moduli = [
            X ^ self.delta - self.omega ^ (2 * j + 1) for j in range(self.tau)
        ]
        assert prod(self.slot_moduli) == self.cyclotomic_poly

        self.slot_fields = [
            GF(self.p ^ self.delta, 'a', modulus=f) for f in self.slot_moduli
        ]

    def _compute_automorphism_group(self):
        self.automorphisms = {}
        self.automorphism_exponents = []
        for k in range(1, 2 * self.N):
            if gcd(k, 2 * self.N) == 1:
                self.automorphism_exponents.append(k)
                self.automorphisms[k] = self._create_automorphism(k)

    def _create_automorphism(self, k):
        def automorphism(poly):
            if isinstance(poly, (int, Integer)):
                return self.quotient_ring(poly)
            if hasattr(poly, 'lift'):
                poly = poly.lift()
            # Send each monomial X^i to X^(i*k), using X^N = -1.
            result = self.R(0)
            for i, coeff in enumerate(poly.list()):
                if coeff != 0:
                    power = (i * k) % (2 * self.N)
                    if power >= self.N:
                        result -= coeff * self.R.gen() ^ (power - self.N)
                    else:
                        result += coeff * self.R.gen() ^ power
            return self.quotient_ring(result)
        return automorphism

    def apply_automorphism(self, element, k):
        if k not in self.automorphisms:
            raise ValueError(f"k = {k} is not a valid automorphism exponent")
        return self.automorphisms[k](element)

    def element_to_crt(self, element):
        if hasattr(element, 'lift'):
            poly = element.lift()
        else:
            poly = self.R(element)
        return [poly(K.gen()) for K in self.slot_fields]

    def crt_to_element(self, crt_components):
        if len(crt_components) != self.tau:
            raise ValueError("Number of CRT components must equal number of slots")

        remainders = []
        for value in crt_components:
            if hasattr(value, 'polynomial'):
                remainders.append(self.R(value.polynomial().list()))
            else:
                remainders.append(self.R(value))
        result = CRT(remainders, self.slot_moduli) % self.cyclotomic_poly
        return self.quotient_ring(result)

    def random_element(self, degree=None):
        """Generate a uniformly random element of the ring."""
        if degree is None:
            degree = self.N - 1
        coeffs = [self.Zp.random_element() for _ in range(degree + 1)]
        return self.quotient_ring(sum(coeffs[i] * self.R.gen() ^ i for i in range(len(coeffs))))


def generate_mds_matrix(xs, ys, K):
    t = len(xs)
    if len(ys) != t:
        raise ValueError("xs and ys must have the same length")
    M = Matrix(K, t, t)
    for i in range(t):
        for j in range(t):
            denominator = K(xs[i]) + K(ys[j])
            if denominator == 0:
                raise ValueError(f"x[{i}] + y[{j}] = 0 in {K}")
            M[i, j] = K(1) / denominator
    return M
