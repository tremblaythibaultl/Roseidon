# Roseidon over the Frog ring

A simple SageMath reference implementation of **Roseidon**, a sponge-based hash function operating on cyclotomic rings.

## Files

- `ring.sage`: Boilerplate code implementing basic algebraic components 
- `roseidon.sage`: The Roseidon permutation and sponge 
- `grs.sage`: The linear-layer sanity check tool of [GRS21](https://extgit.isec.tugraz.at/krypto/linear-layer-tool/-/blob/master/code/algorithms_gfp.sage) (unmodified)
- `test_roseidon.sage`: Sanity checks (run `sage test_roseidon.sage` from this directory) 

## The instantiation 

The Frog ring: `d = 16` with the 64-bit prime `q = 15912092521325583641`.
Since `q ≡ 9 (mod 16)`, the ring splits into `tau = 4` NTT slots of degree `delta = 4`.

The sponge parameters are `(RF, RP, alpha, r, c) = (12, 22, 7, 3, 1)`, so the state has `t = r + c = 4 = tau` ring elements for a square NTT state matrix.

## The construction

The permutation follows the HADES design strategy: `RF = 12` full rounds split around `RP = 22` partial rounds.

All public constants are derived deterministically from a seed string with
SHAKE-128 and rejection sampling, so every `Roseidon()` instance is the same
function.

## Running

```bash
cd roseidon
sage test_roseidon.sage   # sanity checks + an example digest
```