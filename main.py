import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io

p = argparse.ArgumentParser()
for k, d in [("tstop", 1500.0), ("noise", 2.5), ("alpha", 1.5), ("seed", 0), ("snr_db", -5.77), ("batch_size", 50), ("trial", 1)]:
    p.add_argument(f"--{k}", type=type(d), default=d)
args = p.parse_args()
t0 = time.perf_counter()

L = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ldpc.mat"))["LDPC"]
H, G = np.array(L["H"][0, 0]), np.array(L["G"][0, 0])
K, N = G.shape
P, B, a, dt = 8, args.batch_size, args.alpha, 1e-1
T = int(args.tstop / dt)
Hl = jnp.array(H)


def mimo_in(s):
    s = s[P:]
    return jnp.concatenate([s[1::2], s[::2]])


def e_ldpc(s):
    return a * jnp.sum(0.5 * (1.0 - jnp.prod(jnp.where(Hl, jnp.sign(s), 1.0), axis=1)))


def e_mimo(s, J, h):
    z = mimo_in(s)
    return -h @ z - 0.5 * (z @ ((J + J.T) @ z))


def grad(s, J, h, c):
    g = jnp.sign(s)
    dl = -0.5 * a * jnp.sum(jnp.where(Hl, jnp.prod(jnp.where(Hl, g, 1.0), axis=1)[:, None] * g[None, :], 0.0), axis=0)
    d = -h - (J + J.T) @ mimo_in(s)
    n = d.shape[0] // 2
    return c * dl + jnp.zeros_like(s).at[P:].set(jnp.stack([d[n:], d[:n]], axis=1).reshape(-1))


@jax.jit
def anneal(s, k, J, h):
    def step(carry, c):
        s, sc, k = carry
        s = s - dt * jax.vmap(grad, in_axes=(0, None, None, None))(s, J, h, c)
        k, sk = jax.random.split(k)
        s = jnp.clip(s + sc * jax.random.normal(sk, (B, N)) * jnp.sqrt(dt), -1.0, 1.0)
        return (s, sc - args.noise / T, k), None

    return jax.lax.scan(step, (s, args.noise, k), jnp.linspace(0.0, 1.0, T))[0][0]


for t in range(args.trial):
    ku, ks, kc = jax.random.split(jax.random.fold_in(jax.random.key(args.seed), t), 3)
    u = np.array(jax.random.randint(ku, (K,), 0, 2))
    c = np.array(u @ G % 2)[P:]
    x = np.where(c == 0, 1.0, -1.0)
    rng = np.random.default_rng(int(jax.random.randint(kc, (), 0, np.iinfo(np.int32).max)))
    Hm = np.kron(np.eye((N - P) // 8), (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2.0))
    s2 = 1.0 / 10.0 ** (args.snr_db / 10.0)
    y = Hm @ ((x[1::2] + 1j * x[::2]) / np.sqrt(2)) + (rng.standard_normal(Hm.shape[0]) + 1j * rng.standard_normal(Hm.shape[0])) * np.sqrt(s2 / 2.0)
    y = jnp.array(y)
    Ht = np.block([[Hm.real, -Hm.imag], [Hm.imag, Hm.real]])
    J = -0.5 * (Ht.T @ Ht)
    np.fill_diagonal(J, 0.0)
    h = np.sqrt(2.0) * (Ht.T @ np.concatenate([y.real, y.imag]))

    s = anneal(jax.random.uniform(ks, (B, N), minval=-1.0, maxval=1.0), jax.random.split(ks)[0], J, h)
    e = jnp.sum((s[:, :K] < 0) != u, axis=1)
    ef = jnp.sum((s[:, P:] < 0) != c, axis=1)
    el = jax.vmap(e_ldpc)(s)
    et = el + jax.vmap(lambda z: e_mimo(z, J, h))(s)
    i, j = jnp.argmin(el), jnp.argmin(et)
    out = {
        "solver": "unified_anneal_alpha_fixed_uday", "trial": t, "seed": args.seed, "tstop": T * dt,
        "noise": args.noise, "alpha": a, "jamieson_lambda": 0,
        "error_rate_by_gt": e[jnp.argmin(e)] / K, "error_rate_by_ldpc": e[i] / K, "error_rate_by_total_energy": e[j] / K,
        "full_error_rate_by_total_energy": ef[j] / N, "full_error_rate_by_ldpc": ef[i] / N,
        "average_error_rate": jnp.mean(e) / K, "mimo_snr_db": args.snr_db,
    }
    print("\n".join(f"{k}: {v}" for k, v in out.items()) + "\n#")

print(f"total_time: {time.perf_counter() - t0:.2f}")
