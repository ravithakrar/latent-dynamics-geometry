"""
Synthetic demonstration for the EM/LDS theory chapter.

Implements the chapter's own equations (Kalman filter, RTS smoother, closed-form
M-step) in plain numpy, simulates data from a known ground-truth model, and
produces the three figures used in the demonstration section:

    fig_demo_loglik.png     log-likelihood against iteration, several inits
    fig_demo_recovery.png   entries of A vs A_hat, and their spectra
    fig_demo_latents.png    true vs smoothed latents after gauge alignment

Deliberately no jax and no project imports: the figures are produced by the
formulas exactly as written in the chapter, so the script doubles as a check on
them. Fixed seed, so it reproduces.

Run:  python3 demo_synthetic.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotstyle as ps          # shared thesis figure style (SciencePlots base)
ps.apply()


SEED = 0
KDIM, PDIM, T, N = 4, 8, 40, 60     # latent dim k, observation dim p, length, trajectories
RHO = 0.9                            # spectral radius of the ground-truth A
N_ITERS, N_INITS = 100, 5
OUT = "figures"

rng = np.random.default_rng(SEED)


# ground truth
def symmetric_A(k, rho, rng):
    """A = A^T rescaled to spectral radius rho."""
    M = rng.standard_normal((k, k))
    A = 0.5 * (M + M.T)
    return A * (rho / np.max(np.abs(np.linalg.eigvals(A))))


A_true = symmetric_A(KDIM, RHO, rng)
C_true = rng.standard_normal((PDIM, KDIM)) / np.sqrt(KDIM)
Q_true = 0.05 * np.eye(KDIM)
R_true = 0.10 * np.eye(PDIM)
mu0_true = np.zeros(KDIM)
V0_true = np.eye(KDIM)


def simulate(A, C, Q, R, mu0, V0, rng):
    x = np.zeros((N, T, KDIM))
    y = np.zeros((N, T, PDIM))
    Lq, Lr, L0 = (np.linalg.cholesky(m) for m in (Q, R, V0))
    for n in range(N):
        x[n, 0] = mu0 + L0 @ rng.standard_normal(KDIM)
        for t in range(1, T):
            x[n, t] = A @ x[n, t - 1] + Lq @ rng.standard_normal(KDIM)
        for t in range(T):
            y[n, t] = C @ x[n, t] + Lr @ rng.standard_normal(PDIM)
    return x, y


X, Y = simulate(A_true, C_true, Q_true, R_true, mu0_true, V0_true, rng)


# E-step: filter then smoother
def e_step(y, A, C, Q, R, mu0, V0):
    xp = np.zeros((T, KDIM)); Vp = np.zeros((T, KDIM, KDIM))
    xf = np.zeros((T, KDIM)); Vf = np.zeros((T, KDIM, KDIM))
    ll = 0.0
    for t in range(T):
        if t == 0:
            xp[t], Vp[t] = mu0, V0
        else:
            xp[t] = A @ xf[t - 1]
            Vp[t] = A @ Vf[t - 1] @ A.T + Q
        S = C @ Vp[t] @ C.T + R
        S = 0.5 * (S + S.T)
        e = y[t] - C @ xp[t]
        Ls = np.linalg.cholesky(S)
        a = np.linalg.solve(Ls, e)
        ll += -0.5 * (PDIM * np.log(2 * np.pi)
                      + 2 * np.sum(np.log(np.diag(Ls))) + a @ a)
        Kg = np.linalg.solve(S, C @ Vp[t]).T          # V C' (C V C' + R)^{-1}
        xf[t] = xp[t] + Kg @ e
        Vf[t] = (np.eye(KDIM) - Kg @ C) @ Vp[t]
        Vf[t] = 0.5 * (Vf[t] + Vf[t].T)

    xs = xf.copy(); Vs = Vf.copy(); J = np.zeros_like(Vf)
    for t in range(T - 2, -1, -1):
        J[t] = np.linalg.solve(Vp[t + 1], A @ Vf[t]).T
        xs[t] = xf[t] + J[t] @ (xs[t + 1] - A @ xf[t])
        Vs[t] = Vf[t] + J[t] @ (Vs[t + 1] - Vp[t + 1]) @ J[t].T
        Vs[t] = 0.5 * (Vs[t] + Vs[t].T)

    Vlag = np.zeros((T, KDIM, KDIM))                  # V_{t,t-1} = V_t J_{t-1}'
    for t in range(T - 1, 0, -1):
        Vlag[t] = Vs[t] @ J[t - 1].T
    return xs, Vs, Vlag, ll


# M-step: closed-form updates
def em_iteration(Y, A, C, Q, R, mu0, V0):
    S_cross = np.zeros((KDIM, KDIM)); S_prev = np.zeros((KDIM, KDIM))
    S_yx = np.zeros((PDIM, KDIM));    S_all = np.zeros((KDIM, KDIM))
    S_yy = np.zeros((PDIM, PDIM));    S_next = np.zeros((KDIM, KDIM))
    x0s = np.zeros((N, KDIM));        V0s = np.zeros((N, KDIM, KDIM))
    ll_total = 0.0
    for n in range(N):
        xs, Vs, Vlag, ll = e_step(Y[n], A, C, Q, R, mu0, V0)
        ll_total += ll
        Pt = Vs + np.einsum("tk,tl->tkl", xs, xs)
        Pc = Vlag + np.einsum("tk,tl->tkl", xs, np.roll(xs, 1, axis=0))
        S_cross += Pc[1:].sum(0)
        S_prev  += Pt[:-1].sum(0)
        S_next  += Pt[1:].sum(0)
        S_all   += Pt.sum(0)
        S_yx    += np.einsum("tp,tk->pk", Y[n], xs)
        S_yy    += np.einsum("tp,tq->pq", Y[n], Y[n])
        x0s[n], V0s[n] = xs[0], Vs[0]

    A_new = S_cross @ np.linalg.inv(S_prev)
    C_new = S_yx @ np.linalg.inv(S_all)
    Q_new = (S_next - A_new @ S_cross.T) / (N * (T - 1))
    R_new = (S_yy - C_new @ S_yx.T) / (N * T)
    Q_new = 0.5 * (Q_new + Q_new.T)
    R_new = 0.5 * (R_new + R_new.T)
    mu0_new = x0s.mean(0)
    d = x0s - mu0_new
    V0_new = (V0s.sum(0) + d.T @ d) / N              # includes between-trial spread
    V0_new = 0.5 * (V0_new + V0_new.T)
    return A_new, C_new, Q_new, R_new, mu0_new, V0_new, ll_total


def random_init(rng):
    M = rng.standard_normal((KDIM, KDIM))
    A0 = M * (0.7 / np.max(np.abs(np.linalg.eigvals(M))))
    return (A0, rng.standard_normal((PDIM, KDIM)) / np.sqrt(KDIM),
            0.1 * np.eye(KDIM), 0.2 * np.eye(PDIM),
            np.zeros(KDIM), np.eye(KDIM))


# runs
curves, fits = [], []
init_rng = np.random.default_rng(SEED + 1)
for i in range(N_INITS):
    th = random_init(init_rng)
    lls = []
    for _ in range(N_ITERS):
        *th, ll = em_iteration(Y, *th)
        lls.append(ll / (N * T))                      # per observation
    curves.append(np.array(lls))
    fits.append(th)
    drops = np.sum(np.diff(lls) < -1e-9)
    print(f"init {i}: final {lls[-1]:.4f} per obs, decreases = {drops}")

best = int(np.argmax([c[-1] for c in curves]))
A_hat, C_hat = fits[best][0], fits[best][1]
print(f"\nbest init = {best}")
print("true eigenvalues :", np.round(np.sort(np.linalg.eigvals(A_true).real), 3))
print("fitted eigenvalues:", np.round(np.sort(np.linalg.eigvals(A_hat).real), 3))
print("max |A - A_hat| entrywise:", np.round(np.abs(A_true - A_hat).max(), 3))


# figure 1
fig, ax = plt.subplots(figsize=(0.78 * ps.WIDTH, 2.80))
for i, c in enumerate(curves):
    ax.plot(c, lw=1.4, label=f"init {i + 1}")
ax.set_xlabel("EM iteration")
ax.set_ylabel(r"$\log p(y)$ per observation")
ax.set_xlim(0, 40)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_demo_loglik.png", dpi=200)

# figure 2
fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH, 2.25),
                         gridspec_kw={"width_ratios": [1, 1, 1.25]})
v = np.abs(np.concatenate([A_true.ravel(), A_hat.ravel()])).max()
for ax, Mx, ttl in zip(axes[:2], [A_true, A_hat], [r"$A$", r"$\hat{A}$"]):
    im = ax.imshow(Mx, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(ttl); ax.set_xticks(range(KDIM)); ax.set_yticks(range(KDIM))
fig.colorbar(im, ax=axes[1], fraction=0.046)
ax = axes[2]
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), ls="--", lw=0.8, color="0.6")
et, eh = np.linalg.eigvals(A_true), np.linalg.eigvals(A_hat)
ax.scatter(et.real, et.imag, s=90, facecolors="none", edgecolors="tab:green", label="true")
ax.scatter(eh.real, eh.imag, s=60, marker="x", color="tab:red", label="fitted")
ax.axhline(0, lw=0.6, color="0.7"); ax.axvline(0, lw=0.6, color="0.7")
ax.set_aspect("equal"); ax.set_xlabel("Re"); ax.set_ylabel("Im")
ax.set_title("spectrum"); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_demo_recovery.png", dpi=200)

# figure 3
xs_hat, _, _, _ = e_step(Y[0], *fits[best])
XS = np.concatenate([e_step(Y[n], *fits[best])[0] for n in range(N)])
XT = X.reshape(-1, KDIM)
S, *_ = np.linalg.lstsq(XS, XT, rcond=None)          # gauge alignment x_hat -> x
xs_al = xs_hat @ S
fig, axes = plt.subplots(2, 1, figsize=(0.82 * ps.WIDTH, 3.20), sharex=True)
for d, ax in enumerate(axes):
    ax.plot(X[0, :, d], color="tab:green", lw=1.8, label="true")
    ax.plot(xs_al[:, d], color="tab:red", lw=1.2, ls="--", label="smoothed")
    ax.set_ylabel(f"latent {d + 1}")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, ncol=2)
axes[-1].set_xlabel("time step")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_demo_latents.png", dpi=200)

# figure 4
# Observation space: no alignment needed, since C x is invariant under x -> Sx.
sig_true = X[0] @ C_true.T                       # noiseless signal C x_t
sig_hat  = xs_hat @ fits[best][1].T              # fitted reconstruction C_hat xhat_t
chans = [0, 1]
fig, axes = plt.subplots(len(chans), 1, figsize=(0.82 * ps.WIDTH, 3.20), sharex=True)
for ax, c in zip(axes, chans):
    ax.scatter(np.arange(T), Y[0, :, c], s=14, color="tab:red", zorder=3,
               label="noisy observation")
    ax.plot(sig_true[:, c], color="tab:green", lw=1.8, label="true signal")
    ax.plot(sig_hat[:, c], color="tab:blue", lw=1.2, ls="--",
            label="smoothed reconstruction")
    ax.set_ylabel(f"channel {c + 1}")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.32))
axes[-1].set_xlabel("time step")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_demo_obs.png", dpi=200)

num = np.mean((sig_true - sig_hat) ** 2)
den = np.mean((sig_true - Y[0]) ** 2)
print(f"obs-space MSE: reconstruction {num:.4f} vs raw measurement {den:.4f}"
      f"  ({100 * (1 - num / den):.0f}% reduction)")

r2 = 1 - np.mean((XT - XS @ S) ** 2) / np.mean((XT - XT.mean(0)) ** 2)
print(f"latent R^2 after gauge alignment: {r2:.3f}")
print("figures written.")
