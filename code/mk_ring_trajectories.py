"""
fig_ring_trajectories: the initial-state ring and the latent trajectories it
generates, session 20160914, M1 at D = 8 (top) and PMd at D = 12 (bottom).

Built from the cached fits in msa_fields.npz (condition means xbar and the
fitted A, C at D = 8/12, pcubic, lambda = 0.3, pushforward). No refitting.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
import plotstyle as ps          # shared thesis figure style (SciencePlots base)
ps.apply()

from mpl_toolkits.mplot3d import Axes3D  # noqa

SESS, ONSET = "20160914", 5
d = np.load("msa_fields.npz", allow_pickle=True)

fig = plt.figure(figsize=(ps.WIDTH, 5.6))
TAGS = [["A", "B"], ["C", "D"]]

for row, REG in enumerate(("M1", "PMd")):
    k = lambda f: d[f"{SESS}|{REG}|full|{f}"]
    xbar, th, C = k("xbar"), k("theta"), k("C")     # (K,T,D), (K,), (M,D)
    K, T, D = xbar.shape

    # principal axes of the condition means, so both panels share one basis
    X = xbar.reshape(-1, D) - xbar.reshape(-1, D).mean(0, keepdims=True)
    U = np.linalg.svd(X, full_matrices=False)[2]
    proj = lambda Z: (Z.reshape(-1, D) - xbar.reshape(-1, D).mean(0)) @ U.T
    cols = plt.cm.twilight((th % (2*np.pi)) / (2*np.pi))

    ax = fig.add_subplot(2, 2, 2*row + 1)
    R0 = xbar[:, ONSET, :] - xbar[:, ONSET, :].mean(0, keepdims=True)
    Ur = np.linalg.svd(R0, full_matrices=False)[2]
    p0 = R0 @ Ur.T
    vr = (R0 @ Ur.T).var(0); vr = 100 * vr[:2].sum() / vr.sum()
    order = np.argsort(th % (2*np.pi))
    ring = np.vstack([p0[order], p0[order][:1]])
    ax.plot(ring[:, 0], ring[:, 1], "-", color="0.6", lw=1.1, zorder=1)
    for i in range(K):
        ax.scatter(p0[i, 0], p0[i, 1], color=cols[i], s=40, zorder=3,
                   edgecolor="k", linewidth=0.4)
    # the reach direction each state belongs to, printed just outside the ring
    for i in range(K):
        r = np.hypot(*p0[i, :2])
        u = p0[i, :2] / r if r > 0 else np.zeros(2)
        deg = np.degrees(th[i] % (2 * np.pi))
        ax.text(*(p0[i, :2] + 0.26 * r * u), rf"${deg:.0f}^\circ$", color=ps.INK,
                ha="center", va="center", fontsize=6.4, zorder=4)
    lim = 1.55 * np.abs(p0[:, :2]).max()
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title(f"{TAGS[row][0]}   {REG}, $D = {D}$  ({vr:.0f}% of the ring variance)",
                 loc="left")
    ax.set_aspect("equal"); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(2, 2, 2*row + 2, projection="3d")
    for i in range(K):
        tr = proj(xbar[i]).reshape(T, -1)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=cols[i], lw=1.2)
        ax.scatter(*tr[ONSET, :3], color=cols[i], s=22, edgecolor="k", linewidth=0.35)
        ax.scatter(*tr[-1, :3], color=cols[i], s=14, marker="s")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_zlabel("PC 3")
    ax.set_title(f"{TAGS[row][1]}   {REG}, $D = {D}$", loc="left")
    ax.view_init(elev=22, azim=-58)

fig.tight_layout(pad=1.4, h_pad=2.4)
STEM = _sys.argv[1] if len(_sys.argv) > 1 else "figures/fig42_ring_trajectories"
fig.savefig(f"{STEM}.pdf", bbox_inches="tight")
fig.savefig(f"{STEM}.png", dpi=200, bbox_inches="tight")
print(f"{SESS} {REG}: K={K} T={T} D={D}, trials/units from meta =", k("meta"))
