"""
Pullback metric schematic.

Reproduces the three-panel figure:
    (1) task domain  M = [t_min, t_max] x S^1   (straight cylinder)
    (2) latent manifold                          x_t = A^t x_0
    (3) observation manifold                     y_t = C A^t x_0

plus the induced-metric ellipse field over the domain and the two 2x2
pullback metrics G(p1), G(p2).

Requires: numpy, matplotlib.  Run:  python pullback_schematic.py
Outputs:  figures/fig_pullback_schematic.pdf and .png
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap

# parameters
CONTRACTION = 0.62      # a   in  r(t) = exp(-a t)
ROTATION    = 1.75      # w   in  phi(t) = theta + w t
HEIGHT      = 1.45      # h   vertical extent of the cylinder
T0, TH0     = 0.22, -1.75      # marked point p1
T1, TH1     = 0.90, -3.05      # marked point p2

C = np.array([[1.00,  0.42, 0.02],
              [0.26,  0.86, 0.46],
              [0.10, -0.36, 1.14]])   # readout / observation map

STEEL, DEEP, ORANGE = "#3f7fae", "#14456b", "#c25c14"
SURF = LinearSegmentedColormap.from_list("surf", ["#eaf3fa", "#a9cde6"])
FIELD = LinearSegmentedColormap.from_list("field", ["#f3f8fc", "#bcd8ec"])

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
})


# manifolds
def cylinder(t, th):
    """Task domain: the undeformed cylinder [t_min,t_max] x S^1."""
    return np.stack([np.cos(th), np.sin(th), HEIGHT * t], axis=-1)


def latent(t, th, a=CONTRACTION, w=ROTATION):
    """Latent trajectory manifold, x_t = A^t x_0 (contract + rotate)."""
    r, p = np.exp(-a * t), th + w * t
    return np.stack([r * np.cos(p), r * np.sin(p), HEIGHT * t], axis=-1)


def observed(t, th):
    """Observation manifold, y_t = C A^t x_0."""
    return latent(t, th) @ C.T


# differential geometry
def metric(f, t, th, eps=8e-3):
    """Pullback metric G = J^T J of the embedding f at (t, theta)."""
    ft = (f(t + eps, th) - f(t - eps, th)) / (2 * eps)
    fh = (f(t, th + eps) - f(t, th - eps)) / (2 * eps)
    return np.array([[ft @ ft, ft @ fh],
                     [ft @ fh, fh @ fh]])


def normaliser(f):
    """G0^{-1/2}, used to express every G relative to the metric at reach onset."""
    g0 = metric(f, 0.0, 0.0)
    lam, V = np.linalg.eigh(g0)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(lam, 1e-12))) @ V.T


# drawing
def draw_surface(ax, f, title, marks=(), tangents_at=None,
                 nt=9, nh=26, elev=22, azim=-58):
    """Shaded quad mesh + coordinate grid + tangent frame, on a 3-D axis."""
    T, TH = np.meshgrid(np.linspace(0, 1, nt + 1),
                        np.linspace(0, 2 * np.pi, nh + 1), indexing="ij")
    P = f(T, TH)

    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2],
                    rstride=1, cstride=1, linewidth=0,
                    facecolors=SURF((P[..., 1] - P[..., 1].min())
                                    / np.ptp(P[..., 1])),
                    shade=False, alpha=0.95, zorder=1)

    th = np.linspace(0, 2 * np.pi, 200)
    for k, t in enumerate([0, 0.25, 0.5, 0.75, 1.0]):          # rings
        R = f(np.full_like(th, t), th)
        ax.plot(*R.T, color=DEEP if k == 0 else STEEL,
                lw=2.0 if k == 0 else 0.9, alpha=1 if k == 0 else 0.85)
    tt = np.linspace(0, 1, 60)
    for a_ in np.linspace(0, 2 * np.pi, 12, endpoint=False):   # flow lines
        F = f(tt, np.full_like(tt, a_))
        ax.plot(*F.T, color=STEEL, lw=0.8, alpha=0.6)

    if tangents_at is not None:
        t0, th0 = tangents_at
        s = np.linspace(0, 0.30, 30)                            # d/dt  curve
        A = f(t0 + s, np.full_like(s, th0))
        ax.plot(*A.T, color=ORANGE, lw=2.0)
        ax.scatter(*A[-1], color=ORANGE, s=18)
        s = np.linspace(0, 1.05, 40)                            # d/dtheta curve
        B = f(np.full_like(s, t0), th0 + s)
        ax.plot(*B.T, color=ORANGE, lw=2.0)
        ax.scatter(*B[-1], color=ORANGE, s=18)

    for (mt, mth), label in marks:
        p = f(np.array(mt), np.array(mth))
        ax.scatter(*p, color=DEEP, s=34, depthshade=False, zorder=6)
        ax.text(*(p * 1.16), label, color=DEEP, fontsize=11,
                style="italic", ha="center")

    ax.set_title(title, fontsize=12, pad=2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1.15))


def draw_field(ax, f):
    """Ellipse field over the (t, theta) domain, shaded by sqrt(det G)."""
    ts = np.array([0, 0.25, 0.5, 0.75, 1.0])
    ths = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    S = normaliser(f)

    E, area = [], []
    for i, th in enumerate(ths):
        for j, t in enumerate(ts):
            g = S @ metric(f, t, th) @ S
            lam, V = np.linalg.eigh(g)
            ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
            E.append((j, i, np.sqrt(lam[-1]), np.sqrt(max(lam[0], 1e-9)), ang, t, th))
            area.append(np.sqrt(max(np.linalg.det(g), 0)))
    area = np.array(area)
    k = 0.42 / max(e[2] for e in E)
    u = (area - area.min()) / (np.ptp(area) or 1)

    marks = [(T0, TH0), (T1, TH1)]
    for (j, i, s1, s2, ang, t, th), ui in zip(E, u):
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                   facecolor=FIELD(ui), edgecolor="none"))
        near = any(abs(t - mt) < 0.15 and
                   min((th - mth) % (2 * np.pi),
                       (mth - th) % (2 * np.pi)) < 0.9 for mt, mth in marks)
        ax.add_patch(Ellipse((j, i), 2 * s1 * k, 2 * s2 * k, angle=ang,
                             fill=False, lw=1.8 if near else 1.0,
                             edgecolor=ORANGE if near else "#2f6690"))

    ax.set_xlim(-.5, len(ts) - .5)
    ax.set_ylim(len(ths) - .5, -.5)
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([f"{t:g}" for t in ts], fontsize=9)
    ax.set_yticks(range(len(ths)))
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"], fontsize=9)
    ax.set_xlabel(r"$t$", fontsize=11)
    ax.set_ylabel(r"$\theta$", fontsize=11)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(r"Induced metric $\mathbf{G}(t,\theta)$ on $\mathcal{M}$",
                 fontsize=12, pad=8)


def draw_metric(ax, G, title):
    """A single 2x2 pullback metric as a labelled heatmap."""
    ax.imshow(G, cmap=FIELD)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, color="#123")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([r"$t$", r"$\theta$"], fontsize=10)
    ax.set_yticklabels([r"$t$", r"$\theta$"], fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=11, pad=6)


# assemble
def main():
    fig = plt.figure(figsize=(14.2, 7.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 1.15, 1.15, 1.0],
                          height_ratios=[1, 1], wspace=0.05, hspace=0.35)

    axc = fig.add_subplot(gs[0, 0], projection="3d")
    axl = fig.add_subplot(gs[0, 1], projection="3d")
    axo = fig.add_subplot(gs[0, 2], projection="3d")

    draw_surface(axc, cylinder,
                 r"$\mathcal{M}=[t_{\min},t_{\max}]\times S^{1}$",
                 tangents_at=(T0, TH0))
    draw_surface(axl, latent, r"latent:  $\bar{x}_t = A^{t}\bar{x}_0$",
                 tangents_at=(T0, TH0))
    draw_surface(axo, observed, r"observed:  $\bar{y}_t = CA^{t}\bar{x}_0$",
                 tangents_at=(T0, TH0),
                 marks=[((T0, TH0), r"$p_1$"), ((T1, TH1), r"$p_2$")])

    draw_field(fig.add_subplot(gs[1, 1]), observed)

    S = normaliser(observed)
    draw_metric(fig.add_subplot(gs[0, 3]), S @ metric(observed, T0, TH0) @ S,
                r"$\mathbf{G}(p_1)$   (early)")
    draw_metric(fig.add_subplot(gs[1, 3]), S @ metric(observed, T1, TH1) @ S,
                r"$\mathbf{G}(p_2)$   (later)")

    fig.savefig("figures/fig_pullback_schematic.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_pullback_schematic.png", dpi=300, bbox_inches="tight")
    print("wrote figures/fig_pullback_schematic.pdf / .png")


if __name__ == "__main__":
    main()
