"""
fig_smoothing_spectrum: dimensionality of the condition-averaged activity as the
Gaussian filter width increases. M1, session 20160914.

The sweep is:
SVD of the condition-averaged activity (8 directions x T bins, one column per
neuron), per-neuron re-standardisation at each width, effective width
sqrt(40^2 + added^2) because the cached tensor is already smoothed at 40 ms.
Widths below 40 ms would need the raw counts re-binned.

Left  : sigma_i / sigma_1, index 1..40, one line per width.
Right : participation ratio PR = (sum sigma^2)^2 / sum sigma^4 against width.

Project figure convention: matplotlib owns the maths, PDF out for Inkscape.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotstyle as ps          # shared thesis figure style (SciencePlots base)
ps.apply()


TENSOR    = "tensor_20160914.npz"
REGION    = "M1"
SIG_CACHE = 40.0
BIN       = 20.0
WIDTHS    = np.array([40., 50., 60., 80., 100., 140., 200.])
NSV       = 40

def gauss_smooth(y, sigma_ms):
    if sigma_ms <= 0:
        return y.copy()
    s = sigma_ms / BIN
    r = max(1, int(np.ceil(4 * s)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    pad = np.pad(y, ((0, 0), (r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(y)
    for i, w in enumerate(k):
        out += w * pad[:, i:i + y.shape[1], :]
    return out

def spectrum(y, target):
    """Re-standardise each neuron at this width, average within reach direction,
    centre, then take singular values. Reproduces the notebook sweep."""
    X0 = y.reshape(-1, y.shape[2])
    sd = X0.std(0, keepdims=True)
    y  = y / np.where(sd > 1e-12, sd, 1.0)
    dirs = np.unique(target)
    xbar = np.stack([y[target == dd].mean(0) for dd in dirs])   # (K, T, M)
    X = xbar.reshape(-1, xbar.shape[2])                         # (K*T, M)
    X = X - X.mean(0, keepdims=True)
    return np.linalg.svd(X, compute_uv=False)


d  = np.load(TENSOR)
y0 = d[f"y_{REGION}"]; tgt = d[f"target_{REGION}"]
sv, pr = {}, {}
for w in WIDTHS:
    s = spectrum(gauss_smooth(y0, np.sqrt(max(w**2 - SIG_CACHE**2, 0.0))), tgt)
    sv[w] = s / s[0]
    pr[w] = (s**2).sum()**2 / (s**4).sum()

fig, axs = plt.subplots(1, 2, figsize=(ps.WIDTH, 2.65))
cols = plt.cm.viridis(np.linspace(0.0, 0.92, len(WIDTHS)))

ax = axs[0]
for c, w in zip(cols, WIDTHS):
    ax.plot(np.arange(1, NSV + 1), sv[w][:NSV], color=c, lw=1.3, label=f"{int(w)} ms")
ax.set_yscale("log")
ax.set_xlabel("singular value index $i$")
ax.set_ylabel(r"$\sigma_i/\sigma_1$")
ax.set_title("A", loc="left")
ax.legend(frameon=False, ncol=2, fontsize=8, handlelength=1.4, columnspacing=1.0)

ax = axs[1]
ax.plot(WIDTHS, [pr[w] for w in WIDTHS], marker="o", ms=4.5, lw=1.4, color="0.15")
ax.set_xlabel("effective Gaussian width (ms)")
ax.set_ylabel("participation ratio")
ax.set_title("B", loc="left")

fig.tight_layout()
fig.savefig("figures/fig_smoothing_spectrum.pdf", bbox_inches="tight")
fig.savefig("figures/fig_smoothing_spectrum.png", dpi=200, bbox_inches="tight")
print("PR:", ", ".join(f"{int(w)}ms={pr[w]:.2f}" for w in WIDTHS))
