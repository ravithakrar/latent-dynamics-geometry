"""
fig_init_floor: does the initialisation change the fitted metric?

EM log-likelihood per trial, time bin and unit against iteration, session
20160914, identity initialisation against the warm start, M1 and PMd. The
likelihood is the log-density of an M-dimensional Gaussian, so dividing by the
unit count as well as the bin count puts the two areas on one scale.

Fitted on live units only: units whose variance falls below the region's
observation-noise floor are dropped at load (tangent_data.UNIT_VAR_FLOOR).
Before that rule, PMd's identity-init trace fell by 15,927 nats on its first
step, because a constant unit makes the likelihood at the PCA initialisation
unbounded above. Every trace here is monotone from iteration 1.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(["science", "no-latex", "bright"])
plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight"})

HERO = "20160914"
d = np.load("msa_fields.npz", allow_pickle=True)

fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
for ax, reg, tag in zip(axs, ("M1", "PMd"), ("A", "B")):
    N, M, D = d[f"{HERO}|{reg}|full|meta"]
    for v, lab in (("full", "identity"), ("alt", "warm start")):
        ll = d[f"{HERO}|{reg}|{v}|lls"] / (N * 25 * M)
        ax.plot(np.arange(1, len(ll) + 1), ll, lw=1.3, label=lab)
    ax.set_title(f"{tag}   {HERO}, {reg}  ({int(M)} units, D = {int(D)})", loc="left")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log-likelihood per bin per unit")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.22), fontsize=7.5)
fig.tight_layout()
import sys
STEM = sys.argv[1] if len(sys.argv) > 1 else "figures/fig42_init_floor"
fig.savefig(f"{STEM}.pdf"); fig.savefig(f"{STEM}.png")
print("ok")
