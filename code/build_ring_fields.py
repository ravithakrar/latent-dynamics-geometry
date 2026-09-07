"""build_ring_fields.py — the direction ring on a fine angular grid.

Builds the ring fields the circumference and MDS figures are drawn from, at the convention
fixed in metric_convention.py: method = "pcubic", lam = 0.3, pushforward.

No EM fits are needed: msa_fields.npz stores xbar, theta, A and C per session and variant, so
this is a few seconds of spline arithmetic.

Writes ring_fields_push.npz:

    thfine_<sess>_<reg>   (n,)      the fine angular grid, one full turn
    gthth_<sess>_<reg>    (T,n)     g_thth on that grid, i.e. radius**2
    L_<sess>_<reg>        (T,)      geodesic circumference, closed integral of sqrt(g_thth)
    geomat_<sess>_<reg>   (T,K,K)   pairwise geodesic distance between the K measured directions
    theta_<sess>_<reg>    (K,)      the measured reach angles

Usage:  python3 build_ring_fields.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import geodesic_ring as gr
import metric_convention as mc

SUBC = ["20160909", "20160914", "20160919", "20160921", "20160929", "20161013"]
SUBM = ["20140203", "20140217", "20140218", "20140303", "20140304", "20140306", "20140307"]
REGIONS = ["M1", "PMd"]
N_FINE = 240

ap = argparse.ArgumentParser()
ap.add_argument("--animal", choices=["subC", "subM"], default="subC")
a = ap.parse_args()
if a.animal == "subC":
    FIELDS, SESS, FOCUS, OUT = "msa_fields.npz", SUBC, "20160914", "ring_fields_push"
else:
    FIELDS, SESS, FOCUS, OUT = "subm_fields.npz", SUBM, "20140303", "ring_fields_push_subm"

d = np.load(FIELDS, allow_pickle=True)
out, summary = {}, {}

for sess in SESS:
    for reg in REGIONS:
        k = f"{sess}|{reg}|full"
        thf, radius = mc.ring(d[k + "|xbar"], d[k + "|theta"], d[k + "|A"], d[k + "|C"],
                              n_fine=N_FINE)                      # radius (n,T)
        theta = np.asarray(d[k + "|theta"], float)
        T = radius.shape[1]
        L = np.array([gr.arc_length(radius[:, t], thf)[0] for t in range(T)])
        geomat = np.stack([gr.arc_matrix(radius[:, t], thf, theta)[0] for t in range(T)])

        out[f"thfine_{sess}_{reg}"] = thf
        out[f"gthth_{sess}_{reg}"] = (radius ** 2).T                # (T,n)
        out[f"L_{sess}_{reg}"] = L
        out[f"geomat_{sess}_{reg}"] = geomat
        out[f"theta_{sess}_{reg}"] = theta
        summary[f"{sess}|{reg}"] = dict(onset=float(L[5]), end=float(L[-1]),
                                        peak=float(L.max()), peak_bin=int(L.argmax()),
                                        ratio=float(L[-1] / L[5]))
        print(f"{sess} {reg:>3}: L onset {L[5]:7.2f}  peak {L.max():7.2f} at bin "
              f"{int(L.argmax()):2d}  end {L[-1]:7.2f}  ratio {L[-1] / L[5]:.3f}", flush=True)

np.savez_compressed(OUT + ".npz", **out)
json.dump(dict(convention=dict(method=mc.METHOD, lam=mc.LAM, push=mc.PUSH),
               sessions=SESS, regions=REGIONS, focus=FOCUS, n_fine=N_FINE,
               onset_bin=5, bin_ms=20.0, L=summary),
          open(OUT + ".json", "w"), indent=1)

for reg in REGIONS:
    m = np.mean([out[f"L_{s}_{reg}"] for s in SESS], axis=0)
    print(f"MEAN {reg:>3}: onset {m[5]:.2f}  peak {m.max():.2f} at bin {int(m.argmax())}  "
          f"end {m[-1]:.2f}  ratio {m[-1] / m[5]:.3f}")
print("wrote", OUT)
