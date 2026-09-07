"""migrate_cache_gpush.py — add g_push to an existing msa_fields.npz without refitting.

`msa_cache.py` now stores both metric constructions side by side. Existing caches predate
that, but they store xbar, A, C and theta, which is everything g needs — so the missing
field can be filled in directly. No EM.

Verifies that the recomputed DIRECT metric matches the stored `g` before writing, which
confirms the cache's declared config really is what produced it.
"""
import json, sys
import numpy as np
import msa_cache as mc, pullback_metric as pb

path = sys.argv[1] if len(sys.argv) > 1 else "msa_fields.npz"
meta = mc.meta(path)
lam, method = meta.get("lam", 0.0), meta.get("method", "cubic")
print(f"{path}: method={method} lam={lam} push={meta.get('push')}")

z = dict(np.load(path))
F = mc.load(path)
worst = 0.0
added = 0
for (s, r, v), d in F.items():
    gd, _, _ = pb.metric(d["xbar"], d["A"], C=d["C"], theta=d["theta"],
                         method=method, lam=lam)
    worst = max(worst, float(np.abs(gd - d["g"]).max()))
    gp, _, _ = pb.metric_pushforward(d["xbar"], d["A"], C=d["C"], theta=d["theta"],
                                     method=method, lam=lam)
    z[f"{s}|{r}|{v}|g_push"] = np.asarray(gp, float)
    added += 1

print(f"recomputed direct g matches stored g to {worst:.2e}  ({added} entries)")
if worst > 1e-9:
    raise SystemExit("MISMATCH — the cache was not produced by its declared config; not writing")

np.savez_compressed(path, **z)
meta["stores"] = ["g (direct)", "g_push (pushforward)", "g_lat (latent, direct tangent)"]
meta["g_alias"] = "push" if meta.get("push") else "direct"
json.dump(meta, open(path.replace(".npz", "_meta.json"), "w"), indent=1)
print(f"WROTE {path} with g_push added")
