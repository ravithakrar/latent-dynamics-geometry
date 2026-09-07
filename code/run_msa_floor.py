"""The missing control for run_msa.py: a within-session noise floor.

Between-session d_MSA mixes two things -- genuine day-to-day change in the geometry, and the
fact that any two metric estimates differ because they come from different trials and different
EM fits. To separate them, split ONE session's trials into two halves, fit each independently,
and compute d_MSA between the halves. That is the same estimator applied to data that is the
same by construction, so whatever it returns is the floor.

Between-session distances are only interpretable relative to this number.
"""
import glob, json, os, time
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import fitting as mr, lds_em_highD as em, pullback_metric as pb
from run_msa import msa, scale_stat, D_REGION, SESSIONS, DATA, NPZ_FALLBACK, load_npz, NIT

def field_from(y, targ, D, rf):
    params, _, _ = mr.fit_identity(y, D, rf, NIT)
    A = np.asarray(params[0]); C = np.asarray(params[1])
    xh = np.asarray(em.e_step_batch(y, *params)[0])
    xbar, theta = pb.condition_means(xh, targ)
    return np.asarray(pb.metric(xbar, A, C=C, theta=theta)[0])

out = []
for sess in SESSIONS:
    hits = glob.glob(f"{DATA}/sub-C_ses-CO-{sess}_behavior+ecephys.nwb")
    if hits: F = mr.preprocess_session(hits[0])
    elif sess in NPZ_FALLBACK and os.path.exists(NPZ_FALLBACK[sess]): F = load_npz(NPZ_FALLBACK[sess])
    else: continue
    for region in ("M1", "PMd"):
        y = np.asarray(F[region]["y"], float); targ = np.asarray(F[region]["targ"], float)
        D, rf = D_REGION[region], mr.RF[region]
        rng = np.random.default_rng(0)
        # stratified half-split: split WITHIN each direction so both halves see all 8 angles
        h1, h2 = [], []
        for d in np.unique(targ):
            idx = rng.permutation(np.where(targ == d)[0])
            h1 += list(idx[:len(idx)//2]); h2 += list(idx[len(idx)//2:])
        h1, h2 = np.array(h1), np.array(h2)
        t0 = time.time()
        gA = field_from(y[h1], targ[h1], D, rf)
        gB = field_from(y[h2], targ[h2], D, rf)
        m = msa(gA, gB); m.update(session=sess, region=region, n1=len(h1), n2=len(h2),
                                  log_scale_ratio=scale_stat(gA, gB))
        out.append(m)
        print(f"  {sess} {region}: within-session floor d_MSA = {m['uniform']:.3f} "
              f"({len(h1)}/{len(h2)} trials) [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open("msa_floor.json", "w"), indent=1)

for region in ("M1", "PMd"):
    v = [m["uniform"] for m in out if m["region"] == region]
    print(f"[{region}] FLOOR mean {np.mean(v):.3f}  range {min(v):.3f}-{max(v):.3f}", flush=True)
print("DONE", flush=True)
