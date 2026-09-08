#!/usr/bin/env python3
"""
Convert an MP-Gadget (BigFile) DM-only snapshot into the map2map-style Lagrangian
displacement (and velocity) cubes used by phase0_octaves.py / octave_flow_toy.py:

    dis.npy : float32 (3, N, N, N)   x - q  in Mpc/h, minimum-image unwrapped
    vel.npy : float32 (3, N, N, N)   velocity in the snapshot's units (recorded in meta.json)

The Lagrangian grid index of a particle is taken from its ID:
    idx = ID - id_offset ;  (i, j, k) = unravel(idx, (N, N, N), order)    order = 'C' or 'F'
    q   = ((i, j, k) + offset) * h ,  h = L / N ,  offset in {0, 0.5}
The right (id_offset, order, offset) combination is the one for which, on the INITIAL
snapshot (z_init), the unwrapped |x - q| is a small fraction of a cell; the script prints
that number and refuses to write if it is not (< 0.25 h by default) unless --force.

Usage
-----
  python hpc/convert_snapshot.py /path/PART_000 out/64/ic_dis.npy --box 100 --ptype 1 --check
  python hpc/convert_snapshot.py /path/PART_020 out/64/dis.npy --vel out/64/vel.npy --box 100

The bigfile package (pip install bigfile) is needed for BigFile snapshots; --npz lets you
feed an already-extracted (pos, id[, vel]) npz instead.  Untested against your files:
run it on the IC snapshot with --check first.
"""
import argparse
import json
import os
import sys

import numpy as np


def load_snapshot(path, ptype, npz=None):
    if npz:
        d = np.load(npz)
        return d["pos"], d["id"], (d["vel"] if "vel" in d else None), float(d["box"]), dict(d.get("meta", {}))
    try:
        import bigfile
    except ImportError:
        sys.exit("pip install bigfile, or pass --npz with pos/id/vel/box arrays")
    f = bigfile.File(path)
    head = f["Header"].attrs
    box = float(np.asarray(head["BoxSize"]).ravel()[0])
    meta = {k: (np.asarray(v).tolist() if not isinstance(v, (bytes, str)) else str(v)) for k, v in head.items()
            if k in ("BoxSize", "Time", "TotNumPart", "MassTable", "HubbleParam", "Omega0", "OmegaLambda",
                     "UsePeculiarVelocity", "TimeIC", "Redshift")}
    grp = f[f"{ptype}/"]
    pos = grp["Position"][:]
    ids = grp["ID"][:]
    vel = grp["Velocity"][:] if "Velocity" in grp.blocks else None
    return pos, ids, vel, box, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("out_dis")
    ap.add_argument("--vel", type=str, default=None, help="also write the velocity cube here")
    ap.add_argument("--npz", type=str, default=None, help="use an npz with pos, id[, vel], box instead of BigFile")
    ap.add_argument("--ptype", type=int, default=1)
    ap.add_argument("--box", type=float, default=None, help="box size in Mpc/h (default: header BoxSize / unit)")
    ap.add_argument("--pos-unit", type=float, default=1e-3, help="multiply positions by this to get Mpc/h (MP-Gadget kpc/h -> 1e-3)")
    ap.add_argument("--id-offset", type=int, default=0, help="ID of the (0,0,0) particle")
    ap.add_argument("--id-order", type=str, default="C", choices=["C", "F"], help="unravel order of ID -> (i,j,k)")
    ap.add_argument("--offset", type=float, default=0.5, help="grid offset o: q = (idx + o) h")
    ap.add_argument("--check", action="store_true", help="only print the |x - q| statistic, write nothing")
    ap.add_argument("--max-frac", type=float, default=0.25, help="refuse to write if rms|x-q|/h exceeds this (IC check)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pos, ids, vel, box_hdr, meta = load_snapshot(args.snapshot, args.ptype, args.npz)
    pos = np.asarray(pos, dtype=np.float64) * args.pos_unit
    L = args.box if args.box is not None else box_hdr * args.pos_unit
    n = pos.shape[0]
    N = int(round(n ** (1 / 3)))
    assert N**3 == n, f"{n} particles is not a cube"
    h = L / N
    idx = np.asarray(ids, dtype=np.int64) - args.id_offset
    assert idx.min() == 0 and idx.max() == n - 1, f"IDs after offset span [{idx.min()}, {idx.max()}], expected [0, {n-1}]"
    ijk = np.stack(np.unravel_index(idx, (N, N, N), order=args.id_order), axis=1)
    q = (ijk + args.offset) * h
    dis = pos - q
    dis -= L * np.round(dis / L)                       # minimum image
    frac = float(np.sqrt((dis**2).sum(1).mean()) / h)
    print(f"N={N}  L={L:.3f} Mpc/h  h={h:.4f}  rms|x-q|/h = {frac:.4f}   (IC snapshot: should be << 1; "
          f"try other --id-offset/--id-order/--offset if not)")
    if args.check:
        return
    if frac > args.max_frac and not args.force:
        sys.exit("rms|x-q|/h too large for an IC snapshot; for evolved snapshots pass --force")
    # scatter into (3, N, N, N) by grid index
    order = np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]), (N, N, N))
    cube = np.zeros((3, N, N, N), dtype=np.float32)
    cube.reshape(3, -1)[:, order] = dis.T.astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_dis)), exist_ok=True)
    np.save(args.out_dis, cube)
    if args.vel is not None and vel is not None:
        vcube = np.zeros((3, N, N, N), dtype=np.float32)
        vcube.reshape(3, -1)[:, order] = np.asarray(vel, dtype=np.float32).T
        np.save(args.vel, vcube)
    meta.update(dict(N=N, L=L, h=h, id_offset=args.id_offset, id_order=args.id_order, grid_offset=args.offset,
                     rms_dis_over_h=frac, source=os.path.abspath(args.snapshot)))
    with open(os.path.splitext(args.out_dis)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", args.out_dis, "and" if args.vel else "", args.vel or "")


if __name__ == "__main__":
    main()
