# PGO benchmark on KITTI-360

End-to-end loop-closure evaluation of the PGO native module against the
KITTI-360 dataset. Reports precision / recall / F1 on the LCDNet-style
groundtruth (≥50-frame gap, ≤4m radius — see ``loop_groundtruth.py``).

## Download (manual, login required)

The dataset is gated behind a free academic registration at
[cvlibs.net/datasets/kitti-360](https://www.cvlibs.net/datasets/kitti-360/).
Once logged in, grab the following from the download page (the official
"smaller" benchmark target is **Test SLAM 3D** at ~12 GB):

| Package                            | Size  | Required for |
|------------------------------------|-------|--------------|
| Test SLAM (3D)                     | 12 GB | Velodyne scans for SLAM evaluation |
| Vehicle Poses                       | 9 MB  | groundtruth poses |
| Calibrations                        | 3 KB  | lidar→camera extrinsics |

Extract under a single root (any path is fine — the benchmark just needs
the standard KITTI-360 sub-tree):

```
<root>/
    data_3d_raw/2013_05_28_drive_<seq>_sync/velodyne_points/
        data/<frame_id>.bin
        timestamps.txt
    data_poses/2013_05_28_drive_<seq>_sync/
        poses.txt
    calibration/
        calib_cam_to_velo.txt
```

The full **Raw Velodyne Scans** package (119 GB, all 9 sequences) is the
other official option; only needed if you want to evaluate on sequences
that aren't in the Test SLAM split.

## Run

```bash
source .venv/bin/activate

# native binary must be built (see ../cpp/)
ls ../cpp/result/bin/pgo  # should exist; otherwise: cd ../cpp && nix build .#default

uv run python -m dimos.navigation.nav_stack.modules.pgo.benchmark.run_kitti360_benchmark \
    --kitti360-root ~/datasets/kitti360 \
    --sequence 9 \
    --max-scans 4000 \
    --output-json /tmp/pgo_kitti360_seq09.json
```

Sequences with substantial loop closures (from the LCDNet paper):
- **seq 02**: ~2,452 loops, mostly forward
- **seq 09**: ~4,670 loops, ~87% reverse — the canonical reverse-loop
  stress test where vanilla position-search degrades and Scan Context
  shines

## Output

The runner prints a per-sequence summary and (optionally) writes a JSON
report. Fields:

```json
{
  "sequence_id": 9,
  "scans_played": 4000,
  "groundtruth_queries_with_loop": 320,
  "groundtruth_total_loop_pairs": 4670,
  "detected_loop_edges": 215,
  "loop_closure_events": 215,
  "metrics": {
    "true_positive": 198,
    "false_positive": 17,
    "false_negative": 122,
    "precision": 0.921,
    "recall": 0.619,
    "f1": 0.740
  },
  "wallclock_seconds": 412.7
}
```

## Reference numbers — state of the art on KITTI-360

Published AP (Average Precision from PR curves) on the two loop-rich
KITTI-360 sequences:

| Method               | seq 02 AP | seq 09 AP | Notes                       |
|---------------------|-----------|-----------|-----------------------------|
| Scan Context (orig) | 0.65–0.78 | 0.62–0.72 | drops on heavy-reverse seq 09 |
| LiDAR-Iris          | 0.66–0.80 | 0.65–0.73 | handcrafted, similar tier  |
| OverlapNet          | 0.55–0.70 | 0.50–0.65 | first learned-features baseline |
| **LCDNet (SOTA)**   | **0.93**  | **0.91**  | learned + reverse-aware    |
| Intensity SC        | ~+24% F1 vs vanilla SC | similar | reverse boost via intensity |

Sources: [LCDNet paper](https://arxiv.org/abs/2103.05056), [Intensity Scan
Context paper](https://arxiv.org/abs/2003.05656).

**Our PGO impl** is a faithful but minimal port of Scan Context (Kim & Kim
2018) — no intensity channel, no SC++ extensions, no learned modules.
Expected ballpark: 0.60–0.75 AP on forward sequences, with the
column-shift trick keeping reverse loops from collapsing entirely. Real
numbers TBD pending the dataset download.

## Implementation notes

- The benchmark spawns the native PGO binary as a subprocess with private
  LCM topic names (so concurrent test runs don't collide). The Python
  side plays (registered_scan, odometry) at a configurable rate.
- Loop pair extraction reads ``pgo_graph_edges`` and filters segments
  with ``orientation.w ≈ 0.4`` (loop closure traversability), then maps
  endpoint world-positions to keyframe indices by nearest-neighbour
  matching against the cached lidar trajectory.
- Recall is "fraction of queries that have *any* valid groundtruth loop
  AND at least one correct detection" — not "fraction of groundtruth
  pairs correctly detected." This matches the LCDNet Protocol 1 spec.
- AP (sweep over match thresholds) is not yet computed by this script —
  for now we report point metrics at the configured threshold. Adding
  AP would require running PGO N times with different
  ``--sc_match_threshold`` values, or modifying PGO to publish raw
  match scores per candidate.
