#!/usr/bin/env python3
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Callable
import logging

import numpy as np
from natnet import NatNetClient, DataDescriptions, DataFrame

# ========= USER CONFIG =========
SERVER_IP = "127.0.0.1"
LOCAL_IP  = "127.0.0.1"

RB_GRID = "Grid"
RB_SUBJ = "Participant"

GRID_LOCAL_POINT = np.array([0.0, 0.0, 0.0])
SUBJ_BONE_TIP    = np.array([0.0, 0.0, 0.0])

# Tolerances drive the "OK / Adjust" status in the UI
TRANS_TOL_MM = 5.0
ROT_TOL_DEG  = 5.0

PRINT_INTERVAL_S = 0.05
LOG_INTERVAL_S   = 0.25

CSV_PATH  = Path("placement_log.csv")
POSE_FILE = Path("target_pose.json")
SNAP_CSV  = Path("snapshots.csv")
SNAP_JSON_DIR = Path("snapshots")
SNAP_JSON_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= END CONFIG =========

# ---------- math helpers ----------
# def quat_to_rot(qx, qy, qz, qw):
#     x, y, z, w = qx, qy, qz, qw
#     return np.array([
#         [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
#         [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
#         [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
#     ], dtype=float)

def rotmat_from_quat(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ], dtype=float)

def relative_RT(R_s, p_s, R_g, p_g):
    R_rel = R_s.T @ R_g
    t_rel = R_s.T @ (p_g - p_s)
    return R_rel, t_rel

def relative_RT_with_points(R_s, p_s, p_s_local, R_g, p_g, p_g_local):
    # world points anchored at chosen local points
    ps_star = p_s + R_s @ p_s_local.reshape(3)
    pg_star = p_g + R_g @ p_g_local.reshape(3)
    R_rel = R_s.T @ R_g
    t_rel = R_s.T @ (pg_star - ps_star)
    return R_rel, t_rel

def rotation_geodesic_deg(R_a, R_b):
    dR = R_a.T @ R_b
    tr = np.clip((np.trace(dR) - 1) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(tr))

# ---------- NatNet state ----------
id2name: Dict[int, str] = {}
name2id: Dict[str, int] = {}
latest_frame: Optional[DataFrame] = None

def on_desc(desc: DataDescriptions):
    global id2name, name2id
    id2name.clear(); name2id.clear()
    candidates = []
    for attr in ("rigid_bodies", "rigid_body_descriptions", "rbds", "rb_desc"):
        if hasattr(desc, attr):
            val = getattr(desc, attr)
            if isinstance(val, (list, tuple)) and len(val) > 0:
                candidates = val
                break
    for rbd in candidates:
        rid = getattr(rbd, "id_num", getattr(rbd, "rigid_body_id", None))
        rname = getattr(rbd, "name", None)
        if rid is not None and rname:
            id2name[int(rid)] = str(rname)
            name2id[str(rname)] = int(rid)

def on_frame(df: DataFrame):
    global latest_frame
    latest_frame = df

# ---------- parsing helpers ----------
def _markerset_centroids(df):
    centroids = {}
    ms_list = None
    for attr in ("marker_sets", "markersets", "markerSets"):
        if hasattr(df, attr):
            ms_list = getattr(df, attr)
            break
    if not ms_list:
        return centroids
    for ms in ms_list:
        name = getattr(ms, "model_name", getattr(ms, "name", None))
        pts  = getattr(ms, "marker_pos_list", None)
        if name and pts and len(pts) > 0:
            arr = np.asarray(pts, dtype=float)
            centroids[str(name)] = arr.mean(axis=0)
    return centroids

def parse_rb_states(df: DataFrame) -> Dict[str, dict]:
    rb_list = None
    for attr in ("rigid_bodies", "rbodies", "rbs"):
        if hasattr(df, attr):
            rb_list = getattr(df, attr)
            break
    if not rb_list:
        return {}

    rb_items = []
    for rb in rb_list:
        rid = getattr(rb, "id_num", getattr(rb, "id", getattr(rb, "rigid_body_id", None)))
        pos = getattr(rb, "pos", None)
        rot = getattr(rb, "rot", None)
        if pos is None and all(hasattr(rb, k) for k in ("x", "y", "z")):
            pos = (rb.x, rb.y, rb.z)
        if rot is None and all(hasattr(rb, k) for k in ("qx", "qy", "qz", "qw")):
            rot = (rb.qx, rb.qy, rb.qz, rb.qw)
        valid = bool(getattr(rb, "tracking_valid", True))
        if rid is None or pos is None or rot is None:
            continue
        pos = np.asarray(pos, dtype=float).reshape(3)
        rot = np.asarray(rot, dtype=float).reshape(4)
        rb_items.append((int(rid), pos, rot, valid))

    if not rb_items:
        return {}

    states: Dict[str, dict] = {}
    centroids = _markerset_centroids(df)

    for rid, pos, quat, valid in rb_items:
        name = id2name.get(rid) if id2name else None
        if not name and centroids:
            nearest_name, nearest_d = None, float("inf")
            for ms_name, c in centroids.items():
                d = float(np.linalg.norm(pos - c))
                if d < nearest_d:
                    nearest_d, nearest_name = d, ms_name
            name = nearest_name
        if not name:
            name = f"RB_{rid}"
        states[name] = {"pos": pos, "quat": quat, "valid": valid}
    return states

# ---------- save/load target ----------
def save_pose(path: Path, rb_states: Dict[str, dict]):
    if RB_GRID not in rb_states or RB_SUBJ not in rb_states:
        return
    g = rb_states[RB_GRID]; s = rb_states[RB_SUBJ]
    if not (g.get("valid", True) and s.get("valid", True)):
        return
    Rg = rotmat_from_quat(g["quat"]); pg = g["pos"]
    Rs = rotmat_from_quat(s["quat"]); ps = s["pos"]
    # R_rel, t_rel = relative_RT(Rs, ps, Rg, pg)

    R_rel, t_rel = relative_RT_with_points(
        Rs, ps, SUBJ_BONE_TIP,
        Rg, pg, GRID_LOCAL_POINT
    )

    data = {
        "target_rel": {"R_rel": R_rel.tolist(), "t_rel": t_rel.tolist()},
        "abs": {
            RB_GRID: {"pos": pg.tolist(), "quat": g["quat"].tolist()},
            RB_SUBJ: {"pos": ps.tolist(), "quat": s["quat"].tolist()}
        }
    }
    path.write_text(json.dumps(data, indent=2))

def load_pose(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if "target_rel" in data:
        return data
    try:
        g = data[RB_GRID]; s = data[RB_SUBJ]
        Rg = rotmat_from_quat(np.array(g["quat"], float)); pg = np.array(g["pos"], float)
        Rs = rotmat_from_quat(np.array(s["quat"], float)); ps = np.array(s["pos"], float)
        # R_rel, t_rel = relative_RT(Rs, ps, Rg, pg)

        R_rel, t_rel = relative_RT_with_points(
            Rs, ps, SUBJ_BONE_TIP,
            Rg, pg, GRID_LOCAL_POINT
        )

        return {"target_rel": {"R_rel": R_rel.tolist(), "t_rel": t_rel.tolist()}, "abs": {RB_GRID: g, RB_SUBJ: s}}
    except Exception:
        return None

def _save_snapshot_row(timestamp_iso: str,
                       trans_err_mm: float,
                       rot_err_deg: float,
                       score: Optional[float],
                       delta_vec_mm: list,
                       within_tol: bool) -> Path:
    """Append a CSV row and write a per-snapshot JSON file."""
    rec = {
        "timestamp": timestamp_iso,
        "translation_error_mm": float(trans_err_mm),
        "rotation_error_deg": float(rot_err_deg),
        "score": (None if score is None else float(score)),
        "within_tol": within_tol,
        "delta_vec_mm_x": float(delta_vec_mm[0]),
        "delta_vec_mm_y": float(delta_vec_mm[1]),
        "delta_vec_mm_z": float(delta_vec_mm[2]),
    }
    # CSV append (write header if new)
    need_header = not SNAP_CSV.exists()
    with SNAP_CSV.open("a", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=list(rec.keys()))
        if need_header:
            w.writeheader()
        w.writerow(rec)

    # JSON
    safe_ts = timestamp_iso.replace(":", "-")
    (SNAP_JSON_DIR / f"snapshot_{safe_ts}.json").write_text(json.dumps(rec, indent=2))
    json_path = SNAP_JSON_DIR / f"snapshot_{safe_ts}.json"
    json_path.write_text(json.dumps(rec, indent=2))
    return json_path

# ---------- UI control (save/report) ----------
class Control:
    def __init__(self):
        self._save_target = False
        self._report_delta = False

    def request_save(self):
        self._save_target = True
    
    def request_report(self):
        self._report_delta = True

    def on_report_clicked(self):
        self._report_delta = True

    def pop_save(self):
        if self._save_target:
            self._save_target = False
            return True
        return False

    def pop_report(self):
        if self._report_delta:
            self._report_delta = False
            return True
        return False

# ---------- main loop callable ----------
def run_tracker(on_update: Optional[Callable[[dict], None]] = None,
                stop_flag: Optional[Callable[[], bool]] = None,
                control: Optional[Control] = None):
    if control is None:
        control = Control()

    client = NatNetClient(server_ip_address=SERVER_IP, local_ip_address=LOCAL_IP, use_multicast=False)
    client.on_data_description_received_event.handlers.append(on_desc)
    client.on_data_frame_received_event.handlers.append(on_frame)

    csv_file = CSV_PATH.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["time_unix_s", "trans_err_mm", "rot_err_deg", "score"])

    last_log = 0.0
    last_emit = 0.0

    with client:
        client.request_modeldef()
        try:
            while True:
                if stop_flag and stop_flag():
                    break

                client.update_sync()
                if latest_frame is None:
                    time.sleep(0.01)
                    continue

                rb_states = parse_rb_states(latest_frame)
                ok = all([
                    RB_GRID in rb_states, RB_SUBJ in rb_states,
                    rb_states[RB_GRID].get("valid", True),
                    rb_states[RB_SUBJ].get("valid", True)
                ])
                now = time.time()

                if ok:
                    g = rb_states[RB_GRID]
                    s = rb_states[RB_SUBJ]
                    Rg = rotmat_from_quat(g["quat"]); pg = g["pos"]
                    Rs = rotmat_from_quat(s["quat"]); ps = s["pos"]
                    # R_rel, t_rel = relative_RT(Rs, ps, Rg, pg)

                    
                    R_rel, t_rel = relative_RT_with_points(
                        Rs, ps, SUBJ_BONE_TIP,
                        Rg, pg, GRID_LOCAL_POINT
                    )


                    # UI requests
                    if control.pop_save():
                        save_pose(POSE_FILE, rb_states)

                        # emit a short save-report so the UI can react (timestamp + path + snapshot of RB states)
                        if on_update:
                            iso_save = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time()))
                            on_update({
                                "event": "save_target",
                                "ts": iso_save,
                                "json": str(POSE_FILE),
                                "grid": {
                                    "pos": pg.tolist(),
                                    "quat": g["quat"].tolist()
                                },
                                "subject": {
                                    "pos": ps.tolist(),
                                    "quat": s["quat"].tolist()
                                },
                                "within_tol": within_tol
                            })


                    if control.pop_report():
                        stamp = time.time()
                        iso   = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp))
                        tgt = load_pose(POSE_FILE)

                        if tgt and "target_rel" in tgt:
                            R_rel_0 = np.array(tgt["target_rel"]["R_rel"], float)
                            t_rel_0 = np.array(tgt["target_rel"]["t_rel"], float)

                            # --- define the vector you want to report (subject-frame mm) ---
                            d_vec_mm = (t_rel - t_rel_0) * 1000.0           # shape (3,)
                            rot_err  = rotation_geodesic_deg(R_rel_0, R_rel)

                            # optional score for the snapshot (same formula as live)
                            trans_err = float(np.linalg.norm(d_vec_mm))
                            trans_score = max(0.0, 1.0 - trans_err / max(1e-6, TRANS_TOL_MM))
                            rot_score   = max(0.0, 1.0 - rot_err  / max(1e-6, ROT_TOL_DEG))
                            score = 0.5*trans_score + 0.5*rot_score
                            within_tol = (trans_err_mm <= TRANS_TOL_MM) and (rot_err_deg <= ROT_TOL_DEG)

                            json_path = _save_snapshot_row(
                                timestamp_iso=iso,
                                trans_err_mm=trans_err,
                                rot_err_deg=float(rot_err),
                                score=float(score),
                                delta_vec_mm=d_vec_mm.tolist(),
                                within_tol=within_tol
                            )

                            # emit a report packet for the UI
                            if on_update:
                                on_update({
                                    "report":     [round(float(d_vec_mm[0]), 1),
                                                round(float(d_vec_mm[1]), 1),
                                                round(float(d_vec_mm[2]), 1)],
                                    "report_rot": round(float(rot_err), 2),
                                    "report_ts":  iso,
                                    "within_tol": within_tol
                                })
                                on_update({
                                    "event": "snapshot",
                                    "ts": iso,
                                    "csv": str(SNAP_CSV),
                                    "json": str(json_path),
                                    "within_tol": within_tol
                                })

                        else:
                            # no target saved → still ping UI so it can warn or snapshot “no_target”
                            if on_update:
                                on_update({
                                    "report": [0.0, 0.0, 0.0],
                                    "report_rot": 0.0,
                                    "report_ts": iso,
                                    "note": "no_target"
                                })

                    
                    tgt = load_pose(POSE_FILE)
                    if tgt and "target_rel" in tgt:
                        R_rel_0 = np.array(tgt["target_rel"]["R_rel"], float)
                        t_rel_0 = np.array(tgt["target_rel"]["t_rel"], float)

                        # Errors
                        trans_err_mm = float(np.linalg.norm(t_rel - t_rel_0) * 1000.0)
                        rot_err_deg  = float(rotation_geodesic_deg(R_rel_0, R_rel))
                        trans_score = max(0.0, 1.0 - trans_err_mm / max(1e-6, TRANS_TOL_MM))
                        rot_score   = max(0.0, 1.0 - rot_err_deg  / max(1e-6, ROT_TOL_DEG))
                        score = 0.5*trans_score + 0.5*rot_score
                        within_tol = (trans_err_mm <= TRANS_TOL_MM) and (rot_err_deg <= ROT_TOL_DEG)

                        # Direction vector to move Grid in the Subject frame (mm)
                        # Move by (t_rel_0 - t_rel) to reach the target:
                        delta_vec_mm = (t_rel_0 - t_rel) * 1000.0
                        # delta_vec_mm *= -1.0   
                        dx, dy, dz = [float(x) for x in delta_vec_mm]


                        if on_update and (now - last_emit) >= 0.05:
                            on_update({
                                "translation_error_mm": round(trans_err_mm, 2),
                                "rotation_error_deg": round(rot_err_deg, 2),
                                "score": round(score, 2),
                                "within_tol": within_tol,
                                "delta_vec_mm": [round(dx, 2), round(dy, 2), round(dz, 2)]
                            })
                            last_emit = now

                        if (now - last_log) >= LOG_INTERVAL_S:
                            writer.writerow([f"{now:.3f}", f"{trans_err_mm:.2f}", f"{rot_err_deg:.2f}", f"{score:.2f}"])
                            csv_file.flush()
                            last_log = now
                    else:
                        trans_mag_mm = float(np.linalg.norm(t_rel) * 1000.0)
                        if on_update and (now - last_emit) >= 0.1:
                            on_update({
                                "translation_error_mm": None,
                                "rotation_error_deg": None,
                                "score": None,
                                "within_tol": False,
                                "preview_mm": round(trans_mag_mm, 2),
                                "delta_vec_mm": [0.0, 0.0, 0.0]
                            })
                            last_emit = now

                time.sleep(0.005)
        finally:
            csv_file.close()
