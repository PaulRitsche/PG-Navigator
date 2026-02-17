import sys, math
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import MeshData, GLMeshItem
from pathlib import Path
from datetime import datetime
import json
import logging
import tracker  # tracker.py alongside
import numpy as np


# TODO Handle loading/saving target poses in UI 
# TODO Handle setting data paths in UI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PRIMARY_BG = "#0f172a"  # slate-900
CARD_BG    = "#111827"  # gray-900
TEXT_MAIN  = "#e5e7eb"  # gray-200
TEXT_DIM   = "#9ca3af"  # gray-400
ACCENT     = "#60a5fa"  # blue-400

PILL_OK_STYLE = """
QLabel {
    background-color: #064e3b;
    color: #a7f3d0;
    border-radius: 14px;
    padding: 6px 12px;
    font-weight: 600;
}
"""
PILL_ADJ_STYLE = """
QLabel {
    background-color: #7f1d1d;
    color: #fecaca;
    border-radius: 14px;
    padding: 6px 12px;
    font-weight: 600;
}
"""
PILL_NEUTRAL_STYLE = PILL_ADJ_STYLE  # reuse red style for "NO TARGET"


def big_value(label_text="—"):
    lbl = QtWidgets.QLabel(label_text)
    lbl.setAlignment(QtCore.Qt.AlignCenter)
    lbl.setStyleSheet("font-size: 56px; font-weight: 700; color: %s;" % TEXT_MAIN)
    lbl.setMinimumHeight(88)
    return lbl

def caption(text):
    lbl = QtWidgets.QLabel(text)
    lbl.setAlignment(QtCore.Qt.AlignCenter)
    lbl.setStyleSheet("font-size: 14px; color: %s; letter-spacing: 1px;" % TEXT_DIM)
    return lbl

def pill(text, ok=True):
    lab = QtWidgets.QLabel(text)
    lab.setAlignment(QtCore.Qt.AlignCenter)
    lab.setStyleSheet(PILL_OK_STYLE if ok else PILL_ADJ_STYLE)
    return lab

def subject_to_view(vec_mm):
    """Convert a vector in Subject frame (Z up) to View frame (Y up)."""
    x, y, z = vec_mm
    return (-x, -z, -y)   # swap Y<->Z and invert new Z

class Card(QtWidgets.QFrame):
    def __init__(self, title):
        super().__init__()
        self.setStyleSheet(f"QFrame {{ background-color: {CARD_BG}; border-radius: 16px; }}")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        self.title = caption(title)
        lay.addWidget(self.title)
        self.value = big_value("—")
        lay.addWidget(self.value)

class GLIndicators:
    """
    Top-view scene overlays:
      - Crosshair centered at origin (reference point)
      - Semi-transparent cone showing translation+tilt (subject frame)
      - Dotted axis along the cone
      - Rotation dial (circle) with floating point at rotation_error angle (magnitude)
    """
    def __init__(self, parent_view: gl.GLViewWidget):
        self.view = parent_view

        # --- Scene / camera: TOP-DOWN (look down +Z) ---
        self.view.setBackgroundColor(QtGui.QColor(12, 18, 36))
        self.view.opts['distance']   = 260
        self.view.opts['elevation']  = 90   # top view
        self.view.opts['azimuth']    = 0
        self.view.opts['fov']        = 40

        # --- Ground grid for orientation (optional, faint) ---
        grid = gl.GLGridItem(glOptions='additive')
        grid.setSize(220, 220, 0)
        grid.setSpacing(10, 10, 10)
        grid.setColor((255, 255, 255, 60))
        grid.rotate(90, 1, 0, 0)  # lay it in XY plane (Z up)
        self.view.addItem(grid)

        # Crosshair (X and Y axes on Z=0)
        self.cross_x = gl.GLLinePlotItem()
        self.cross_y = gl.GLLinePlotItem()
        self._update_crosshair(length=200.0)

        # Cone (semi-transparent) aligned with live vector (Z-up base) 
        self.cone_md  = MeshData.cylinder(rows=16, cols=48, radius=[0.0, 1.0], length=1.0)  # unit cone along +Z
        self.cone     = GLMeshItem(meshdata=self.cone_md, smooth=True, drawFaces=True, drawEdges=False, glOptions='additive')
        self.cone.setColor((0.38, 0.64, 0.98, 0.28)) 
        self.view.addItem(self.cone)

        # Dotted axis along the cone (made from small points)
        self.axis_points = gl.GLScatterPlotItem(size=4.0, pxMode=True)
        self.axis_points.setGLOptions('additive')
        self.axis_points.setData(pos=np.zeros((1,3)), color=(1,1,1,0.7))
        self.view.addItem(self.axis_points)

        # Rotation dial (circle in XY plane) + floating point marker 
        self.dial_radius = 60.0
        self.dial = gl.GLLinePlotItem(glOptions='additive')
        circ_pts = self._circle_xy(self.dial_radius, n=180)
        self.dial.setData(pos=circ_pts, color=(1,1,1,0.3), width=2)
        self.view.addItem(self.dial)

        self.rot_marker = gl.GLScatterPlotItem(size=8.0, pxMode=True)
        self.rot_marker.setData(pos=np.array([[self.dial_radius, 0.0, 0.0]]), color=(0.95,0.75,0.2,1.0), size=2)
        self.view.addItem(self.rot_marker)

        # initialize with zero vector
        self.set_translation_vec_mm((0.0, 0.0, 0.0))
        self.set_rotation_deg(0.0)

    # ---------- helpers ----------
    def _update_crosshair(self, length=200.0):
        # X axis line: from -L..+L on X, Y=0, Z=0
        xs = np.array([[-length, 0.0, 0.0], [length, 0.0, 0.0]], dtype=float)
        ys = np.array([[0.0, -length, 0.0], [0.0, length, 0.0]], dtype=float)
        self.cross_x.setData(pos=xs, color=(1,1,1,0.6), width=2)
        self.cross_y.setData(pos=ys, color=(1,1,1,0.6), width=2)
        self.view.addItem(self.cross_x)
        self.view.addItem(self.cross_y)

    @staticmethod
    def _unit_and_len(vec):
        x,y,z = vec
        L = math.sqrt(x*x + y*y + z*z)
        if L < 1e-6:
            return (0.0,0.0,1.0), 0.0
        return (x/L, y/L, z/L), L

    @staticmethod
    def _axis_angle_from_z(u):
        # rotate from +Z to unit vector u
        ux, uy, uz = u
        # axis = zhat × u
        ax = 0.0*uz - 1.0*uy
        ay = 1.0*ux - 0.0*uz
        az = 0.0*uy - 0.0*ux
        dot = max(-1.0, min(1.0, 1.0*uz + 0.0*uy + 0.0*ux))
        ang_deg = math.degrees(math.acos(dot))
        if abs(ax)+abs(ay)+abs(az) < 1e-6:
            ax, ay, az = (1.0, 0.0, 0.0)
        return ang_deg, ax, ay, az

    @staticmethod
    def _circle_xy(radius, n=180, z=0.0):
        th = np.linspace(0, 2*np.pi, n, endpoint=True)
        x = radius*np.cos(th); y = radius*np.sin(th)
        z = np.full_like(x, z)
        return np.vstack([x,y,z]).T

    def _set_cone_transform(self, vec_mm):
        """
        Place a unit cone at origin, scale it to the vector length, rotate it toward vec direction.
        The cone base remains at origin; cone extends along the vector.
        """
        u, L = self._unit_and_len(vec_mm)

        # Reset transform
        self.cone.resetTransform()

        # Scale: unit cone has length 1 along +Z and radius 1 at top → scale Z by L, XY by (L * opening)
        # Opening: make radius proportional to length so "tilt" is visible. Use mild opening.
        opening = 0.20  # radius at far end is opening*L
        self.cone.scale(opening*L, opening*L, L)

        # Move so cone base at origin (unit cone base is at z=0, tip at z=+1 → OK already)

        # Rotate from +Z to direction
        ang, ax, ay, az = self._axis_angle_from_z(u)
        self.cone.rotate(ang, ax, ay, az)

    def _set_dotted_axis(self, vec_mm, dots=20):
        """
        Scatter points along the vector to emulate a dotted axis inside the cone.
        """
        u, L = self._unit_and_len(vec_mm)
        if L < 1e-6:
            pts = np.zeros((1,3))
        else:
            t = np.linspace(0.0, L, dots)
            pts = np.stack([u[0]*t, u[1]*t, u[2]*t], axis=1)
        self.axis_points.setData(pos=pts, color=(1,1,1,0.7))

    def set_translation_vec_mm(self, vec_mm):
        """
        vec_mm is the live delta (Subject frame) you already send to the view, after subject_to_view().
        This updates: the cone transform and the dotted axis.
        """
        self._set_cone_transform(vec_mm)
        self._set_dotted_axis(vec_mm)

    def set_rotation_deg(self, angle_deg, signed=False):
        """
        Places a floating point on a top-view dial at 'angle_deg'.
        NOTE: with current backend we receive magnitude only, so we plot at +angle.
              If you later stream a signed angle, pass signed=True and a signed value.
        """
        a = math.radians(angle_deg if signed else abs(angle_deg))
        x = self.dial_radius * math.cos(a)
        y = self.dial_radius * math.sin(a)
        self.rot_marker.setData(pos=np.array([[x, y, 0.0]]), color=(0.95,0.75,0.2,1.0), size = 15)

class GLArrow:
    """
    High-visibility 3D arrow: shaft (cylinder) + head (cone).
    Oriented from origin to vec_mm (in mm). No grid/axes.
    """
    def __init__(self, parent_view: gl.GLViewWidget):
        self.view = parent_view
        self.upp_axis = "Y"  # which axis is up in the global frame
       
        # Background a touch lighter for contrast
        self.view.setBackgroundColor(QtGui.QColor(12, 18, 36))

        # Camera params
        self.view.opts['distance'] = 260    # camera distance
        self.view.opts['elevation'] = 18
        self.view.opts['azimuth'] = 90
        self.view.opts['fov'] = 40

        # Base meshes (unit size; we scale/rotate/translate each update)
        self.shaft_md = MeshData.cylinder(rows=16, cols=32, radius=[1.0, 1.0], length=1.0)
        self.shaft = GLMeshItem(meshdata=self.shaft_md, smooth=True, drawFaces=True, drawEdges=False)
        self.shaft.setGLOptions('additive')  # brighter
        self.view.addItem(self.shaft)

        # Add Grid
        grid = gl.GLGridItem(glOptions='additive')
        grid.setSize(200, 200, 200)
        grid.setSpacing(10, 10, 10)
        grid.setColor((255, 255, 255, 76.5))
        self.view.addItem(grid)  # ground grid

        # Add Axis
        axis = gl.GLAxisItem()
        axis.setSize(100, 100, 100)
        self.view.addItem(axis)  # XYZ axis

        # current vector
        self.set_vector_mm((0.0, 0.0, 0.0))

    @staticmethod
    def _unit_and_len(vec):
        x,y,z = vec
        L = math.sqrt(x*x + y*y + z*z)
        if L < 1e-6:
            return (0.0,0.0,1.0), 0.0  # default dir if zero
        return (x/L, y/L, z/L), L

    @staticmethod
    def _blend_color(length_mm, full_scale_mm=60.0):
        # 0 mm => green; >= full_scale => red
        t = max(0.0, min(length_mm / full_scale_mm, 1.0))
        r = 0.1 + 0.9*t
        g = 0.9*(1.0 - t) + 0.1*t
        b = 0.2*(1.0 - t) + 0.1*t
        return (r, g, b, 1.0)

    @staticmethod
    def _axis_angle_from_z(u):
        # rotate from +Z to unit vector u  (pyqtgraph is Z-up)
        zx, zy, zz = 0.0, 0.0, 1.0
        ux, uy, uz = u
        ax = zy*uz - zz*uy
        ay = zz*ux - zx*uz
        az = zx*uy - zy*ux
        dot = max(-1.0, min(1.0, zz*uz + zy*uy + zx*ux))
        angle_deg = math.degrees(math.acos(dot))
        if abs(ax) + abs(ay) + abs(az) < 1e-6:
            ax, ay, az = (1.0, 0.0, 0.0)
        return angle_deg, ax, ay, az


    def set_vector_mm(self, vec_mm):
        # Decompose
        u, L = self._unit_and_len(vec_mm)
        # Visibility: always show something, but clamp tiny arrows
        show_L = max(L, 1.0)     # visually visible even if tiny
        #head_len = min(40.0, max(8.0, 0.18 * show_L))
        #shaft_len = max(0.0, show_L - head_len)
        shaft_len = max(0.0, show_L)
        shaft_radius = 3.5
        color = self._blend_color(L)

        # Reset transforms, then scale/rotate/translate each part
        #for part in (self.shaft, self.head):
        for part in (self.shaft,):
            part.resetTransform()
            part.setColor(color)

        # Orient from +Z to target dir
        ang, ax, ay, az = self._axis_angle_from_z(u)

        # Shaft: unit cylinder along +Z from -0.5..+0.5 → we scale Z to length, then move to sit on origin
        if shaft_len > 0.1:
            self.shaft.scale(shaft_radius, shaft_radius, shaft_len)
            # move so base at origin: translate by +shaft_len/2 along +Z, then rotate toward u
            self.shaft.translate(0, 0, shaft_len/2.0)
            self.shaft.rotate(ang, ax, ay, az)
        else:
            # collapse shaft if nearly zero
            self.shaft.scale(0.001, 0.001, 0.001)


class TrackerWorker(QtCore.QObject):
    status = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()

    def __init__(self, control: tracker.Control):
        super().__init__()
        self._stop = False
        self._control = control

    def stop(self):
        self._stop = True

    def _stop_flag(self):
        return self._stop

    def _on_update(self, data: dict):
        self.status.emit(data)

    @QtCore.pyqtSlot()
    def run(self):
        try:
            tracker.run_tracker(on_update=self._on_update,
                                stop_flag=self._stop_flag,
                                control=self._control)
        finally:
            self.finished.emit()

# ---- Abs error dialog helpers (PyQt5 + pyqtgraph) ----
def _snapshot_label(d: dict, fallback_name: str = "") -> str:
    ts = d.get("ts", "") or d.get("timestamp", "")
    if ts:
        return ts
    return fallback_name or "snapshot"

def _safe_read_snapshot_meta(json_path: str) -> dict:
    """
    Read snapshot JSON and return a small dict with:
      - ts label
      - json path
      - extracted position (if possible)
    Never throws; returns minimal info.
    """
    p = Path(json_path)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        pos = _extract_position_mm(d)  # uses your tolerant extractor
        return {
            "ts": _snapshot_label(d, p.name),
            "json": str(p),
            "pos_mm": pos.tolist(),
        }
    except Exception:
        return {
            "ts": p.name,
            "json": str(p),
            "pos_mm": None,
        }

def _scan_snapshot_folder(folder: str, pattern: str = "*.json") -> list:
    """
    Scan a folder recursively for snapshot JSONs.
    Returns a list of history entries: {"ts":..., "json":..., "pos_mm":...}
    """
    folder = str(folder)
    base = Path(folder)
    out = []
    for p in base.rglob(pattern):
        out.append(_safe_read_snapshot_meta(str(p)))
    # newest first if ts looks sortable; otherwise keep filesystem order
    return out

def _load_snapshot_like(obj_or_path):
    """
    Accepts:
      - dict already loaded
      - path to .json file
    Returns dict.
    """
    if isinstance(obj_or_path, dict):
        return obj_or_path
    p = Path(obj_or_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))

def _extract_vec_and_rot(d: dict):
    """
    Extracts (delta_vec_mm, rotation_error_deg) from a snapshot-like dict.
    Make tolerant here if your keys differ.
    """
    # vector
    vec = d.get("delta_vec_mm", None)
    if vec is None:
        # fallback attempts
        vec = d.get("delta_vec", d.get("delta_mm", d.get("vec_mm", [0.0, 0.0, 0.0])))
    vec = np.array(vec, dtype=float).reshape(3,)

    # rotation
    rot = d.get("rotation_error_deg", None)
    if rot is None:
        rot = d.get("rot_error_deg", d.get("rotation_deg", 0.0))
    rot = float(rot or 0.0)
    return vec, rot

def compute_abs_errors_vec_rot(ref: dict, other: dict):
    """
    Absolute errors in SAME coordinate system:
      - component abs errors (mm)
      - euclidean distance (mm)
      - abs rotation error difference (deg)
    """
    v_ref, r_ref = _extract_vec_and_rot(ref)
    v_oth, r_oth = _extract_vec_and_rot(other)

    dv = v_oth - v_ref
    abs_comp = np.abs(dv)
    dist = float(np.linalg.norm(dv))
    abs_rot = float(abs(r_oth - r_ref))

    return {
        "v_ref": v_ref, "v_oth": v_oth,
        "abs_comp_mm": abs_comp,
        "dist_mm": dist,
        "rot_abs_deg": abs_rot,
        "rot_ref_deg": r_ref,
        "rot_oth_deg": r_oth,
    }

def _extract_position_mm(d: dict) -> np.ndarray:
    """
    Try to extract an absolute 3D position (mm) from snapshot-like dict.
    Supported:
      - current_pos_mm: [x,y,z]
      - current_pose: {"pos_mm":[x,y,z]} or {"pos":[...]} or {"t_mm":[...]}
      - target_pos_mm similarly
    Fallback:
      - uses delta_vec_mm as a pseudo-position (error-space endpoint)
    """
    # 1) direct absolute position
    if "current_pos_mm" in d and d["current_pos_mm"] is not None:
        return np.array(d["current_pos_mm"], dtype=float).reshape(3,)

    # 2) nested pose forms
    pose = d.get("current_pose", None)
    if isinstance(pose, dict):
        for k in ("pos_mm", "pos", "t_mm", "t"):
            if k in pose and pose[k] is not None:
                return np.array(pose[k], dtype=float).reshape(3,)

    # 3) fallback to delta vector endpoint (NOT absolute, but still visualizable)
    vec, _ = _extract_vec_and_rot(d)
    return np.array(vec, dtype=float).reshape(3,)

def _as_pos3(arr, fallback=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Return (1,3) float32 finite array suitable for GLScatterPlotItem."""
        try:
            a = np.asarray(arr, dtype=np.float32).reshape(-1)
            if a.size < 3:
                a = np.array(fallback, dtype=np.float32)
            else:
                a = a[:3]
        except Exception:
            a = np.array(fallback, dtype=np.float32)

        # sanitize NaN/Inf
        if not np.all(np.isfinite(a)):
            a = np.array(fallback, dtype=np.float32)

        return a.reshape(1, 3)

def _as_path_pts(pts_list, fallback_a, fallback_b) -> np.ndarray:
    """Return (N,3) float32 finite path array."""
    clean = []
    for p in pts_list:
        try:
            a = np.asarray(p, dtype=np.float32).reshape(-1)[:3]
            if a.size == 3 and np.all(np.isfinite(a)):
                clean.append(a)
        except Exception:
            pass

    if len(clean) == 0:
        clean = [np.asarray(fallback_a, dtype=np.float32).reshape(-1)[:3],
                np.asarray(fallback_b, dtype=np.float32).reshape(-1)[:3]]

    return np.vstack(clean).astype(np.float32)

class AbsErrorDialog(QtWidgets.QDialog):
    """
    Compare:
      - Snapshot vs Snapshot (choose two from history)
      - Live vs Snapshot (current live status vs chosen snapshot)
    Shows bar plots + textual stats.
    """
    def __init__(self, parent, *, history, get_live_dict_callable):
        super().__init__(parent)
        self.setWindowTitle("Absolute Errors (same coordinate system)")
        self.resize(900, 520)

        self.history = history  # list of {"ts":..., "json":..., "csv":...}
        self.get_live = get_live_dict_callable

        root = QtWidgets.QVBoxLayout(self)

        # --- selectors
        row = QtWidgets.QHBoxLayout()
        root.addLayout(row)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Snapshot vs Snapshot", "Live vs Snapshot"])
        row.addWidget(QtWidgets.QLabel("Mode:"))
        row.addWidget(self.mode, 1)

        self.btn_add_folder = QtWidgets.QPushButton("Add folder…")
        self.btn_add_files  = QtWidgets.QPushButton("Add files…")
        row.addWidget(self.btn_add_folder)
        row.addWidget(self.btn_add_files)


        self.cmb_a = QtWidgets.QComboBox()
        self.cmb_b = QtWidgets.QComboBox()
        row.addWidget(QtWidgets.QLabel("A (ref):"))
        row.addWidget(self.cmb_a, 2)
        row.addWidget(QtWidgets.QLabel("B (other):"))
        row.addWidget(self.cmb_b, 2)

        self.btn_compute = QtWidgets.QPushButton("Compute")
        row.addWidget(self.btn_compute)

        # --- plots + 3D + stats
        mid = QtWidgets.QHBoxLayout()
        root.addLayout(mid, 1)

        # LEFT: 3D scene
        self.view3d = gl.GLViewWidget()
        self.view3d.setBackgroundColor(QtGui.QColor(17, 24, 39))  # similar to CARD_BG
        self.view3d.opts["distance"] = 400
        self.view3d.opts["elevation"] = 18
        self.view3d.opts["azimuth"] = 45

        # optional grid/axis
        grid = gl.GLGridItem(glOptions="additive")
        grid.setSize(400, 400, 0)
        grid.setSpacing(20, 20, 20)
        grid.setColor((255, 255, 255, 60))
        grid.rotate(90, 1, 0, 0)  # put in XY plane
        self.view3d.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(120, 120, 120)
        self.view3d.addItem(axis)

        # 3D objects: points and paths
        self.pt_ref = gl.GLScatterPlotItem(size=10, pxMode=True)
        self.pt_oth = gl.GLScatterPlotItem(size=10, pxMode=True)
        self.pt_path = gl.GLLinePlotItem(glOptions="additive", width=2)

        # ref->other vector
        self.vec_line = gl.GLLinePlotItem(glOptions="additive", width=3)

        self.view3d.addItem(self.pt_path)
        self.view3d.addItem(self.vec_line)
        self.view3d.addItem(self.pt_ref)
        self.view3d.addItem(self.pt_oth)

        # wrap 3D in a frame for style consistency
        view_wrap = QtWidgets.QFrame()
        view_wrap.setStyleSheet(f"QFrame {{ background-color: {CARD_BG}; border-radius: 12px; }}")
        vlay = QtWidgets.QVBoxLayout(view_wrap)
        vcap = QtWidgets.QLabel("3D pose/path view (mm)")
        vcap.setStyleSheet(f"color:{TEXT_DIM}; padding:6px;")
        vlay.addWidget(vcap)
        vlay.addWidget(self.view3d, 1)

        mid.addWidget(view_wrap, 2)

        # MIDDLE: bar plots (same as before)
        plots_col = QtWidgets.QVBoxLayout()
        mid.addLayout(plots_col, 2)

        self.plot_vec = pg.PlotWidget()
        self.plot_vec.setBackground(CARD_BG)
        self.plot_vec.showGrid(x=True, y=True, alpha=0.2)
        self.plot_vec.setTitle("Abs component errors |Δx| |Δy| |Δz| (mm)", color=TEXT_MAIN)

        self.plot_rot = pg.PlotWidget()
        self.plot_rot.setBackground(CARD_BG)
        self.plot_rot.showGrid(x=True, y=True, alpha=0.2)
        self.plot_rot.setTitle("Abs rotation error |Δrot| (deg)", color=TEXT_MAIN)

        plots_col.addWidget(self.plot_vec, 2)
        plots_col.addWidget(self.plot_rot, 1)

        # RIGHT: stats
        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setStyleSheet(f"background-color:{CARD_BG}; color:{TEXT_MAIN}; border-radius:8px; padding:8px;")
        mid.addWidget(self.txt, 2)

        # populate combos
        self._refresh_history()

        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_compute.clicked.connect(self._on_compute)
        self.btn_add_folder.clicked.connect(self._on_add_folder)
        self.btn_add_files.clicked.connect(self._on_add_files)


        self._on_mode_changed()

    def _refresh_history(self):
        """
        Populate combos from self.history (which may include runtime snapshots + imported ones).
        De-duplicate by json path.
        """
        # de-dup by json path
        seen = set()
        new_hist = []
        for h in self.history:
            jp = h.get("json")
            if not jp:
                continue
            jp = str(Path(jp))
            if jp in seen:
                continue
            seen.add(jp)
            new_hist.append(h)
        self.history = new_hist

        # fill combos
        self.cmb_a.clear()
        self.cmb_b.clear()
        for h in self.history:
            jp = Path(h.get("json", ""))
            label = f"{h.get('ts', jp.name)}  —  {jp.parent.name}/{jp.name}"
            self.cmb_a.addItem(label, userData=h)
            self.cmb_b.addItem(label, userData=h)

        if self.cmb_a.count() > 0 and self.cmb_a.currentIndex() < 0:
            self.cmb_a.setCurrentIndex(0)
        if self.cmb_b.count() > 1:
            self.cmb_b.setCurrentIndex(self.cmb_b.count() - 1)
        elif self.cmb_b.count() == 1:
            self.cmb_b.setCurrentIndex(0)

    def _on_add_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder with snapshot JSONs")
        if not folder:
            return

        added = _scan_snapshot_folder(folder, pattern="*.json")
        if not added:
            QtWidgets.QMessageBox.information(self, "No snapshots found", "No .json files found in that folder.")
            return

        self.history.extend(added)
        self._refresh_history()
        QtWidgets.QMessageBox.information(self, "Snapshots added", f"Added {len(added)} snapshot JSON files.")

    def _on_add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select snapshot JSON files", "", "Snapshot JSON (*.json)"
        )
        if not paths:
            return

        added = [_safe_read_snapshot_meta(p) for p in paths]
        self.history.extend(added)
        self._refresh_history()
        QtWidgets.QMessageBox.information(self, "Snapshots added", f"Added {len(added)} snapshot JSON files.")


    def _on_mode_changed(self):
        m = self.mode.currentText()
        if m == "Live vs Snapshot":
            self.cmb_a.setEnabled(False)  # A is live
            self.cmb_a.setToolTip("A is LIVE in this mode")
        else:
            self.cmb_a.setEnabled(True)
            self.cmb_a.setToolTip("")

    def _barplot_vec(self, abs_comp_mm):
        self.plot_vec.clear()
        # 3 bars at x=0,1,2
        x = np.array([0, 1, 2], dtype=float)
        y = np.array(abs_comp_mm, dtype=float)
        bg = pg.BarGraphItem(x=x, height=y, width=0.6, brush=pg.mkBrush(96, 165, 250, 190))
        self.plot_vec.addItem(bg)
        ax = self.plot_vec.getAxis("bottom")
        ax.setTicks([[(0, "x"), (1, "y"), (2, "z")]])
        self.plot_vec.setLabel("left", "mm")

    def _barplot_rot(self, rot_abs_deg):
        self.plot_rot.clear()
        x = np.array([0], dtype=float)
        y = np.array([rot_abs_deg], dtype=float)
        bg = pg.BarGraphItem(x=x, height=y, width=0.6, brush=pg.mkBrush(245, 158, 11, 200))
        self.plot_rot.addItem(bg)
        ax = self.plot_rot.getAxis("bottom")
        ax.setTicks([[(0, "Δrot")]])
        self.plot_rot.setLabel("left", "deg")

    def _on_compute(self):
        try:
            mode = self.mode.currentText()

            # ---- ALWAYS define these first ----
            ref_dict = None
            oth_dict = None
            a_label = ""
            b_label = ""

            if mode == "Live vs Snapshot":
                # A = snapshot (ref), B = live
                live = self.get_live()
                if not isinstance(live, dict):
                    raise ValueError("Live provider did not return a dict.")

                b = self.cmb_b.currentData()
                if not b or not b.get("json"):
                    raise ValueError("Pick a snapshot (B).")

                ref_dict = _load_snapshot_like(b["json"])
                oth_dict = live

                a_label = "SNAPSHOT (ref)"
                b_label = "LIVE (other)"

            else:  # Snapshot vs Snapshot
                a = self.cmb_a.currentData()
                b = self.cmb_b.currentData()

                if not a or not b or not a.get("json") or not b.get("json"):
                    raise ValueError("Pick two snapshots.")

                ref_dict = _load_snapshot_like(a["json"])
                oth_dict = _load_snapshot_like(b["json"])

                a_label = "A (ref)"
                b_label = "B (other)"

            # ---- COMPUTE ----
            res = compute_abs_errors_vec_rot(ref_dict, oth_dict)

            # ---- RENDER ----
            self._barplot_vec(res["abs_comp_mm"])
            self._barplot_rot(res["rot_abs_deg"])
            self._render_3d(ref_dict, oth_dict)

            # ---- TEXT ----
            self.txt.setPlainText(
                f"{a_label} vec (mm): {res['v_ref'].tolist()}\n"
                f"{b_label} vec (mm): {res['v_oth'].tolist()}\n\n"
                f"Abs component errors (mm): {res['abs_comp_mm'].tolist()}\n"
                f"Euclidean |Δvec| (mm): {res['dist_mm']:.4f}\n\n"
                f"{a_label} rot (deg): {res['rot_ref_deg']:.4f}\n"
                f"{b_label} rot (deg): {res['rot_oth_deg']:.4f}\n"
                f"Abs |Δrot| (deg): {res['rot_abs_deg']:.4f}\n"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Abs error failed", str(e))

    def _render_3d(self, ref_dict: dict, oth_dict: dict):
        """
        Render:
        - ref point and other point
        - vector ref -> other
        - path through snapshot history
        """
        # Extract positions (may fallback to delta_vec_mm)
        p_ref_raw = _extract_position_mm(ref_dict)
        p_oth_raw = _extract_position_mm(oth_dict)

        p_ref = _as_pos3(p_ref_raw)
        p_oth = _as_pos3(p_oth_raw)

        # Build path from history
        path_pts_raw = []
        for h in self.history:
            jp = h.get("json")
            if not jp:
                continue
            try:
                d = _load_snapshot_like(jp)
                p = _extract_position_mm(d)
                path_pts_raw.append(p)
            except Exception:
                continue

        path = _as_path_pts(path_pts_raw, p_ref[0], p_oth[0])

        # ---- IMPORTANT: always pass float32 Nx3 arrays ----
        # Points
        self.pt_ref.setData(
            pos=p_ref,
            color=(0.38, 0.64, 0.98, 1.0),  # RGBA in 0..1
            size=10,
            pxMode=True
        )
        self.pt_oth.setData(
            pos=p_oth,
            color=(0.95, 0.75, 0.20, 1.0),
            size=10,
            pxMode=True
        )

        # Path line
        self.pt_path.setData(
            pos=path,
            color=(1.0, 1.0, 1.0, 0.45),
            mode="line_strip",
            width=2
        )

        # Ref->Other vector line
        vec = np.vstack([p_ref[0], p_oth[0]]).astype(np.float32)
        self.vec_line.setData(
            pos=vec,
            color=(1.0, 0.45, 0.45, 0.85),
            mode="lines",
            width=3
        )

        # Camera center (safe)
        center = path.mean(axis=0)
        if not np.all(np.isfinite(center)):
            center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        span = float(np.max(np.linalg.norm(path - center, axis=1))) if len(path) > 1 else 50.0
        self.view3d.opts["center"] = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self.view3d.opts["distance"] = float(max(200.0, min(1200.0, span * 3.0)))

class TrackerUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OptiTrack Placement — Live")
        self.setStyleSheet(f"QWidget {{ background-color: {PRIMARY_BG}; color: {TEXT_MAIN}; }}")
        self._running = False  


        # --- Header ---
        title = QtWidgets.QLabel("PG-Navigator")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: %s" % TEXT_MAIN)

        self.status_pill = pill("ADJUST", ok=False)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status_pill)

        # --- Cards for values ---
        self.card_trans = Card("TRANSLATION ERROR (mm)")
        self.card_rot   = Card("ROTATION ERROR (°)")
        self.card_score = Card("ALIGNMENT SCORE")

        cards = QtWidgets.QHBoxLayout()
        cards.setSpacing(16)
        cards.addWidget(self.card_trans, 1)
        cards.addWidget(self.card_rot, 1)
        cards.addWidget(self.card_score, 1)

        # --- snapshots ---
        self._last_status = None
        self._snapshot_csv_path = Path("snapshots.csv")
        self._saved_history = [] 
        self._toast_timer = QtCore.QTimer()
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(lambda: self.preview_label.setText(""))


        # --- 3D view (arrow only) ---
        # self.view3d = gl.GLViewWidget()
        # self.arrow = GLArrow(self.view3d)
        self.view3d = gl.GLViewWidget()
        self.ind = GLIndicators(self.view3d)

        view_wrap = QtWidgets.QFrame()
        view_wrap.setStyleSheet(f"QFrame {{ background-color: {CARD_BG}; border-radius: 16px; }}")
        vlay = QtWidgets.QVBoxLayout(view_wrap)
        vlay.setContentsMargins(8,8,8,8)
        vcap = caption("MOVE GRID — Arrow shows Δ to target (Subject frame)")
        vlay.addWidget(vcap)
        vlay.addWidget(self.view3d, 1)

        # --- Preview / messages ---
        self.preview_label = QtWidgets.QLabel("")
        self.preview_label.setStyleSheet("color: %s; font-size: 14px;" % TEXT_DIM)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)

        # --- Buttons ---
        self.btn_start  = QtWidgets.QPushButton("Start")
        self.btn_stop   = QtWidgets.QPushButton("Stop")
        self.btn_save   = QtWidgets.QPushButton("Save Target")
        self.btn_report = QtWidgets.QPushButton("Report Δpos")
        self.btn_abs_err = QtWidgets.QPushButton("Abs Error...")

        for b in (self.btn_start, self.btn_stop, self.btn_save, self.btn_report, self.btn_abs_err):
            b.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            b.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {ACCENT};
                    color: #0b1020;
                    border: none;
                    border-radius: 10px;
                    padding: 10px 16px;
                    font-weight: 700;
                }}
                QPushButton:disabled {{
                    background-color: #374151;
                    color: #9ca3af;
                }}
                """
            )
        self.btn_stop.setStyleSheet(
            """
            QPushButton {
                background-color: #f87171;
                color: #1f2937;
                border: none; border-radius: 10px;
                padding: 10px 16px; font-weight: 700;
            }
            QPushButton:disabled { background-color: #374151; color: #9ca3af; }
            """
        )

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_stop)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_save)
        buttons.addWidget(self.btn_report)
        buttons.addWidget(self.btn_abs_err)

        # --- Root layout ---
        top = QtWidgets.QVBoxLayout()
        top.setSpacing(16)
        top.addLayout(header)
        top.addLayout(cards)
        top.addWidget(self.preview_label)
        top.addLayout(buttons)

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)
        root.addLayout(top, 1)
        root.addWidget(view_wrap, 1)

        # Thread/worker
        self.thread = None
        self.worker = None
        self.control = tracker.Control()

        # Wire buttons
        self.btn_start.clicked.connect(self.start_tracker)
        self.btn_stop.clicked.connect(self.stop_tracker)
        self.btn_save.clicked.connect(self.control.request_save)
        self.btn_report.clicked.connect(self.control.on_report_clicked)
        self.btn_abs_err.clicked.connect(self.open_abs_error_dialog)

        self._set_controls_enabled(False)
    
    # ------- threading control -------
    def start_tracker(self):
        if self._safe_thread_running():
            return

        # Create fresh thread/worker
        self.thread = QtCore.QThread(self)
        self.worker = TrackerWorker(self.control)
        self.worker.moveToThread(self.thread)

        # Wire life-cycle
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.on_status)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # When the thread actually finishes (and C++ object is about to go),
        # clear Python refs so we never touch a deleted object later.
        self.thread.finished.connect(self._on_thread_finished)

        self.thread.start()
        self._running = True
        self.btn_start.setEnabled(False)
        self._set_controls_enabled(True)

    def stop_tracker(self):
        # Tell worker to stop and wait for thread to exit
        if getattr(self, "worker", None):
            self.worker.stop()
        if getattr(self, "thread", None):
            try:
                self.thread.quit()
                self.thread.wait(1500)
            except RuntimeError:
                # already deleted; ignore
                pass
        # UI state will be finalized in _on_thread_finished; do a local fallback too
        self.btn_start.setEnabled(True)
        self._set_controls_enabled(False)

    def _on_thread_finished(self):
        # Final cleanup after the thread is done
        self.worker = None
        self.thread = None
        self._running = False
        self.btn_start.setEnabled(True)
        self._set_controls_enabled(False)


    def closeEvent(self, event):
        try:
            self.stop_tracker()
        finally:
            event.accept()

    def _set_controls_enabled(self, ok: bool):
        self.btn_stop.setEnabled(ok)
        self.btn_save.setEnabled(ok)
        self.btn_report.setEnabled(ok)

    # ------- UI updates -------
    @QtCore.pyqtSlot(dict)
    def on_status(self, data: dict):
        self._last_status = data

        # --- Handle tracker "snapshot" event and exit early ---
        if data.get("event") == "snapshot":

            ts = data.get("ts", "")
            csv_path = data.get("csv")
            json_path = data.get("json")
            self._saved_history.append({"ts": ts, "csv": csv_path, "json": json_path})

            # Toast + pill flip
            self.preview_label.setText(f"Saved snapshot @ {ts}" if ts else "Saved snapshot")
            logger.info(f"Snapshot saved: CSV={csv_path}, JSON={json_path}")
            self._toast_timer.start(2000)
            return   
        
        # --- Handle tracker "save_target" event and exit early ---
        if data.get("event") == "save_target":

            ts = data.get("ts", "")
            json_path = data.get("json")
            self.preview_label.setText(f"Saved target pose @ {ts}" if ts else "Saved target pose")
            logger.info(f"Target pose saved: JSON={json_path}")
            self._toast_timer.start(2000)
            return

        # live values
        t = float(data.get("translation_error_mm", 0) or 0)
        r = float(data.get("rotation_error_deg", 0) or 0)
        s = float(data.get("score", 0) or 0)
        vec = data.get("delta_vec_mm", [0.0, 0.0, 0.0])

        self.card_trans.value.setText(f"{t:.2f}")
        self.card_rot.value.setText(f"{r:.2f}")
        self.card_score.value.setText(f"{s:.2f}")

        ok = bool(data.get("within_tol", False))
        self.status_pill.setText("OK" if ok else "ADJUST")
        self.status_pill.setStyleSheet(PILL_OK_STYLE if ok else PILL_ADJ_STYLE)

        # Update visuals:
        # 1) Translation+tilt cone (use your Subject→View mapping)
        self.ind.set_translation_vec_mm(subject_to_view(vec))
        # 2) Rotation dial (currently magnitude only)
        self.ind.set_rotation_deg(r, signed=False)
    
    def _safe_thread_running(self):
        """Return True if thread exists and is alive; swallow deleted-object states."""
        t = getattr(self, "thread", None)
        if t is None:
            return False
        try:
            return t.isRunning()
        except RuntimeError:
            # underlying C++ object was deleted; clear our reference
            self.thread = None
            return False

    # Absolute error dialog helper
    def _get_live_snapshot_like(self) -> dict:
        """
        Convert current live status (self._last_status) into the same dict schema
        as the saved snapshot JSON, so we can compare apples-to-apples.
        """
        if not self._last_status:
            raise ValueError("No live status yet. Start tracker first.")
        d = self._last_status

        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "delta_vec_mm": d.get("delta_vec_mm", [0.0, 0.0, 0.0]),
            "translation_error_mm": float(d.get("translation_error_mm", 0.0) or 0.0),
            "rotation_error_deg": float(d.get("rotation_error_deg", 0.0) or 0.0),
            "score": float(d.get("score", 0.0) or 0.0),
            "within_tol": bool(d.get("within_tol", False)),
        }

    def open_abs_error_dialog(self):
        if not self._saved_history:
            QtWidgets.QMessageBox.information(
                self,
                "No snapshots yet",
                "Save at least one snapshot first (then you can compare snapshot↔snapshot or live↔snapshot).",
            )
            return

        dlg = AbsErrorDialog(self, history=self._saved_history, get_live_dict_callable=self._get_live_snapshot_like)
        dlg.exec_()




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    app.setStyle("Fusion")
    w = TrackerUI()
    w.resize(1180, 520)
    w.show()
    sys.exit(app.exec_())
