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

        for b in (self.btn_start, self.btn_stop, self.btn_save, self.btn_report):
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

        # # live values
        # t = float(data.get("translation_error_mm", 0))
        # r = float(data.get("rotation_error_deg", 0))
        # s = float(data.get("score", 0))
        # vec = data.get("delta_vec_mm", [0.0,0.0,0.0])

        # self.card_trans.value.setText(f"{t:.2f}")
        # self.card_rot.value.setText(f"{r:.2f}")
        # self.card_score.value.setText(f"{s:.2f}")

        # ok = bool(data.get("within_tol", False))
        # self.status_pill.setText("OK" if ok else "ADJUST")
        # self.status_pill.setStyleSheet(PILL_OK_STYLE if ok else PILL_ADJ_STYLE)
      
        # # Update 3D arrow (direction & magnitude)
        # self.arrow.set_vector_mm(subject_to_view(vec))

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




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    app.setStyle("Fusion")
    w = TrackerUI()
    w.resize(1180, 520)
    w.show()
    sys.exit(app.exec_())
