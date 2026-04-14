# PG-Navigator UI — Usage & Navigation Modes

## 1) Start the system
1. Open **Motive** and start streaming NatNet data (**Grid** + **Participant** rigid bodies must be tracked).
2. Run the UI:
   ```bash
   python ui.py
   ```
3. Click **Start** in the UI to connect to NatNet and begin live updates.

You’ll see:
- **Translation error (mm)**, **Rotation error (°)**, and an **Alignment score**
- A **top-view navigation display** (cone/axis + rotation dial)

---

## Buttons

### Start / Stop
- **Start**: begins the NatNet tracking loop in a worker thread.
- **Stop**: stops the thread and disconnects.

### Save Target
Stores the **current Grid↔Participant relative pose** (grid-in-participant frame) into `target_pose.json`.
This defines your **reference target** for the *relative navigation mode*.

### Report Δpos
Creates a **snapshot JSON** in your snapshots folder (and appends to `snapshots.csv`).  
Snapshots are used by **Abs Error…**.

### Abs Error…
Opens the dialog to compare two states:
- Snapshot vs Snapshot
- Live vs Snapshot

and renders a **static 3D scene** of the *absolute rigid body positions*.

### Set baseline…
Select a **baseline snapshot JSON** (typically from the previous session).  
This defines where the **Participant** should be relocated to (absolute mode).

### Compute grid target
After the Participant is relocated, this computes a **corrected absolute target** for the Grid so you can navigate the probe/grid back to the initial placement.

---

## Navigation Modes (dropdown)

The dropdown changes **what the UI considers the “target”** and **what the arrow/cone means**.

> Important:
> - “Relative” mode uses **participant coordinate frame** (subject frame).
> - “Absolute” modes use the **global/world coordinate system** from Motive/NatNet.

---

## Mode 1 — Grid → target (relative)

**Goal:** Move the **Grid (probe)** so it matches the saved target pose relative to the Participant.

### What it uses
- `target_pose.json` (saved via **Save Target**) contains:
  - `R_rel_0`, `t_rel_0` = target relative transform (grid in participant frame)
- Live tracking provides:
  - `R_rel`, `t_rel` = current relative transform

### What is computed (tracker side)
- **Translation error (mm):**
  \[
  \|t_{rel} - t_{rel0}\|\cdot 1000
  \]
- **Rotation error (deg):**
  geodesic angle between \(R_{rel}\) and \(R_{rel0}\)
- **Direction vector displayed in the view:**
  \[
  \Delta_{mm} = (t_{rel0} - t_{rel})\cdot 1000
  \]
This vector is in the **Participant frame**: it tells you how the Grid must move **relative to the Participant** to reach the target.

### What you see
- The cone/arrow points in the direction you should move the **probe/grid**.
- The rotation dial shows rotation mismatch magnitude.

**Use this mode when:**  
You want to reproduce probe placement relative to the participant *within the same session* (or within a setup where participant is already placed correctly).

---

## Mode 2 — Participant → baseline (absolute)

**Goal:** Reposition the Participant (cluster) to match the baseline session’s Participant pose in **global/world coordinates**.

### What it uses
- Baseline snapshot (selected via **Set baseline…**) provides:
  - `subject_pos_mm`, `subject_quat` at baseline
- Live tracking provides:
  - current `subject_pos_mm`, `subject_quat`

### What is computed (UI side)
- **Translation delta (mm):**
  \[
  \Delta p_{mm} = (p_{target} - p_{live})\cdot 1000
  \]
- **Translation error (mm):**
  \[
  \|\Delta p_{mm}\|
  \]
- **Rotation error (deg):**
  geodesic angle between participant’s live rotation and baseline rotation

### What you see
- The arrow/cone shows how to move the **participant** to match the baseline pose in the lab/world frame.

**Use this mode when:**  
You come back another day/session and need to relocate the **participant** to the previous session pose.

---

## Mode 3 — Grid → corrected target (absolute)

**Goal:** After the participant has been relocated, compute the “where should the grid be today” target, and navigate the grid to it in **global/world coordinates**.

This is your relocation workflow for probe replacement:

1) relocate participant (Mode 2)  
2) compute corrected grid target (button)  
3) navigate grid to corrected absolute target (Mode 3)

### What it uses
- Baseline snapshot provides baseline poses:
  - \(T_{S0}\) = Participant baseline pose
  - \(T_{G0}\) = Grid baseline pose
- Live provides current participant pose:
  - \(T_S\)

### What is computed (UI side) when you click **Compute grid target**
1) Compute grid relative-to-participant from baseline:
\[
R_{S\rightarrow G} = R_{S0}^T R_{G0}, \quad
t_{S\rightarrow G} = R_{S0}^T (p_{G0}-p_{S0})
\]
2) Rebuild the corrected grid target for today:
\[
R_{G,target} = R_S \, R_{S\rightarrow G}, \quad
p_{G,target} = p_S + R_S \, t_{S\rightarrow G}
\]

This “anchors” the grid placement to the participant, but expressed in world coordinates.

### What is computed live (UI side)
- Translation delta:
\[
\Delta p_{mm} = (p_{G,target} - p_{G,live})\cdot 1000
\]
- Rotation error:
\[
\theta = \text{geo}(R_{G,live}, R_{G,target})
\]

**Use this mode when:**  
You want to relocate the probe/grid back to its baseline placement **after** moving the participant back.

---

## Absolute Error Dialog (“Abs Error…”)

This dialog is for analysis and debugging.

### Snapshot vs Snapshot
- Pick two saved snapshots.
- It plots the two sets of absolute positions and shows how the grid and participant changed.

### Live vs Snapshot
- Compare current live absolute positions to a snapshot.

### What it shows
- A 3D plot with:
  - Participant(A), Grid(A)
  - Participant(B), Grid(B)
  - Lines grid↔participant and A→B movement vectors

> Tip: For Live vs Snapshot to work, your tracker must always emit `grid_pos_mm` and `subject_pos_mm` (not `None`) even when no target is saved.

---

## Recommended workflow (typical session-to-session relocation)

1) **Session 1 (baseline)**
   - Start tracker
   - Click **Report Δpos** to save a baseline snapshot (includes absolute world poses)

2) **Session 2 (relocation)**
   - Start tracker
   - Click **Set baseline…** and select the baseline snapshot
   - Switch to **Participant → baseline (absolute)** and relocate the participant until OK
   - Click **Compute grid target**
   - Switch to **Grid → corrected target (absolute)** and relocate the probe/grid until OK
