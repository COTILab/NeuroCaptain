"""
schematic_2d.py  —  NeuroCaptain
2-D optode schematic using Blender's image pixel buffer.
No GPU module required.

This is v2 (the working dark-navy version) with three additions:
  1. collect_connections() reads mesh edges + optode_names_ordered  (was broken)
  2. Bitmap font for landmark labels
  3. Proper nose triangle + oval ears
"""

import math
import bpy

# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK LABEL ORDERS
# ─────────────────────────────────────────────────────────────────────────────

LABELS_1010 = [
    "Nz", "Iz", "Lpa", "Rpa", "Cz",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz",
    "FT7", "F7", "AF7", "Fp1", "TP7", "P7", "PO7", "O1",
    "FT8", "F8", "AF8", "Fp2", "TP8", "P8", "PO8", "O2",
    "FC1", "FC3", "FC5", "FC2", "FC4", "FC6",
    "F1", "F3", "F5", "F2", "F4", "F6", "AF3", "AF4",
    "CP1", "CP3", "CP5", "CP2", "CP4", "CP6",
    "P1", "P3", "P5", "P2", "P4", "P6", "PO3", "PO4",
    "FT9", "F9", "", "", "TP9", "P9", "PO9", "O9",
    "FT10", "F10", "", "", "TP10", "P10", "PO10", "O10",
]
LABELS_1020 = [
    "Nz", "Iz", "Lpa", "Rpa", "Cz",
    "T3", "C3", "Cz", "C4", "T4",
    "Fpz", "Fz", "Cz", "Pz", "Oz",
    "F7", "Fp1", "T5", "O1", "F8", "Fp2", "T6", "O2",
    "F3", "F4", "P3", "P4",
]

# Frozensets for tier filtering — include both old (T3/T5/T6) and new (T7/P7/P8) names
_SET_1020 = frozenset([
    "Nz","Iz","Lpa","Rpa","LPA","RPA",
    "Fpz","Fp1","Fp2",
    "F7","F3","Fz","F4","F8",
    "T3","T7","C3","Cz","C4","T4","T8",
    "T5","P7","P3","Pz","P4","T6","P8",
    "O1","Oz","O2",
])

_SET_1010 = frozenset(l for l in LABELS_1010 if l) | frozenset([
    "T3","T4","T5","T6",   # old-name aliases
])

_SKIP_LABELS = {""}   # show everything including Lpa/Rpa

IMAGE_NAME = "NeuroCaptain_Schematic"
IMG_W = IMG_H = 2800


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTION  — exactly v2, untouched
# ─────────────────────────────────────────────────────────────────────────────

def _azeq(t_deg, p_deg):
    """Azimuthal-equidistant: normalised (x, y) in [-1, 1].
    p_deg=0 → +Y (Nz) → ny positive → plots DOWNWARD in pixel space (cy + ny*R).
    So ny = +cos gives Nz at bottom. We want Nz at bottom since nose points down.
    +x = right ear (Rpa), -x = left ear (Lpa).
    r is clamped to 1.0 so all points stay inside the head circle."""
    r   = min(t_deg / 115.0, 1.0)   # clamp: colatitude > 115° maps to circle edge
    rad = math.radians(p_deg)
    return r * math.sin(rad), r * math.cos(rad)   # ny positive = toward Nz = downward in pixels


def cartesian_to_2d(pos):
    """
    3-D XYZ → normalised 2-D using azimuthal-equidistant projection.
    Confirmed axes: +Z=up/Cz, +Y=forward/Nz, +X=right ear (Rpa).
    p=0 → +Y (Nz) → ny=+r → plots at cy+r*R = lower part of image (nose side).
    p=90 → +X (Rpa/right) → nx=+r → plots right. Correct.
    p=270 → -X (Lpa/left) → nx=-r → plots left. Correct.
    """
    x, y, z = pos
    r = math.sqrt(x*x + y*y + z*z)
    if r < 1e-6:
        return 0.0, 0.0
    t = math.degrees(math.acos(max(-1.0, min(1.0, z / r))))
    # atan2(x, y): 0 at +Y (Nz), 90 at +X (right ear) — correct orientation
    p = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    return _azeq(t, p)


# ─────────────────────────────────────────────────────────────────────────────
# SCENE DATA
# ─────────────────────────────────────────────────────────────────────────────

def get_landmark_positions():
    """Read LandmarkMesh → {label: [x,y,z]}.

    Label source priority:
      1. mesh_obj["landmark_labels"]  — written by brain1020mesh during generation,
         exact match between vertex index and label name.
      2. Hardcoded LABELS_1010 / LABELS_1020 fallback (vertex-count heuristic).
    """
    # Find the landmark mesh object
    obj = bpy.data.objects.get("LandmarkMesh")
    if obj is None:
        # Prefer any object that has the stored labels property
        for o in bpy.data.objects:
            if o.type != 'MESH': continue
            if "landmark_labels" in o:
                obj = o; break
    if obj is None:
        # Last resort: size heuristic
        for o in bpy.data.objects:
            if o.type != 'MESH': continue
            n = len(o.data.vertices)
            if 27 <= n <= 500 and not o.name.startswith("Anchor_"):
                obj = o; break
    if obj is None:
        return {}

    # Priority 1: labels stored by brain1020mesh (correct order guaranteed)
    stored = list(obj.get("landmark_labels", []))
    if stored:
        labels = stored
    else:
        # Priority 2: hardcoded fallback based on vertex count
        n = len(obj.data.vertices)
        labels = LABELS_1010 if n >= 77 else LABELS_1020

    out = {}
    for i, v in enumerate(obj.data.vertices):
        if i >= len(labels):
            break
        lbl = labels[i]
        if lbl and lbl not in _SKIP_LABELS and lbl not in out:
            wp = obj.matrix_world @ v.co
            out[lbl] = [wp.x, wp.y, wp.z]
    return out


def collect_optodes():
    optodes = []
    for cname, otype in (("Sources","source"), ("Detectors","detector")):
        coll = bpy.data.collections.get(cname)
        if coll:
            for obj in coll.objects:
                p = obj.matrix_world.translation
                optodes.append({"name": obj.name, "type": otype,
                                 "position": [p.x, p.y, p.z]})
    return optodes


def collect_connections():
    """
    Read S-D pairs from Optode_Connections mesh edges.
    Uses optode_names_ordered custom property (confirmed working: 49 S-D pairs).
    """
    conn_obj = bpy.data.objects.get("Optode_Connections")
    if not conn_obj or conn_obj.type != 'MESH':
        return []
    names = list(conn_obj.get("optode_names_ordered", []))
    if not names:
        return []
    conns = []
    for edge in conn_obj.data.edges:
        i0, i1 = edge.vertices
        if i0 >= len(names) or i1 >= len(names): continue
        n0, n1 = names[i0], names[i1]
        if n0.startswith("Source_") != n1.startswith("Source_"):
            conns.append({"optode1": n0, "optode2": n1})
    return conns


def load_json(filepath):
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    optodes = [{"name": o["name"], "type": o["type"], "position": o["position"]}
               for o in data.get("optodes", [])]
    connections = [{"optode1": c["optode1"], "optode2": c["optode2"]}
                   for c in data.get("connections", [])
                   if c["optode1"].startswith("Source_") != c["optode2"].startswith("Source_")]
    lm_hints = {}
    for o in data.get("optodes", []):
        ai  = o.get("anchor_info") or {}
        nl  = (ai.get("nearest_landmark") or {})
        lm  = nl.get("landmark")
        off = nl.get("offset")
        pos = o.get("position")
        if lm and off and pos and lm not in lm_hints:
            lm_hints[lm] = [pos[i]-off[i] for i in range(3)]
    return optodes, connections, lm_hints


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE  — v2 original logic, untouched
# ─────────────────────────────────────────────────────────────────────────────

def compute_schematic(optodes, connections, lm_positions,
                      show_channels=True, max_sd_mm=60.0, tier="NONE"):
    optode_points = []
    for o in optodes:
        nx, ny = cartesian_to_2d(o["position"])
        optode_points.append({"name": o["name"], "type": o["type"],
                               "nx": nx, "ny": ny})

    channel_lines = []
    if show_channels:
        # Compute channels directly from all source×detector pairs using 3D
        # Euclidean distance (Blender world units, same scale as optode positions).
        # This is independent of whether spring-physics connections were created.
        sources   = [o for o in optodes if o["type"] == "source"]
        detectors = [o for o in optodes if o["type"] == "detector"]
        for src in sources:
            for det in detectors:
                sp, dp = src["position"], det["position"]
                d3d = math.sqrt((sp[0]-dp[0])**2 + (sp[1]-dp[1])**2 + (sp[2]-dp[2])**2)
                if d3d > max_sd_mm:
                    continue
                nx1, ny1 = cartesian_to_2d(sp)
                nx2, ny2 = cartesian_to_2d(dp)
                channel_lines.append({
                    "name1": src["name"], "name2": det["name"],
                    "nx1": nx1, "ny1": ny1, "nx2": nx2, "ny2": ny2,
                    "dist": d3d,
                })

    # Build the set of labels to show for the chosen tier.
    # tier="NONE"  → no landmarks at all
    # tier="1020"  → standard 10-20 electrodes only  (~21 + fiducials)
    # tier="1010"  → 10-10 system                    (~81 electrodes)
    # tier="105"   → show every label present in lm_positions (full 10-5)
    landmark_points = []
    if tier != "NONE":
        if tier == "1020":
            keep = _SET_1020
        elif tier == "1010":
            keep = _SET_1010
        else:  # "105" — show everything available
            keep = None

        for label, pos in lm_positions.items():
            if label in _SKIP_LABELS:
                continue
            if keep is not None and label not in keep:
                continue
            nx, ny = cartesian_to_2d(pos)
            landmark_points.append((label, nx, ny))

    return {"optode_points":   optode_points,
            "channel_lines":   channel_lines,
            "landmark_points": landmark_points}


# ─────────────────────────────────────────────────────────────────────────────
# PIXEL HELPERS  — v2 originals
# ─────────────────────────────────────────────────────────────────────────────

def _sp(buf, W, x, y, c):
    x, y = int(round(x)), int(round(y))
    if 0 <= x < W and 0 <= y < W:
        i = (y*W+x)*4
        r,g,b,a = c
        ea=buf[i+3]; na=a+ea*(1-a)
        if na>0:
            buf[i]  =(r*a+buf[i]  *ea*(1-a))/na
            buf[i+1]=(g*a+buf[i+1]*ea*(1-a))/na
            buf[i+2]=(b*a+buf[i+2]*ea*(1-a))/na
            buf[i+3]=na

def _fc(buf,W,cx,cy,r,c):
    ir=int(r)+1; r2=r*r
    for dy in range(-ir,ir+1):
        for dx in range(-ir,ir+1):
            if dx*dx+dy*dy<=r2: _sp(buf,W,cx+dx,cy+dy,c)

def _oc(buf,W,cx,cy,r,c,t=2):
    steps=max(360,int(2*math.pi*r*3))
    for i in range(steps):
        a=2*math.pi*i/steps
        for k in range(t): _sp(buf,W,cx+(r-k)*math.cos(a),cy+(r-k)*math.sin(a),c)

def _line(buf,W,x0,y0,x1,y1,c,t=2):
    x0,y0,x1,y1=int(round(x0)),int(round(y0)),int(round(x1)),int(round(y1))
    dx=abs(x1-x0);dy=abs(y1-y0)
    sx=1 if x0<x1 else -1;sy=1 if y0<y1 else -1
    err=dx-dy;h=t//2
    while True:
        for tx in range(-h,h+1):
            for ty in range(-h,h+1): _sp(buf,W,x0+tx,y0+ty,c)
        if x0==x1 and y0==y1: break
        e2=2*err
        if e2>-dy: err-=dy;x0+=sx
        if e2<dx:  err+=dx;y0+=sy

def _fe(buf,W,cx,cy,rx,ry,c):
    for dy in range(-ry-1,ry+2):
        for dx in range(-rx-1,rx+2):
            if (dx/max(rx,1))**2+(dy/max(ry,1))**2<=1.0:
                _sp(buf,W,cx+dx,cy+dy,c)

def _oe(buf,W,cx,cy,rx,ry,c,t=3):
    steps=max(180,int(2*math.pi*max(rx,ry)*3))
    for i in range(steps):
        a=2*math.pi*i/steps
        for k in range(t):
            s=1.0-k/(max(rx,ry)+1)
            _sp(buf,W,cx+rx*s*math.cos(a),cy+ry*s*math.sin(a),c)


# ── 5×7 bitmap font ──────────────────────────────────────────────────────────
_F={
    'A':[0b01110,0b10001,0b11111,0b10001,0b10001,0b01110,0],
    'B':[0b11110,0b10001,0b11110,0b10001,0b10001,0b11110,0],
    'C':[0b01110,0b10001,0b10000,0b10000,0b10001,0b01110,0],
    'D':[0b11100,0b10010,0b10001,0b10001,0b10010,0b11100,0],
    'E':[0b11111,0b10000,0b11110,0b10000,0b10000,0b11111,0],
    'F':[0b10000,0b10000,0b11110,0b10000,0b10000,0b11111,0],
    'G':[0b01111,0b10001,0b10011,0b10000,0b10001,0b01110,0],
    'H':[0b10001,0b10001,0b11111,0b10001,0b10001,0b10001,0],
    'I':[0b01110,0b00100,0b00100,0b00100,0b00100,0b01110,0],
    'J':[0b01110,0b10010,0b00010,0b00010,0b00010,0b00111,0],
    'K':[0b10001,0b10010,0b11100,0b10100,0b10010,0b10001,0],
    'L':[0b11111,0b10000,0b10000,0b10000,0b10000,0b10000,0],
    'M':[0b10001,0b11011,0b10101,0b10001,0b10001,0b10001,0],
    'N':[0b10001,0b10011,0b10101,0b11001,0b10001,0b10001,0],
    'O':[0b01110,0b10001,0b10001,0b10001,0b10001,0b01110,0],
    'P':[0b10000,0b10000,0b11110,0b10001,0b10001,0b11110,0],
    'Q':[0b01101,0b10011,0b10001,0b10001,0b10001,0b01110,0],
    'R':[0b10001,0b10010,0b11110,0b10001,0b10001,0b11110,0],
    'S':[0b01110,0b10001,0b00001,0b01110,0b10000,0b01111,0],
    'T':[0b00100,0b00100,0b00100,0b00100,0b00100,0b11111,0],
    'U':[0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0],
    'V':[0b00100,0b01010,0b10001,0b10001,0b10001,0b10001,0],
    'W':[0b10001,0b11011,0b10101,0b10001,0b10001,0b10001,0],
    'X':[0b10001,0b01010,0b00100,0b00100,0b01010,0b10001,0],
    'Y':[0b00100,0b00100,0b01010,0b10001,0b10001,0b10001,0],
    'Z':[0b11111,0b10000,0b01000,0b00100,0b00010,0b11111,0],
    '0':[0b01110,0b10011,0b10101,0b11001,0b10001,0b01110,0],
    '1':[0b01110,0b00100,0b00100,0b00100,0b01100,0b00100,0],
    '2':[0b11111,0b10000,0b01000,0b00110,0b00001,0b01110,0],
    '3':[0b01110,0b10001,0b00001,0b00110,0b00001,0b01110,0],
    '4':[0b00010,0b00010,0b11111,0b10010,0b01010,0b00110,0],
    '5':[0b01110,0b10001,0b00001,0b01111,0b10000,0b11111,0],
    '6':[0b01110,0b10001,0b10001,0b11110,0b10000,0b01110,0],
    '7':[0b01000,0b01000,0b00100,0b00010,0b00001,0b11111,0],
    '8':[0b01110,0b10001,0b10001,0b01110,0b10001,0b01110,0],
    '9':[0b01110,0b00001,0b01111,0b10001,0b10001,0b01110,0],
    'z':[0b11111,0b10000,0b01000,0b00100,0b00010,0b11111,0],
    'h':[0b10001,0b10001,0b11001,0b10110,0b10000,0b10000,0],
}

def _txt(buf,W,x,y,s,c,sc=1):
    """Draw text; x,y = top-left pixel."""
    cx=int(x)
    for ch in s:
        rows=_F.get(ch.upper()) or _F.get(ch)
        if not rows: cx+=4*sc; continue
        for ri,rb in enumerate(rows):
            py=y+ri*sc
            for col in range(5):
                if rb&(1<<(4-col)):
                    for sx_ in range(sc):
                        for sy_ in range(sc):
                            _sp(buf,W,cx+col*sc+sx_,py+sy_,c)
        cx+=6*sc


# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
BG        = (0.05, 0.10, 0.16, 1.0)   # dark navy — v2 original
DISK_FILL = (0.08, 0.14, 0.22, 1.0)
OUTLINE   = (0.24, 0.48, 0.68, 1.0)
DASH      = (0.14, 0.26, 0.40, 0.6)
LM_COL    = (0.55, 0.75, 0.85, 1.0)   # light blue landmark dots
LM_LABEL  = (0.75, 0.90, 1.00, 1.0)   # slightly brighter for text
CH_COL    = (0.80, 0.20, 0.95, 1.0)   # bright purple — must be visible on dark bg
CH_COL2   = (0.95, 0.50, 1.00, 0.7)
SRC_FILL  = (0.95, 0.25, 0.25, 1.0)   # red
SRC_OUT   = (1.00, 0.55, 0.55, 1.0)
DET_FILL  = (0.10, 0.75, 0.30, 1.0)   # green
DET_OUT   = (0.20, 0.90, 0.45, 1.0)   # lighter green
WHITE     = (1.0,  1.0,  1.0,  1.0)
BLACK     = (0.0,  0.0,  0.0,  1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DRAW  — v2 structure, + nose/ears/channels/labels
# ─────────────────────────────────────────────────────────────────────────────

def draw_schematic_to_image(data, show_lm_labels=False):
    W = H = IMG_W
    img = bpy.data.images.get(IMAGE_NAME)
    if img is None or img.size[0] != W:
        if img: bpy.data.images.remove(img)
        img = bpy.data.images.new(IMAGE_NAME, width=W, height=H, alpha=True)
    img.colorspace_settings.name = 'Non-Color'

    # v2: dark navy background
    buf = []
    for _ in range(W*H):
        buf += list(BG)

    cx, cy = W//2, H//2
    R = int(W * 0.44)   # v2 original radius

    # ── Head disk fill ────────────────────────────────────────
    _fc(buf,W,cx,cy,R,DISK_FILL)

    # ── Dashed crosshairs + equator ring (v2) ─────────────────
    for i in range(-R, R+1, 6):
        _sp(buf,W,cx+i,cy,DASH)
        _sp(buf,W,cx,cy+i,DASH)
    for deg in range(0,360,3):
        ex = cx + R*0.5*math.cos(math.radians(deg))
        ey = cy + R*0.5*math.sin(math.radians(deg))
        _sp(buf,W,ex,ey,DASH)

    # ── Ears — filled dark ovals with outline ─────────────────
    # In v2 pixel space: cy is the vertical midpoint, ears sit left/right
    erx,ery = int(R*0.07), int(R*0.13)
    _fe(buf,W,cx-R,cy,erx,ery,DISK_FILL)
    _fe(buf,W,cx+R,cy,erx,ery,DISK_FILL)
    _oe(buf,W,cx-R,cy,erx,ery,OUTLINE,t=8)
    _oe(buf,W,cx+R,cy,erx,ery,OUTLINE,t=8)

    # ── Head outline ─────────────────────────────────────────
    _oc(buf,W,cx,cy,R,  OUTLINE,t=8)
    _oc(buf,W,cx,cy,R-1,OUTLINE,t=4)

    # ── Nose triangle pointing DOWN in pixel space ────────────
    # v2 uses cy + ny*R  (plus), so +ny = DOWN in pixels.
    # Nz is at ny≈+0.8 which is near the BOTTOM of the image.
    # The nose should point toward Nz, i.e. downward in pixel space.
    tip_y  = cy + R + int(R*0.11)    # below circle
    base_y = cy + R - int(R*0.02)    # just inside circle bottom
    nhw    = int(R*0.07)
    for py in range(base_y, tip_y+1):
        if tip_y <= base_y: break
        t  = (py - base_y) / float(tip_y - base_y)
        hw = int(nhw * (1.0-t))
        for px in range(cx-hw, cx+hw+1):
            _sp(buf,W,px,py,OUTLINE)

    # ── Channels (drawn before optodes) ──────────────────────
    for ch in data["channel_lines"]:
        # v2 to_px: cx + nx*R,  cy + ny*R  (+ not -)
        x1 = cx + ch["nx1"]*R;  y1 = cy + ch["ny1"]*R
        x2 = cx + ch["nx2"]*R;  y2 = cy + ch["ny2"]*R
        _line(buf,W,x1,y1,x2,y2,CH_COL, t=8)
        _line(buf,W,x1,y1,x2,y2,CH_COL2,t=3)

    # ── Landmark dots + labels ────────────────────────────────
    for lbl,lnx,lny in data["landmark_points"]:
        lx = cx + lnx*R
        ly = cy + lny*R
        _fc(buf,W,lx,ly,19,WHITE)
        if show_lm_labels:
            sc = 10
            tw  = len(lbl)*6*sc
            tx  = int(lx) - tw//2
            ty  = int(ly) + 26
            for bx in range(tx-3, tx+tw+3):
                for by in range(ty-3, ty+7*sc+4):
                    _sp(buf,W,bx,by,(0.05,0.10,0.16,0.80))
            _txt(buf,W,tx,ty,lbl,WHITE,sc=sc)

    # ── Optodes ───────────────────────────────────────────────
    for pt in data["optode_points"]:
        px_ = cx + pt["nx"]*R
        py_ = cy + pt["ny"]*R
        if pt["type"] == "source":
            _fc(buf,W,px_,py_,27, SRC_FILL)
        else:
            _fc(buf,W,px_,py_,23, DET_FILL)

    # ── Legend ────────────────────────────────────────────────
    lx,ly = 24, 24
    _fc(buf,W,lx+20,ly+28,20,SRC_FILL)
    _txt(buf,W,lx+50,ly+7,"SOURCE",SRC_OUT,sc=7)
    ly+=70
    _fc(buf,W,lx+20,ly+28,20,DET_FILL)
    _txt(buf,W,lx+50,ly+7,"DETECTOR",DET_OUT,sc=7)
    if data["channel_lines"]:
        ly+=70
        _line(buf,W,lx,ly+28,lx+40,ly+28,CH_COL,t=7)
        _txt(buf,W,lx+50,ly+7,f"CHANNEL {len(data['channel_lines'])}",CH_COL,sc=7)

    img.pixels.foreach_set(buf)
    img.update()
    return img


# ─────────────────────────────────────────────────────────────────────────────
# REBUILD
# ─────────────────────────────────────────────────────────────────────────────

def _rebuild(context, optodes=None, connections=None, lm_override=None):
    nc = context.scene.neurocaptain_settings
    if optodes     is None: optodes     = collect_optodes()
    if connections is None: connections = collect_connections()
    if not optodes:
        return None, "No optodes found in Sources/Detectors collections"

    lm_positions = lm_override or get_landmark_positions()
    if not lm_positions:
        import ast
        try:    lm_positions = ast.literal_eval(context.scene.get('schematic_json_landmarks','{}'))
        except: lm_positions = {}

    data = compute_schematic(
        optodes, connections, lm_positions,
        show_channels = nc.schematic_show_channels,
        max_sd_mm     = nc.sd_max_distance,
        tier          = nc.schematic_landmark_tier,
    )
    img = draw_schematic_to_image(data, show_lm_labels=nc.schematic_show_lm_labels)
    for area in context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            area.spaces.active.image = img
    return img, None


# ─────────────────────────────────────────────────────────────────────────────
# OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_image_editor(context, img):
    if not any(a.type == 'IMAGE_EDITOR' for a in context.screen.areas):
        largest = max(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'),
            key=lambda a: a.width*a.height, default=None)
        if largest:
            with context.temp_override(area=largest):
                bpy.ops.screen.area_split(direction='VERTICAL', factor=0.38)
            context.screen.areas[-1].type = 'IMAGE_EDITOR'
    for area in context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            area.spaces.active.image = img


class NEUROCAPTAIN_OT_open_schematic(bpy.types.Operator):
    """Generate 2-D schematic from scene optodes, or import JSON if scene is empty"""
    bl_idname  = "neurocaptain.open_2d_schematic"
    bl_label   = "Open 2D Schematic"
    bl_options = {'REGISTER'}

    filepath:    bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    @classmethod
    def description(cls, context, properties):
        if collect_optodes():
            return "Generate a 2D schematic from the optodes already in the scene"
        return "Import a probe JSON file and generate a 2D schematic from it"

    def invoke(self, context, event):
        if collect_optodes():
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        optodes     = collect_optodes()
        connections = collect_connections()
        lm_override = None

        if not optodes:
            if not self.filepath:
                self.report({'WARNING'}, "No optodes in scene and no file selected")
                return {'CANCELLED'}
            try:
                optodes, connections, lm_hints = load_json(self.filepath)
            except Exception as e:
                self.report({'ERROR'}, f"Could not read JSON: {e}")
                return {'CANCELLED'}
            if not optodes:
                self.report({'WARNING'}, "JSON contained no optodes")
                return {'CANCELLED'}
            context.scene['schematic_json_optodes']     = str(optodes)
            context.scene['schematic_json_connections'] = str(connections)
            context.scene['schematic_json_landmarks']   = str(lm_hints)
            lm_override = lm_hints

        img, err = _rebuild(context, optodes, connections, lm_override)
        if err:
            self.report({'WARNING'}, err)
            return {'CANCELLED'}
        _ensure_image_editor(context, img)
        self.report({'INFO'}, f"Schematic: {len(optodes)} optodes, "
                              f"{len(connections)} S-D channels")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_refresh_schematic(bpy.types.Operator):
    """Refresh schematic with current optode positions and settings"""
    bl_idname  = "neurocaptain.refresh_2d_schematic"
    bl_label   = "Refresh"
    bl_options = {'REGISTER'}

    def execute(self, context):
        optodes     = collect_optodes()
        connections = collect_connections()
        if not optodes:
            import ast
            try:
                optodes     = ast.literal_eval(context.scene.get('schematic_json_optodes',  '[]'))
                connections = ast.literal_eval(context.scene.get('schematic_json_connections','[]'))
            except Exception:
                pass
        img, err = _rebuild(context, optodes or None, connections or None)
        if err:
            self.report({'WARNING'}, err)
            return {'CANCELLED'}
        self.report({'INFO'}, "Schematic refreshed")
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = [NEUROCAPTAIN_OT_open_schematic, NEUROCAPTAIN_OT_refresh_schematic]

def register():
    for cls in CLASSES: bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(CLASSES): bpy.utils.unregister_class(cls)