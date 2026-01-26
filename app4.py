import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =========================
# STARBOW (Aberration + Doppler + Beaming) with Look Direction
# Ship frame: +X forward, +Y right, +Z up
# View: azimuthal-equidistant projection around current look axis (90° radius)
# =========================

st.set_page_config(page_title="STARBOW (Look Around)", layout="wide")

# -------- UI --------
col1, col2 = st.columns([1.2, 1.8], gap="large")
with col1:
    beta = st.slider("v/c", 0.0, 0.99, 0.0, 0.01)
    n_stars = st.slider("星の数", 7, 800, 120, 1)  # 7〜800（最初は軽め）
    yaw_deg = st.slider("視線の向き（Yaw：左右）[deg]", -180, 180, 0, 1)
    pitch_deg = st.slider("視線の向き（Pitch：上下）[deg]", -80, 80, 0, 1)
    show_orion = st.checkbox("オリオン座（固定7点）を含める", True)
    seed = st.number_input("星の配置 Seed（同じ数なら同じ星空にする）", min_value=0, max_value=999999, value=1234, step=1)

with col2:
    st.markdown(
        """
- **Yaw/Pitch** を動かすと、正面以外の方向の星空が見られます  
- 速度を上げると **光行差で前方へ集まり**、同時に **ドップラーで色が変化**、**ビーミングで前方が明るく**なります
        """
    )

# -------- constants / helpers --------
VISIBLE_THETA_MAX = np.pi / 2  # 90 deg

def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ ca, 0.0, sa],
                     [0.0, 1.0, 0.0],
                     [-sa, 0.0, ca]], dtype=float)

def dir_to_uv(n):
    """
    Azimuthal equidistant around +X:
      theta = arccos(nx) in [0..pi/2]
      r = theta/(pi/2)
      phi = atan2(nz, ny)  (note: y-right, z-up)
      u = r*cos(phi), v = r*sin(phi)
    """
    n = n / np.linalg.norm(n)
    nx, ny, nz = n
    nx = np.clip(nx, -1.0, 1.0)
    theta = np.arccos(nx)

    if theta > VISIBLE_THETA_MAX:
        return None  # outside 90° view

    r = theta / VISIBLE_THETA_MAX
    phi = np.arctan2(nz, ny)
    u = r * np.cos(phi)
    v = r * np.sin(phi)
    return float(u), float(v), float(theta), float(nx)

def aberrate_dir(n_lab, beta):
    """
    Relativistic aberration from lab -> ship, boost along +X.
    """
    n = n_lab / np.linalg.norm(n_lab)
    nx, ny, nz = n
    if beta == 0.0:
        return n.copy()

    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    denom = (1.0 + beta * nx)

    nx_p = (nx + beta) / denom
    ny_p = ny / (gamma * denom)
    nz_p = nz / (gamma * denom)

    out = np.array([nx_p, ny_p, nz_p], dtype=float)
    return out / np.linalg.norm(out)

def wavelength_to_rgb(wl_nm):
    """
    Rough visible-spectrum to RGB (380-780nm). Output in 0..1.
    """
    w = float(wl_nm)
    w = max(380.0, min(780.0, w))

    if 380 <= w < 440:
        r = -(w - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= w < 490:
        r = 0.0
        g = (w - 440) / (490 - 440)
        b = 1.0
    elif 490 <= w < 510:
        r = 0.0
        g = 1.0
        b = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        r = (w - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= w < 645:
        r = 1.0
        g = -(w - 645) / (645 - 580)
        b = 0.0
    else:  # 645..780
        r = 1.0
        g = 0.0
        b = 0.0

    # intensity correction at extremes (simple)
    if 380 <= w < 420:
        f = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif 420 <= w <= 700:
        f = 1.0
    else:  # 700..780
        f = 0.3 + 0.7 * (780 - w) / (780 - 700)

    return (r * f, g * f, b * f)

def doppler_factor_lab_to_ship(n_lab, beta):
    """
    Frequency transform for photon from lab direction n_lab (unit).
    For boost +X: nu' = nu * gamma*(1 + beta*nx)
    => Doppler factor D = gamma*(1 + beta*nx)
    """
    n = n_lab / np.linalg.norm(n_lab)
    nx = float(n[0])
    if beta == 0.0:
        return 1.0
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    return gamma * (1.0 + beta * nx)

# -------- Create star catalog (lab frame) --------
# We generate a fixed "max catalog" and slice by n_stars
MAX_STARS = 800
rng = np.random.default_rng(int(seed))

def random_unit_vectors(m):
    # isotropic on sphere
    u = rng.random(m)
    v = rng.random(m)
    theta = np.arccos(1 - 2*u)     # 0..pi
    phi = 2*np.pi*v               # 0..2pi
    x = np.cos(theta)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(theta) * np.sin(phi)
    return np.stack([x, y, z], axis=1)

# random stars
lab_dirs_rand = random_unit_vectors(MAX_STARS)

# base wavelengths & brightness (in lab)
# wavelength: biased toward ~550nm a bit (looks nicer), but still broad
wl0 = np.clip(rng.normal(560.0, 90.0, MAX_STARS), 380.0, 780.0)
# brightness: log-ish distribution
mag = rng.random(MAX_STARS)
base_brightness = 0.15 + 0.85 * (mag**2)  # 0.15..1.0

# Orion (screen-fixed look for beta=0 in your style)
# We'll interpret these as SCREEN points at beta=0, look forward.
# Convert to ship directions at beta=0, then treat those as LAB too at beta=0.
# (This is a "teaching constellation" not astronomical realism.)
orion_uv = np.array([
    [0.58, 0.58],
    [0.78, 0.52],
    [0.68, 0.44],
    [0.66, 0.34],
    [0.72, 0.30],
    [0.78, 0.26],
    [0.84, 0.05],
], dtype=float)

def uv_to_dir(u, v):
    r = float(np.sqrt(u*u + v*v))
    r = np.clip(r, 0.0, 1.0)
    theta = r * VISIBLE_THETA_MAX
    phi = np.arctan2(v, u)
    nx = np.cos(theta)
    s = np.sin(theta)
    ny = s * np.cos(phi)
    nz = s * np.sin(phi)
    out = np.array([nx, ny, nz], dtype=float)
    return out / np.linalg.norm(out)

orion_dirs_ship0 = np.array([uv_to_dir(u, v) for u, v in orion_uv])
# treat as lab directions too (teaching model)
orion_dirs_lab = orion_dirs_ship0.copy()
orion_wl0 = np.array([560, 520, 600, 500, 560, 650, 480], dtype=float)
orion_b = np.array([0.9, 0.7, 0.7, 0.6, 0.85, 0.8, 0.55], dtype=float)

# Combine catalog
dirs_lab = []
wl0_all = []
b_all = []

if show_orion:
    dirs_lab.append(orion_dirs_lab)
    wl0_all.append(orion_wl0)
    b_all.append(orion_b)

# slice random stars to fill up to n_stars
n_needed = n_stars - (7 if show_orion else 0)
if n_needed > 0:
    dirs_lab.append(lab_dirs_rand[:n_needed])
    wl0_all.append(wl0[:n_needed])
    b_all.append(base_brightness[:n_needed])

dirs_lab = np.vstack(dirs_lab) if len(dirs_lab) else np.zeros((0, 3))
wl0_all = np.concatenate(wl0_all) if len(wl0_all) else np.zeros((0,))
b_all = np.concatenate(b_all) if len(b_all) else np.zeros((0,))

# -------- Apply aberration (lab -> ship) --------
dirs_ship = np.array([aberrate_dir(n, beta) for n in dirs_lab])

# -------- Apply look direction (rotate ship -> camera) --------
yaw = np.deg2rad(float(yaw_deg))
pitch = np.deg2rad(float(pitch_deg))

# Camera rotation: from camera coords to ship coords
R = rot_z(yaw) @ rot_y(pitch)
# Ship -> camera
Rt = R.T
dirs_cam = (Rt @ dirs_ship.T).T

# -------- Project & colorize --------
uv_list = []
colors = []
sizes = []

for i, nlab in enumerate(dirs_lab):
    ncam = dirs_cam[i]

    proj = dir_to_uv(ncam)
    if proj is None:
        continue

    u, v, theta, nx_cam = proj

    # Doppler factor based on LAB direction relative to velocity (+X in lab)
    D = doppler_factor_lab_to_ship(nlab, beta)
    # observed wavelength
    wl_obs = wl0_all[i] / D
    wl_obs = float(np.clip(wl_obs, 380.0, 780.0))

    # Beaming (simple): intensity ~ D^3
    I = b_all[i] * (D ** 3)

    # Clamp for nice visuals
    I = float(np.clip(I, 0.02, 6.0))

    rgb = wavelength_to_rgb(wl_obs)
    colors.append(rgb)

    # Size: base + beaming + slightly larger near forward
    s = 40 + 180 * np.sqrt(I)
    sizes.append(float(np.clip(s, 30, 380)))

    uv_list.append((u, v))

uv = np.array(uv_list, dtype=float) if uv_list else np.zeros((0, 2))
colors = np.array(colors, dtype=float) if colors else np.zeros((0, 3))
sizes = np.array(sizes, dtype=float) if sizes else np.zeros((0,))

# -------- Plot --------
fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=160)
ax.set_aspect("equal")

# Rings (30/60/90 deg)
for r, lw in [(1.0, 2.8), (2/3, 1.8), (1/3, 1.3)]:
    ax.add_patch(plt.Circle((0, 0), r, fill=False, linewidth=lw, color="black"))

# Crosshair
ax.axhline(0, color="#1f77b4", linewidth=1.6)
ax.axvline(0, color="#1f77b4", linewidth=1.6)

# Degree labels on +u axis
ax.text(1/3 + 0.02, 0.02, "30°", fontsize=22, weight="bold")
ax.text(2/3 + 0.02, 0.02, "60°", fontsize=22, weight="bold")
ax.text(1.00 + 0.02, 0.02, "90°", fontsize=22, weight="bold")

# Stars
if len(uv) > 0:
    ax.scatter(uv[:, 0], uv[:, 1], s=sizes, c=colors, edgecolor="none", alpha=0.95, zorder=5)

# Cosmetics
ax.set_xlim(-1.08, 1.12)
ax.set_ylim(-1.08, 1.08)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# Top label: v/c only（+ 視線情報は出したくなければ消せる）
ax.set_title(f"v/c = {beta:.2f}", fontsize=26, pad=14)

st.pyplot(fig, clear_figure=True)
