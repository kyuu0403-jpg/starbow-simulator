import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="STARBOW Simulator (Center View)", layout="wide")

# ----------------------------
# UI
# ----------------------------
st.title("STARBOW Simulator（中心視点）")

col1, col2 = st.columns([1.1, 1.9], gap="large")
with col1:
    beta = st.slider("v/c", 0.0, 0.99, 0.0, 0.01)
    n_stars = st.slider("星の数", 200, 20000, 4000, 200)
    seed = st.number_input("星の配置 Seed（同じ値なら同じ星空）", min_value=0, max_value=999999, value=1234, step=1)

    yaw_deg = st.slider("視線（Yaw：左右）[deg]", -180, 180, 0, 1)
    pitch_deg = st.slider("視線（Pitch：上下）[deg]", -89, 89, 0, 1)

    fov_deg = st.slider("視野角（FOV）[deg]", 60, 160, 120, 5)

    base_lambda = st.slider("基準波長 λ0 [nm]（色を統一）", 520, 650, 580, 1)  # 黄色寄りデフォ
    glow = st.slider("グロー（にじみ）", 0.0, 1.0, 0.35, 0.05)

with col2:
    st.markdown(
        """
- 視点は **宇宙船（中心）**。Yaw/Pitchで見回しできます  
- 速度を上げると  
  - **光行差**で進行方向側へ星が集まる  
  - **ドップラー効果**で中心は青寄り、外側は赤寄りになりやすい  
  - **可視外（380–780nm）**は暗くして見えなくします（リングが出やすい）
        """
    )

# ----------------------------
# Style
# ----------------------------
BG = (14/255, 15/255, 43/255)  # #0E0F2B
FG = (1.0, 1.0, 1.0)

# ----------------------------
# Math helpers
# ----------------------------
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

def random_unit_vectors(rng, m):
    # isotropic on sphere
    u = rng.random(m)
    v = rng.random(m)
    theta = np.arccos(1 - 2*u)
    phi = 2*np.pi*v
    x = np.cos(theta)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(theta) * np.sin(phi)
    return np.stack([x, y, z], axis=1)

def aberrate_dir_lab_to_ship(n_lab, beta):
    # boost +X
    if beta == 0.0:
        return n_lab.copy()
    nx, ny, nz = n_lab
    gamma = 1.0 / np.sqrt(1.0 - beta*beta)
    denom = (1.0 + beta * nx)
    nx_p = (nx + beta) / denom
    ny_p = ny / (gamma * denom)
    nz_p = nz / (gamma * denom)
    out = np.array([nx_p, ny_p, nz_p], dtype=float)
    return out / np.linalg.norm(out)

def doppler_factor(n_lab, beta):
    if beta == 0.0:
        return 1.0
    gamma = 1.0 / np.sqrt(1.0 - beta*beta)
    return gamma * (1.0 + beta * float(n_lab[0]))

def wavelength_to_rgb(wl_nm):
    # return None if invisible
    w = float(wl_nm)
    if w < 380.0 or w > 780.0:
        return None

    if 380 <= w < 440:
        r = -(w - 440) / (440 - 380); g = 0.0; b = 1.0
    elif 440 <= w < 490:
        r = 0.0; g = (w - 440) / (490 - 440); b = 1.0
    elif 490 <= w < 510:
        r = 0.0; g = 1.0; b = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        r = (w - 510) / (580 - 510); g = 1.0; b = 0.0
    elif 580 <= w < 645:
        r = 1.0; g = -(w - 645) / (645 - 580); b = 0.0
    else:
        r = 1.0; g = 0.0; b = 0.0

    # edge intensity correction
    if 380 <= w < 420:
        f = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif 420 <= w <= 700:
        f = 1.0
    else:
        f = 0.3 + 0.7 * (780 - w) / (780 - 700)

    return (r*f, g*f, b*f)

# ----------------------------
# Generate stars (LAB frame)
# ----------------------------
rng = np.random.default_rng(int(seed))
dirs_lab = random_unit_vectors(rng, int(n_stars))

# 全星の基準波長を統一（黄色付近）
wl0 = np.full(int(n_stars), float(base_lambda), dtype=float)

# ほんの少しだけ明るさバラつき（見やすさ）
brightness0 = 0.4 + 0.6 * (rng.random(int(n_stars))**2)

# ----------------------------
# Apply physics: aberration + doppler + beaming
# ----------------------------
beta_f = float(beta)
gamma = 1.0 / np.sqrt(1.0 - beta_f*beta_f) if beta_f > 0 else 1.0

dirs_ship = np.array([aberrate_dir_lab_to_ship(n, beta_f) for n in dirs_lab])

# view rotation (ship -> camera)
yaw = np.deg2rad(float(yaw_deg))
pitch = np.deg2rad(float(pitch_deg))
R = rot_z(yaw) @ rot_y(pitch)
dirs_cam = (R.T @ dirs_ship.T).T  # ship -> cam

# projection: pinhole with FOV
half = np.deg2rad(float(fov_deg) / 2.0)
tan_lim = np.tan(half)

xs, ys, cols, sizes = [], [], [], []

for i in range(int(n_stars)):
    nlab = dirs_lab[i]
    ncam = dirs_cam[i]

    # show only front hemisphere in camera: x>0 (forward)
    if ncam[0] <= 0:
        continue

    # screen coords
    x = (ncam[1] / ncam[0]) / tan_lim
    y = (ncam[2] / ncam[0]) / tan_lim
    if abs(x) > 1.0 or abs(y) > 1.0:
        continue

    D = doppler_factor(nlab, beta_f)
    wl_obs = wl0[i] / D  # no clip

    rgb = wavelength_to_rgb(wl_obs)
    if rgb is None:
        continue  # invisible -> background (not plotted)

    # beaming (visual)
    I = brightness0[i] * (D ** 2.2)
    I_disp = np.log1p(2.5 * I)
    I_disp = float(np.clip(I_disp, 0.0, 3.0))

    s = 4.0 + 10.0 * I_disp
    xs.append(x); ys.append(y)
    cols.append(rgb)
    sizes.append(s)

xs = np.array(xs); ys = np.array(ys)
cols = np.array(cols); sizes = np.array(sizes)

# ----------------------------
# Plot (night sky + glow)
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal", "box")
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# glow: draw a blurred layer first (bigger, transparent)
if glow > 0 and len(xs) > 0:
    ax.scatter(xs, ys, s=(sizes * (18 * glow)), c=cols, alpha=0.10 + 0.20*glow, edgecolors="none")

# main stars
ax.scatter(xs, ys, s=(sizes * 7.0), c=cols, alpha=0.95, edgecolors="none")

# title (white)
ax.set_title(f"v/c = {beta_f:.2f}", color=FG, fontsize=22, pad=14)

# thin white frame (like window)
ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, linewidth=1.2, edgecolor=FG, alpha=0.25))

st.pyplot(fig, clear_figure=True)

st.caption("不可視（380–780nm外）は描画せず背景と同化させています。星は全球で一様に配置（乱数）され、Seedで再現可能です。")

