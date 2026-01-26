import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="STARBOW Simulator", layout="wide")

# ----------------------------
# UI
# ----------------------------
st.title("STARBOW Simulator")

col1, col2 = st.columns([1.1, 1.9], gap="large")
with col1:
    beta = st.slider("v/c", 0.0, 0.99, 0.0, 0.01)
    n_stars = st.slider("星の数", 50, 2000, 800, 50)
    seed = st.number_input("星の配置 Seed（同じ値なら同じ星空）", min_value=0, max_value=999999, value=1234, step=1)
    add_orion = st.checkbox("オリオン座（7点）を追加", True)

    # 追加：見た目調整（重くなければ上げてOK）
    beaming_p = st.slider("ビーミング強度 p（D^p）", 1.0, 3.0, 2.2, 0.1)
    size_gain = st.slider("星のサイズ倍率", 0.5, 2.5, 1.2, 0.1)

with col2:
    st.markdown(
        """
- 画面を **ドラッグ** すると、視点を自由に回して **正面以外の景色**を見られます  
- 速度を上げると  
  - **光行差**：進行方向へ星が集まる  
  - **ドップラー**：前方ほど青、後方ほど赤（可視外は暗く消える）  
  - **ビーミング**：前方が明るくなる  
        """
    )

# ----------------------------
# Style (C: night sky)
# ----------------------------
BG = "#0E0F2B"   # 夜空っぽい青みの黒
FG = "#FFFFFF"   # 白文字
GRID = "rgba(255,255,255,0.08)"  # 薄いグリッド

# ----------------------------
# Physics helpers
# ----------------------------
def wavelength_to_rgb_visible_only(wl_nm: float):
    """
    380-780nm の可視域のみRGB化（0..1）。可視外は None を返す（背景と同化させるため）。
    """
    w = float(wl_nm)
    if w < 380.0 or w > 780.0:
        return None

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

    # 端の暗さ補正
    if 380 <= w < 420:
        f = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif 420 <= w <= 700:
        f = 1.0
    else:  # 700..780
        f = 0.3 + 0.7 * (780 - w) / (780 - 700)

    return (r * f, g * f, b * f)

def doppler_factor_lab_to_ship(n_lab_unit: np.ndarray, beta: float) -> float:
    """
    nu' = nu * gamma*(1 + beta*n_x)  （ブースト +X）
    D = gamma*(1 + beta*n_x)
    """
    if beta == 0.0:
        return 1.0
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    nx = float(n_lab_unit[0])
    return gamma * (1.0 + beta * nx)

def aberrate_dir_lab_to_ship(n_lab_unit: np.ndarray, beta: float) -> np.ndarray:
    """
    光行差：lab方向n を ship方向n'へ（+Xブースト）
    """
    if beta == 0.0:
        return n_lab_unit.copy()

    nx, ny, nz = n_lab_unit
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    denom = (1.0 + beta * nx)

    nx_p = (nx + beta) / denom
    ny_p = ny / (gamma * denom)
    nz_p = nz / (gamma * denom)

    out = np.array([nx_p, ny_p, nz_p], dtype=float)
    out /= np.linalg.norm(out)
    return out

def random_unit_vectors(rng: np.random.Generator, m: int) -> np.ndarray:
    """
    等方的に球面上へ m 点（単位ベクトル）
    """
    u = rng.random(m)
    v = rng.random(m)
    theta = np.arccos(1 - 2*u)
    phi = 2*np.pi*v
    x = np.cos(theta)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(theta) * np.sin(phi)
    return np.stack([x, y, z], axis=1)

# ----------------------------
# Generate star catalog (lab frame)
# ----------------------------
rng = np.random.default_rng(int(seed))

# ランダム星：可視外も含む波長分布（リングが出やすい）
dirs_lab = random_unit_vectors(rng, n_stars)

wl0 = np.clip(rng.normal(650.0, 320.0, n_stars), 200.0, 1600.0)  # 紫外〜赤外まで
base = rng.random(n_stars)
brightness0 = 0.10 + 0.90 * (base**2)  # 0.10..1.0

# オリオン（教材用）
if add_orion:
    # 以前の「スライドっぽい」7点を前方半球に置く（便宜的）
    orion_uv = np.array([
        [0.58, 0.58],
        [0.78, 0.52],
        [0.68, 0.44],
        [0.66, 0.34],
        [0.72, 0.30],
        [0.78, 0.26],
        [0.84, 0.05],
    ], dtype=float)

    # azimuthal equidistant を逆にして方向ベクトルへ（前方 = +X）
    theta_max = np.pi/2
    def uv_to_dir(u, v):
        r = float(np.sqrt(u*u + v*v))
        r = np.clip(r, 0.0, 1.0)
        theta = r * theta_max
        phi = np.arctan2(v, u)
        nx = np.cos(theta)
        s = np.sin(theta)
        ny = s * np.cos(phi)
        nz = s * np.sin(phi)
        out = np.array([nx, ny, nz], dtype=float)
        return out / np.linalg.norm(out)

    orion_dirs = np.array([uv_to_dir(u, v) for u, v in orion_uv])
    orion_wl0 = np.array([560, 520, 600, 500, 560, 650, 480], dtype=float)
    orion_b = np.array([0.9, 0.7, 0.7, 0.6, 0.85, 0.8, 0.55], dtype=float)

    dirs_lab = np.vstack([orion_dirs, dirs_lab])
    wl0 = np.concatenate([orion_wl0, wl0])
    brightness0 = np.concatenate([orion_b, brightness0])

# ----------------------------
# Apply STARBOW: aberration + doppler + beaming
# ----------------------------
beta_f = float(beta)
gamma = 1.0 / np.sqrt(1.0 - beta_f*beta_f) if beta_f > 0 else 1.0

dirs_ship = np.empty_like(dirs_lab)
colors = []
sizes = []

for i in range(dirs_lab.shape[0]):
    nlab = dirs_lab[i]
    nlab = nlab / np.linalg.norm(nlab)

    # 光行差（方向）
    nship = aberrate_dir_lab_to_ship(nlab, beta_f)
    dirs_ship[i] = nship

    # ドップラー（色）
    D = doppler_factor_lab_to_ship(nlab, beta_f)
    wl_obs = wl0[i] / D   # クリップしない
    rgb = wavelength_to_rgb_visible_only(wl_obs)

    # 不可視 → 背景と同化（見えない）
    if rgb is None:
        colors.append(f"rgba(14,15,43,0.0)")  # BGと同化（透明）
        sizes.append(0.0)
        continue

    # ビーミング（見た目用に圧縮）
    I = brightness0[i] * (D ** float(beaming_p))
    I_disp = np.log1p(3.0 * I)          # 圧縮
    I_disp = float(np.clip(I_disp, 0.0, 3.0))

    # サイズ
    s = (4.0 + 10.0 * I_disp) * float(size_gain)
    s = float(np.clip(s, 1.0, 24.0))

    r, g, b = rgb
    # 明るさで少し増幅（ただし白潰れしにくい程度）
    boost = 0.6 + 0.25 * I_disp
    rr = min(1.0, r * boost)
    gg = min(1.0, g * boost)
    bb = min(1.0, b * boost)
    colors.append(f"rgba({int(rr*255)},{int(gg*255)},{int(bb*255)},1.0)")
    sizes.append(s)

sizes = np.array(sizes, dtype=float)

# 可視星だけ残す（速度改善＆不要な点を消す）
mask = sizes > 0
dirs_vis = dirs_ship[mask]
colors_vis = np.array(colors, dtype=object)[mask]
sizes_vis = sizes[mask]

# ----------------------------
# Plotly 3D (drag-to-look)
# ----------------------------
x, y, z = dirs_vis[:, 0], dirs_vis[:, 1], dirs_vis[:, 2]

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers",
    marker=dict(
        size=sizes_vis,
        color=colors_vis,
        opacity=1.0
    ),
    hoverinfo="skip"
))

# “天球”っぽさ：軸は消して、背景を夜空に
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False, showbackground=False),
        yaxis=dict(visible=False, showbackground=False),
        zaxis=dict(visible=False, showbackground=False),
        bgcolor=BG,
        aspectmode="cube",
        # 初期カメラ：+X方向（進行方向）を見る
        camera=dict(
            eye=dict(x=1.8, y=0.0, z=0.0),
            up=dict(x=0.0, y=0.0, z=1.0)
        )
    ),
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=FG, size=16),
    margin=dict(l=0, r=0, t=40, b=0),
    title=dict(text=f"v/c = {beta_f:.2f}　（ドラッグで見回し）", x=0.02, y=0.98, font=dict(color=FG, size=22)),
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar": True,
    "scrollZoom": True
})

st.caption("不可視（380–780nmの外）になった星は背景と同化して見えなくしています。Seed は NumPy の乱数生成器に渡しており、引用ではなく「再現性のための番号」です。")
