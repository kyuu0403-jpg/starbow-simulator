import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="STARBOW simulator (3D)", layout="wide")

BG = "#060a14"
TXT = "#ffffff"

st.markdown(
    f"""
    <style>
      .stApp {{
        background: radial-gradient(circle at 30% 20%, #0b1636 0%, {BG} 60%, #000000 100%);
        color: {TXT};
      }}
      h1, h2, h3, p, div, span, label {{
        color: {TXT} !important;
      }}
      .stButton > button {{
        color: {TXT} !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        background: rgba(255,255,255,0.06) !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 物理
# =========================
def random_unit_vectors(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def aberrate_directions(n: np.ndarray, beta: float) -> np.ndarray:
    if beta <= 0:
        return n.copy()
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    nx, ny, nz = n[:,0], n[:,1], n[:,2]
    denom = 1.0 - beta * nx
    denom = np.clip(denom, 1e-12, None)
    npx = (nx - beta) / denom
    npy = ny / (gamma * denom)
    npz = nz / (gamma * denom)
    v = np.stack([npx, npy, npz], axis=1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def doppler_factor(n_prime: np.ndarray, beta: float) -> np.ndarray:
    if beta <= 0:
        return np.ones(len(n_prime))
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    return gamma * (1.0 + beta * n_prime[:,0])

def wavelength_to_rgb(wl_nm: np.ndarray) -> np.ndarray:
    wl = wl_nm
    rgb = np.zeros((len(wl), 3), dtype=float)

    m = (wl >= 380) & (wl < 440)
    rgb[m,0] = -(wl[m]-440)/(440-380); rgb[m,2] = 1
    m = (wl >= 440) & (wl < 490)
    rgb[m,1] = (wl[m]-440)/(490-440); rgb[m,2] = 1
    m = (wl >= 490) & (wl < 510)
    rgb[m,1] = 1; rgb[m,2] = -(wl[m]-510)/(510-490)
    m = (wl >= 510) & (wl < 580)
    rgb[m,0] = (wl[m]-510)/(580-510); rgb[m,1] = 1
    m = (wl >= 580) & (wl < 645)
    rgb[m,0] = 1; rgb[m,1] = -(wl[m]-645)/(645-580)
    m = (wl >= 645) & (wl <= 780)
    rgb[m,0] = 1

    rgb = np.clip(rgb,0,1)**0.8
    return rgb

def rgb_to_hex(rgb):
    rgb255 = (np.clip(rgb,0,1)*255).astype(int)
    return [f"rgb({r},{g},{b})" for r,g,b in rgb255]

# =========================
# UI
# =========================
st.markdown("# **STARBOW simulator (3D)**")

colL, colR = st.columns([1, 3])

with colL:
    beta = st.slider("v/c", 0.0, 0.99, 0.0, 0.01)
    n_stars = st.slider("星の数", 1000, 20000, 10000, 500)
    seed = st.number_input("配置シード", value=12345, step=1)

    st.markdown("## 表示")
    star_size = st.slider("星の大きさ", 0.5, 6.0, 1.0, 0.1)
    glow = st.slider("グロー強さ", 0.0, 1.0, 0.0, 0.02)
    show_invisible = st.toggle("不可視光を表示（白枠）", value=False)

# =========================
# データ
# =========================
base_dirs = random_unit_vectors(int(n_stars), int(seed))
dirs = aberrate_directions(base_dirs, beta)
D = doppler_factor(dirs, beta)
obs_lambda = 560.0 / D
visible = (obs_lambda >= 380) & (obs_lambda <= 780)
colors = rgb_to_hex(wavelength_to_rgb(obs_lambda))

# 方向マーカー
markers = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
marker_text = ["+x(前)","-x(後)","+y(右)","-y(左)","+z(上)","-z(下)"]

# =========================
# 3D描画
# =========================
fig = go.Figure()

# グロー
if glow > 0 and np.any(visible):
    fig.add_trace(go.Scatter3d(
        x=dirs[visible,0], y=dirs[visible,1], z=dirs[visible,2],
        mode="markers",
        marker=dict(
            size=float(star_size*6*(1+glow*2)),
            color=np.array(colors)[visible],
            opacity=float(min(0.2, 0.06+glow*0.2))
        ),
        hoverinfo="skip"
    ))

# 可視星
fig.add_trace(go.Scatter3d(
    x=dirs[visible,0], y=dirs[visible,1], z=dirs[visible,2],
    mode="markers",
    marker=dict(size=float(star_size*2.3), color=np.array(colors)[visible], opacity=0.95),
    hoverinfo="skip"
))

# 不可視
if show_invisible:
    inv = ~visible
    fig.add_trace(go.Scatter3d(
        x=dirs[inv,0], y=dirs[inv,1], z=dirs[inv,2],
        mode="markers",
        marker=dict(size=float(star_size*2.5), color="rgba(0,0,0,0)",
                    line=dict(color="rgba(255,255,255,0.95)", width=2)),
        hoverinfo="skip"
    ))

# 方向マーカー
fig.add_trace(go.Scatter3d(
    x=markers[:,0], y=markers[:,1], z=markers[:,2],
    mode="markers+text",
    marker=dict(size=7, color="white"),
    text=marker_text,
    textposition="top center"
))

# ship
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode="markers+text",
    marker=dict(size=6, color="white"),
    text=["ship"],
    textposition="bottom center"
))

# カメラ：+xを正面
fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    scene=dict(
        bgcolor=BG,
        aspectmode="cube",
        camera=dict(
            eye=dict(x=1.6, y=0.0, z=0.0),
            center=dict(x=0,y=0,z=0),
            up=dict(x=0,y=0,z=1)
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True, config=dict(scrollZoom=True, displaylogo=False))

st.caption("※不可視光は「不可視光を表示（白枠）」OFF のとき完全に表示しません。ONで白枠として表示します。")
