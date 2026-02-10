import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="STARBOW simulator(3D)", layout="wide")

BG = "#060a14"
PANEL_BG = "#060a14"
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
      [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background: {PANEL_BG};
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 物理・幾何ユーティリティ
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
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    denom = np.clip(1.0 - beta * nx, 1e-12, None)
    npx = (nx - beta) / denom
    npy = ny / (gamma * denom)
    npz = nz / (gamma * denom)
    out = np.stack([npx, npy, npz], axis=1)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out

def doppler_factor(n_prime: np.ndarray, beta: float) -> np.ndarray:
    if beta <= 0:
        return np.ones(len(n_prime))
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    return gamma * (1.0 + beta * n_prime[:, 0])

def wavelength_to_rgb(wl_nm: np.ndarray) -> np.ndarray:
    wl = wl_nm
    rgb = np.zeros((len(wl), 3))
    m = (wl >= 380) & (wl <= 780)

    rgb[m & (wl < 440), 0] = -(wl[m & (wl < 440)] - 440) / 60
    rgb[m & (wl < 440), 2] = 1
    rgb[m & (wl >= 440) & (wl < 490), 1] = (wl[m & (wl >= 440) & (wl < 490)] - 440) / 50
    rgb[m & (wl >= 440) & (wl < 490), 2] = 1
    rgb[m & (wl >= 490) & (wl < 510), 1] = 1
    rgb[m & (wl >= 490) & (wl < 510), 2] = -(wl[m & (wl >= 490) & (wl < 510)] - 510) / 20
    rgb[m & (wl >= 510) & (wl < 580), 0] = (wl[m & (wl >= 510) & (wl < 580)] - 510) / 70
    rgb[m & (wl >= 510) & (wl < 580), 1] = 1
    rgb[m & (wl >= 580) & (wl < 645), 0] = 1
    rgb[m & (wl >= 580) & (wl < 645), 1] = -(wl[m & (wl >= 580) & (wl < 645)] - 645) / 65
    rgb[m & (wl >= 645), 0] = 1

    return np.clip(rgb, 0, 1) ** 0.8

def rgb_to_hex(rgb):
    return [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in rgb]

# =========================
# UI + レイアウト（ここが重要）
# =========================
st.markdown("# **STARBOW simulator(3D)**")

colL, colR = st.columns([1.05, 2.2], gap="large")

# ---------- 左：UI ----------
with colL:
    st.markdown("## パラメータ")
    beta = st.slider("v/c", 0.0, 0.99, 0.00, 0.01)
    n_stars = st.slider("星の数", 1000, 20000, 10000, 500)
    seed = st.number_input("配置シード", value=12345)

    st.markdown("## 表示")
    star_size = st.slider("星の大きさ", 0.5, 6.0, 1.0, 0.1)
    glow = st.slider("グロー強さ", 0.0, 1.0, 0.0, 0.02)
    show_invisible_as_ring = st.toggle("不可視光を表示（白枠）", value=False)

# =========================
# データ生成
# =========================
dirs = random_unit_vectors(int(n_stars), int(seed))
dirs_ship = aberrate_directions(dirs, beta)
D = doppler_factor(dirs_ship, beta)
lam = 560.0 / D
visible = (lam >= 380) & (lam <= 780)
colors = rgb_to_hex(wavelength_to_rgb(lam))

# =========================
# 3D描画
# =========================
with colR:
    st.markdown(f"## v/c = {beta:.2f}（ドラッグで見回し）")

    fig = go.Figure()

    if glow > 0:
        fig.add_trace(go.Scatter3d(
            x=dirs_ship[visible,0], y=dirs_ship[visible,1], z=dirs_ship[visible,2],
            mode="markers",
            marker=dict(size=star_size*6*(1+glow*2), color=np.array(colors)[visible], opacity=0.15)
        ))

    fig.add_trace(go.Scatter3d(
        x=dirs_ship[visible,0], y=dirs_ship[visible,1], z=dirs_ship[visible,2],
        mode="markers",
        marker=dict(size=star_size*2.3, color=np.array(colors)[visible], opacity=0.95)
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=-0.25, y=0.25, z=0.25),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
            bgcolor=BG
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep"
    )

    st.plotly_chart(fig, use_container_width=True,
                    config=dict(scrollZoom=True, displaylogo=False))
