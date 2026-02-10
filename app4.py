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
      .stButton > button {{
        color: {TXT} !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        background: rgba(255,255,255,0.06) !important;
      }}
      .stButton > button:hover {{
        background: rgba(255,255,255,0.12) !important;
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

    nx = n[:, 0]
    ny = n[:, 1]
    nz = n[:, 2]

    denom = (1.0 - beta * nx)
    denom = np.clip(denom, 1e-12, None)

    npx = (nx - beta) / denom
    npy = ny / (gamma * denom)
    npz = nz / (gamma * denom)

    npv = np.stack([npx, npy, npz], axis=1)
    npv /= np.linalg.norm(npv, axis=1, keepdims=True)
    return npv

def doppler_factor(n_prime: np.ndarray, beta: float) -> np.ndarray:
    if beta <= 0:
        return np.ones(len(n_prime))
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    cos_th = n_prime[:, 0]
    return gamma * (1.0 + beta * cos_th)

def wavelength_to_rgb(wl_nm: np.ndarray) -> np.ndarray:
    wl = wl_nm
    rgb = np.zeros((len(wl), 3), dtype=float)

    m = (wl >= 380) & (wl < 440)
    rgb[m, 0] = -(wl[m] - 440) / (440 - 380)
    rgb[m, 2] = 1.0

    m = (wl >= 440) & (wl < 490)
    rgb[m, 1] = (wl[m] - 440) / (490 - 440)
    rgb[m, 2] = 1.0

    m = (wl >= 490) & (wl < 510)
    rgb[m, 1] = 1.0
    rgb[m, 2] = -(wl[m] - 510) / (510 - 490)

    m = (wl >= 510) & (wl < 580)
    rgb[m, 0] = (wl[m] - 510) / (580 - 510)
    rgb[m, 1] = 1.0

    m = (wl >= 580) & (wl < 645)
    rgb[m, 0] = 1.0
    rgb[m, 1] = -(wl[m] - 645) / (645 - 580)

    m = (wl >= 645) & (wl <= 780)
    rgb[m, 0] = 1.0

    rgb = np.clip(rgb, 0, 1) ** 0.8
    return rgb

def rgb_to_hex(rgb: np.ndarray) -> list[str]:
    rgb255 = (np.clip(rgb, 0, 1) * 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb255]

# =========================
# UI
# =========================
st.markdown("# **STARBOW simulator(3D)**")

colL, colR = st.columns([1.05, 2.2], gap="large")

with colL:
    st.markdown("## パラメータ")
    beta = st.slider("v/c", 0.0, 0.99, 0.00, 0.01)             # 初期0
    n_stars = st.slider("星の数", 1000, 20000, 10000, 500)      # 初期10000

    seed = st.number_input("配置シード（同じ値で同じ星配置）", value=12345, step=1)

    st.markdown("## 表示")
    star_size = st.slider("星の大きさ", 0.5, 6.0, 1.0, 0.1)
    glow = st.slider("グロー強さ", 0.0, 1.0, 0.0, 0.02)

    show_invisible_as_ring = st.toggle("不可視光を表示（白枠）", value=False)

with colR:
    st.markdown(f"## v/c = {beta:.2f} （ドラッグで見回し）")
    st.caption("※ドラッグで見回し・ホイール/ピンチでズームできます。")

# =========================
# データ生成
# =========================
base_dirs = random_unit_vectors(int(n_stars), int(seed))

base_lambda_nm = 560.0
dirs_ship = aberrate_directions(base_dirs, beta)
D = doppler_factor(dirs_ship, beta)
obs_lambda = base_lambda_nm / D

visible = (obs_lambda >= 380.0) & (obs_lambda <= 780.0)

colors_rgb = wavelength_to_rgb(obs_lambda)
colors = rgb_to_hex(colors_rgb)

# =========================
# 目印（前後左右上下 + 船）
# =========================
markers = np.array(
    [
        [ 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0,-1.0, 0.0],
        [ 0.0, 0.0, 1.0],
        [ 0.0, 0.0,-1.0],
    ],
    dtype=float
)
marker_text = ["+x(前)", "-x(後)", "+y(右)", "-y(左)", "+z(上)", "-z(下)"]

# =========================
# 3D描画（本体）
# =========================
def build_3d_fig() -> go.Figure:
    if show_invisible_as_ring:
        vis_mask = visible
        inv_mask = ~visible
    else:
        vis_mask = visible
        inv_mask = np.zeros_like(visible, dtype=bool)

    m3 = markers * 1.15

    fig = go.Figure()

    # glow
    if glow > 0 and np.any(vis_mask):
        fig.add_trace(
            go.Scatter3d(
                x=dirs_ship[vis_mask, 0],
                y=dirs_ship[vis_mask, 1],
                z=dirs_ship[vis_mask, 2],
                mode="markers",
                marker=dict(
                    size=float(star_size * 6.0 * (1.0 + glow * 2.0)),
                    color=np.array(colors)[vis_mask],
                    opacity=float(min(0.20, 0.06 + glow * 0.20)),
                ),
                hoverinfo="skip",
                name="glow",
            )
        )

    # visible
    if np.any(vis_mask):
        fig.add_trace(
            go.Scatter3d(
                x=dirs_ship[vis_mask, 0],
                y=dirs_ship[vis_mask, 1],
                z=dirs_ship[vis_mask, 2],
                mode="markers",
                marker=dict(
                    size=float(star_size * 2.3),
                    color=np.array(colors)[vis_mask],
                    opacity=0.95,
                ),
                hoverinfo="skip",
                name="visible",
            )
        )

    # invisible (white outline)
    if np.any(inv_mask):
        fig.add_trace(
            go.Scatter3d(
                x=dirs_ship[inv_mask, 0],
                y=dirs_ship[inv_mask, 1],
                z=dirs_ship[inv_mask, 2],
                mode="markers",
                marker=dict(
                    size=float(star_size * 2.5),
                    color="rgba(0,0,0,0)",
                    opacity=0.9,
                    line=dict(color="rgba(255,255,255,0.95)", width=2),
                ),
                hoverinfo="skip",
                name="invisible",
            )
        )

    # axis markers
    fig.add_trace(
        go.Scatter3d(
            x=m3[:, 0],
            y=m3[:, 1],
            z=m3[:, 2],
            mode="markers+text",
            marker=dict(size=7, color="white", opacity=0.95),
            text=marker_text,
            textposition="top center",
            textfont=dict(color="white", size=12),
            hoverinfo="skip",
            name="markers",
        )
    )

    # ship at origin
    fig.add_trace(
        go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers+text",
            marker=dict(size=6, color="white"),
            text=["ship"],
            textposition="bottom center",
            textfont=dict(color="white", size=12),
            hoverinfo="skip",
            name="ship",
        )
    )

    # 初期向き：+x が画面中心に来るように固定（位置は維持したいので距離感は同程度に）
    # eye は「原点からカメラの位置」。center=(0,0,0)を見る。
    # +x を正面にしたい → カメラを -x 側に置いて原点を見る。
    camera = dict(
        eye=dict(x=-0.25, y=0.25, z=0.25),
        center=dict(x=0.0, y=0.0, z=0.0),
        up=dict(x=0.0, y=0.0, z=1.0),
    )

    lim = 1.25

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=10, b=10),
        scene=dict(
            xaxis=dict(visible=False, range=[-lim, lim]),
            yaxis=dict(visible=False, range=[-lim, lim]),
            zaxis=dict(visible=False, range=[-lim, lim]),
            bgcolor=BG,
            aspectmode="cube",
            camera=camera,
        ),
        showlegend=False,
        uirevision="3d-keep-camera",
    )

    return fig

# =========================
# 描画
# =========================
fig = build_3d_fig()
st.plotly_chart(
    fig,
    use_container_width=True,
    config=dict(scrollZoom=True, displaylogo=False),
)

st.caption("※不可視光は「不可視光を表示（白枠）」OFF のとき完全に表示しません。ONで白枠として表示します。")
