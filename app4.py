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
    npv = np.stack([npx, npy, npz], axis=1)
    return npv / np.linalg.norm(npv, axis=1, keepdims=True)

def doppler_factor(n_prime: np.ndarray, beta: float) -> np.ndarray:
    if beta <= 0:
        return np.ones(len(n_prime))
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    return gamma * (1.0 + beta * n_prime[:, 0])

def wavelength_to_rgb(wl_nm: np.ndarray) -> np.ndarray:
    wl = wl_nm
    rgb = np.zeros((len(wl), 3))
    m = (wl >= 380) & (wl <= 780)
    x = wl[m]

    r = np.zeros_like(x)
    g = np.zeros_like(x)
    b = np.zeros_like(x)

    r[(x >= 510) & (x < 580)] = (x[(x >= 510) & (x < 580)] - 510) / 70
    g[(x >= 440) & (x < 490)] = (x[(x >= 440) & (x < 490)] - 440) / 50
    b[(x >= 380) & (x < 440)] = 1

    rgb[m] = np.clip(np.stack([r, g, b], axis=1), 0, 1) ** 0.8
    return rgb

def rgb_to_hex(rgb):
    return [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r,g,b in rgb]

# =========================
# UI（1回だけ）
# =========================
st.markdown("# **STARBOW simulator(3D)**")

colL, colR = st.columns([1.1, 2.3], gap="large")

with colL:
    st.subheader("パラメータ")
    beta = st.slider("v/c", 0.0, 0.99, 0.0, 0.01)
    n_stars = st.slider("星の数", 1000, 20000, 10000, 500)
    seed = st.number_input("配置シード", value=12345, step=1)

    st.subheader("表示")
    star_size = st.slider("星の大きさ", 0.5, 6.0, 1.0, 0.1)
    glow = st.slider("グロー強さ", 0.0, 1.0, 0.0, 0.02)
    show_invisible_as_ring = st.toggle("不可視光を表示（白枠）", False)

with colR:
    st.markdown(f"## v/c = {beta:.2f}（ドラッグで見回し）")

# =========================
# データ生成
# =========================
dirs0 = random_unit_vectors(n_stars, seed)
dirs = aberrate_directions(dirs0, beta)
D = doppler_factor(dirs, beta)
lam = 560 / D
visible = (lam >= 380) & (lam <= 780)
colors = rgb_to_hex(wavelength_to_rgb(lam))

# =========================
# マーカー
# =========================
markers = np.array([
    [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
])
marker_text = ["+x(前)", "-x(後)", "+y(右)", "-y(左)", "+z(上)", "-z(下)"]

# =========================
# 3D描画
# =========================
def build_3d_fig():
    fig = go.Figure()

    if glow > 0:
        fig.add_trace(go.Scatter3d(
            x=dirs[visible,0], y=dirs[visible,1], z=dirs[visible,2],
            mode="markers",
            marker=dict(size=star_size*6, color=np.array(colors)[visible], opacity=0.15),
            hoverinfo="skip"
        ))

    fig.add_trace(go.Scatter3d(
        x=dirs[visible,0], y=dirs[visible,1], z=dirs[visible,2],
        mode="markers",
        marker=dict(size=star_size*2.3, color=np.array(colors)[visible]),
        hoverinfo="skip"
    ))

    if show_invisible_as_ring:
        fig.add_trace(go.Scatter3d(
            x=dirs[~visible,0], y=dirs[~visible,1], z=dirs[~visible,2],
            mode="markers",
            marker=dict(size=star_size*2.5, color="rgba(0,0,0,0)",
                        line=dict(color="white", width=2)),
            hoverinfo="skip"
        ))

    fig.add_trace(go.Scatter3d(
        x=markers[:,0]*1.15, y=markers[:,1]*1.15, z=markers[:,2]*1.15,
        mode="markers+text",
        marker=dict(size=7, color="white"),
        text=marker_text,
        textposition="top center"
    ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=6, color="white"),
        text=["ship"],
        textposition="bottom center"
    ))

    fig.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=-0.25, y=0.25, z=0.25),
                center=dict(x=0,y=0,z=0),
                up=dict(x=0,y=0,z=1)
            ),
            bgcolor=BG
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor=BG,
        showlegend=False,
        uirevision="keep"
    )
    return fig

with colR:
    st.plotly_chart(build_3d_fig(), use_container_width=True,
                    config=dict(scrollZoom=True, displaylogo=False))
