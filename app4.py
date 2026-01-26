import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="STARBOW Simulator (Center View)", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #070a16; color: #ffffff; }
      h1,h2,h3,h4,h5,h6,p,span,div,label { color: #ffffff !important; }
      section[data-testid="stSidebar"] { background: #070a16; }
    </style>
    """,
    unsafe_allow_html=True,
)

VISIBLE_MIN = 380.0
VISIBLE_MAX = 780.0

def wavelength_to_rgb_nm(lam_nm: np.ndarray) -> np.ndarray:
    lam = lam_nm.copy()
    rgb = np.zeros((lam.size, 3), dtype=float)
    m = (lam >= 380) & (lam <= 780)
    x = lam[m]

    r = np.zeros_like(x); g = np.zeros_like(x); b = np.zeros_like(x)

    k = (x >= 380) & (x < 440)
    r[k] = -(x[k] - 440) / (440 - 380); b[k] = 1
    k = (x >= 440) & (x < 490)
    g[k] = (x[k] - 440) / (490 - 440); b[k] = 1
    k = (x >= 490) & (x < 510)
    g[k] = 1; b[k] = -(x[k] - 510) / (510 - 490)
    k = (x >= 510) & (x < 580)
    r[k] = (x[k] - 510) / (580 - 510); g[k] = 1
    k = (x >= 580) & (x < 645)
    r[k] = 1; g[k] = -(x[k] - 645) / (645 - 580)
    k = (x >= 645) & (x <= 780)
    r[k] = 1

    a = np.ones_like(x)
    k = (x >= 380) & (x < 420)
    a[k] = 0.3 + 0.7 * (x[k] - 380) / (420 - 380)
    k = (x > 700) & (x <= 780)
    a[k] = 0.3 + 0.7 * (780 - x[k]) / (780 - 700)

    gamma = 0.8
    r = (a * r) ** gamma
    g = (a * g) ** gamma
    b = (a * b) ** gamma

    rgb[m, 0] = r; rgb[m, 1] = g; rgb[m, 2] = b
    return rgb

def rot_from_yaw_pitch(yaw_deg: float, pitch_deg: float):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    fx = np.cos(yaw) * np.cos(pitch)
    fy = np.sin(yaw) * np.cos(pitch)
    fz = np.sin(pitch)
    f = np.array([fx, fy, fz], dtype=float)
    f = f / np.linalg.norm(f)

    upw = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(f, upw)) > 0.95:
        upw = np.array([0.0, 1.0, 0.0])

    r = np.cross(f, upw); r = r / np.linalg.norm(r)
    u = np.cross(r, f);  u = u / np.linalg.norm(u)
    return f, r, u

def aberrate_and_doppler_sources(s_xyz: np.ndarray, beta: float, lam0_nm: float):
    if beta <= 0:
        return s_xyz.copy(), np.full(s_xyz.shape[0], lam0_nm, dtype=float)

    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    s = s_xyz
    n = -s

    nx = n[:, 0]; ny = n[:, 1]; nz = n[:, 2]
    den = (1.0 - beta * nx)

    npx = (nx - beta) / den
    npy = ny / (gamma * den)
    npz = nz / (gamma * den)

    n_prime = np.stack([npx, npy, npz], axis=1)
    n_prime = n_prime / np.linalg.norm(n_prime, axis=1, keepdims=True)
    s_prime = -n_prime

    doppler = gamma * (1.0 + beta * s[:, 0])
    lam_obs = lam0_nm / doppler
    return s_prime, lam_obs

def project_azimuthal(s_cam: np.ndarray):
    x = s_cam[:, 0]; y = s_cam[:, 1]; z = s_cam[:, 2]
    front = z > 1e-9
    x = x[front]; y = y[front]; z = z[front]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    r = theta / (0.5 * np.pi)
    phi = np.arctan2(y, x)
    X = r * np.cos(phi)
    Y = r * np.sin(phi)
    return front, X, Y

def add_glow_traces(fig, X, Y, rgb, base_size, glow_strength):
    glow_size = base_size * (1.0 + 3.0 * glow_strength)
    glow_opacity = 0.10 + 0.25 * glow_strength

    fig.add_trace(
        go.Scatter(
            x=X, y=Y, mode="markers",
            marker=dict(size=glow_size, color=rgb, opacity=glow_opacity, line=dict(width=0)),
            hoverinfo="skip", showlegend=False
        )
    )

    core_opacity = 0.65 + 0.25 * glow_strength
    fig.add_trace(
        go.Scatter(
            x=X, y=Y, mode="markers",
            marker=dict(size=base_size, color=rgb, opacity=core_opacity, line=dict(width=0)),
            hoverinfo="skip", showlegend=False
        )
    )

st.title("STARBOW Simulator（宇宙船中心視点）")

colL, colR = st.columns([1.0, 3.2], gap="large")

with colL:
    st.subheader("パラメータ")
    beta = st.slider("v/c", 0.0, 0.99, 0.50, 0.01)
    nstars = st.slider("星の数", 200, 8000, 2500, 100)
    seed = st.number_input("配置シード（同じ値で同じ星配置）", min_value=0, max_value=10_000_000, value=12345, step=1)

    st.divider()
    st.subheader("視点（見る方向）")
    yaw = st.slider("ヨー（左右）", -180.0, 180.0, 0.0, 1.0)
    pitch = st.slider("ピッチ（上下）", -89.0, 89.0, 0.0, 1.0)

    st.divider()
    st.subheader("表示")
    zoom = st.slider("ズーム（大きいほど拡大）", 1.0, 6.0, 2.2, 0.1)
    base_size = st.slider("星の大きさ", 3, 14, 7, 1)
    glow = st.slider("グロー強さ", 0.0, 1.0, 0.65, 0.01)
    show_invisible_outline = st.toggle("不可視光を表示（白枠）", value=False)

rng = np.random.default_rng(int(seed))
u = rng.random(nstars); v = rng.random(nstars)
cos_theta = 2.0 * u - 1.0
phi = 2.0 * np.pi * v
sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
s_xyz = np.stack([sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta], axis=1)

lam0 = 550.0
s_prime, lam_obs = aberrate_and_doppler_sources(s_xyz, beta, lam0)

fwd, right, up = rot_from_yaw_pitch(yaw, pitch)
s_cam = np.stack([s_prime @ right, s_prime @ up, s_prime @ fwd], axis=1)

front_mask, X, Y = project_azimuthal(s_cam)
lam_front = lam_obs[front_mask]
vis = (lam_front >= VISIBLE_MIN) & (lam_front <= VISIBLE_MAX)

rgb_vis = wavelength_to_rgb_nm(lam_front[vis])
rgb_vis_str = [f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" for r, g, b in rgb_vis]

X_vis, Y_vis = X[vis], Y[vis]
X_inv, Y_inv = X[~vis], Y[~vis]

fig = go.Figure()

add_glow_traces(fig, X_vis, Y_vis, rgb_vis_str, base_size, glow)

if show_invisible_outline and X_inv.size > 0:
    fig.add_trace(
        go.Scatter(
            x=X_inv, y=Y_inv, mode="markers",
            marker=dict(
                size=base_size+2,
                color="rgba(0,0,0,0)",
                line=dict(color="rgba(255,255,255,0.65)", width=1.2),
                opacity=1.0,
            ),
            hoverinfo="skip", showlegend=False
        )
    )

# rings + cross
for deg in [30, 60, 90]:
    r = (np.deg2rad(deg)) / (0.5 * np.pi)
    fig.add_shape(
        type="circle", xref="x", yref="y",
        x0=-r, y0=-r, x1=r, y1=r,
        line=dict(
            color="rgba(255,255,255,0.35)" if deg < 90 else "rgba(255,255,255,0.55)",
            width=2.0 if deg == 90 else 1.4,
        ),
    )

fig.add_shape(type="line", x0=-1.02, y0=0, x1=1.02, y1=0,
              line=dict(color="rgba(120,170,255,0.55)", width=2))
fig.add_shape(type="line", x0=0, y0=-1.02, x1=0, y1=1.02,
              line=dict(color="rgba(120,170,255,0.55)", width=2))

# 方向マーカー（前/後/右/左/上/下）
dirs = {
    "forward": np.array([+1.0, 0.0, 0.0]),
    "back":    np.array([-1.0, 0.0, 0.0]),
    "right":   np.array([0.0, +1.0, 0.0]),
    "left":    np.array([0.0, -1.0, 0.0]),
    "up":      np.array([0.0, 0.0, +1.0]),
    "down":    np.array([0.0, 0.0, -1.0]),
}

marker_pts = []
for d in dirs.values():
    dc = np.array([np.dot(d, right), np.dot(d, up), np.dot(d, fwd)], dtype=float)
    if dc[2] <= 1e-9:
        continue
    th = np.arccos(np.clip(dc[2], -1, 1))
    rr = th / (0.5 * np.pi)
    ph = np.arctan2(dc[1], dc[0])
    marker_pts.append((rr*np.cos(ph), rr*np.sin(ph)))

if marker_pts:
    mx = [p[0] for p in marker_pts]
    my = [p[1] for p in marker_pts]
    fig.add_trace(
        go.Scatter(
            x=mx, y=my, mode="markers",
            marker=dict(size=8, color="rgba(255,255,255,0.85)"),
            hoverinfo="skip", showlegend=False
        )
    )

# 角度ラベル
fig.add_annotation(x=(np.deg2rad(30)/(0.5*np.pi))*1.02, y=0.02, text="30°",
                   showarrow=False, font=dict(color="white", size=28))
fig.add_annotation(x=(np.deg2rad(60)/(0.5*np.pi))*1.02, y=0.02, text="60°",
                   showarrow=False, font=dict(color="white", size=28))
fig.add_annotation(x=(np.deg2rad(90)/(0.5*np.pi))*1.02, y=0.02, text="90°",
                   showarrow=False, font=dict(color="white", size=28))

fig.update_layout(
    margin=dict(l=10, r=10, t=40, b=10),
    paper_bgcolor="#070a16",
    plot_bgcolor="#070a16",
    xaxis=dict(visible=False, range=[-1/zoom, 1/zoom]),
    yaxis=dict(visible=False, range=[-1/zoom, 1/zoom], scaleanchor="x", scaleratio=1),

    # ★これが「1本指ドラッグ＝見回し」にする核心
    dragmode="pan",

    # 速度変更でも視点維持
    uirevision="KEEP_VIEW",

    title=dict(
        text=f"v/c = {beta:.2f}",
        x=0.5, y=0.98,
        xanchor="center", yanchor="top",
        font=dict(color="white", size=36),
    ),
)

# ★操作系の設定（Box Selectを封印）
plotly_config = {
    "displayModeBar": False,
    "scrollZoom": True,  # トラックパッド/2本指スクロール等でズーム
    "modeBarButtonsToRemove": [
        "select2d", "lasso2d", "zoom2d", "autoScale2d", "resetScale2d"
    ],
}

with colR:
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)
