import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="STARBOW Simulator", layout="wide")

# -----------------------------
# Theme + Button Fix
# -----------------------------
st.markdown(
    """
    <style>
      .stApp { background: #070a16; color: #ffffff; }
      h1,h2,h3,h4,h5,h6,p,span,div,label { color: #ffffff !important; }
      section[data-testid="stSidebar"] { background: #070a16; }

      div.stButton > button {
        background: rgba(35, 45, 85, 0.65) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
      }
      div.stButton > button:hover {
        background: rgba(60, 80, 140, 0.75) !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

VISIBLE_MIN = 380.0
VISIBLE_MAX = 780.0

# -----------------------------
# Utils
# -----------------------------
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

def add_glow_traces_2d(fig, X, Y, rgb, base_size, glow_strength):
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

# -----------------------------
# Defaults + Reset
# -----------------------------
DEFAULTS = dict(
    beta=0.50,
    nstars=2500,
    seed=12345,
    yaw=0.0,
    pitch=0.0,
    zoom=2.2,
    base_size=7,
    glow=0.65,
    show_invisible=False,
    interactive3d=True,
)

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# 初回だけカメラをセットするフラグ
st.session_state.setdefault("camera_initialized", False)

def do_reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["camera_initialized"] = False
    st.rerun()

# -----------------------------
# UI
# -----------------------------
st.title("STARBOW Simulator（宇宙船中心視点）")

colL, colR = st.columns([1.0, 3.2], gap="large")

with colL:
    st.subheader("パラメータ")
    st.session_state.beta = st.slider("v/c", 0.0, 0.99, float(st.session_state.beta), 0.01, key="beta_slider")
    st.session_state.nstars = st.slider("星の数", 200, 12000, int(st.session_state.nstars), 100, key="nstars_slider")
    st.session_state.seed = st.number_input("配置シード（同じ値で同じ星配置）", min_value=0, max_value=10_000_000, value=int(st.session_state.seed), step=1, key="seed_input")

    st.divider()
    st.session_state.interactive3d = st.toggle("ドラッグで見回し（3Dモード）", value=bool(st.session_state.interactive3d), key="interactive3d_toggle")

    if not st.session_state.interactive3d:
        st.subheader("視点（見る方向）")
        st.session_state.yaw = st.slider("ヨー（左右）", -180.0, 180.0, float(st.session_state.yaw), 1.0, key="yaw_slider")
        st.session_state.pitch = st.slider("ピッチ（上下）", -89.0, 89.0, float(st.session_state.pitch), 1.0, key="pitch_slider")

    st.divider()
    st.subheader("表示")
    st.session_state.zoom = st.slider("ズーム（大きいほど拡大）", 1.0, 6.0, float(st.session_state.zoom), 0.1, key="zoom_slider")
    st.session_state.base_size = st.slider("星の大きさ", 3, 14, int(st.session_state.base_size), 1, key="size_slider")
    st.session_state.glow = st.slider("グロー強さ", 0.0, 1.0, float(st.session_state.glow), 0.01, key="glow_slider")
    st.session_state.show_invisible = st.toggle("不可視光を表示（白枠）", value=bool(st.session_state.show_invisible), key="inv_toggle")

    st.divider()
    if st.button("初期化（パラメータ・視点・ズーム）", use_container_width=True):
        do_reset()

# -----------------------------
# Data generation (uniform sphere)
# -----------------------------
beta = float(st.session_state.beta)
nstars = int(st.session_state.nstars)
seed = int(st.session_state.seed)

rng = np.random.default_rng(seed)
u = rng.random(nstars); v = rng.random(nstars)
cos_theta = 2.0 * u - 1.0
phi = 2.0 * np.pi * v
sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
s_xyz = np.stack([sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta], axis=1)

lam0 = 550.0
s_prime, lam_obs = aberrate_and_doppler_sources(s_xyz, beta, lam0)

# -----------------------------
# Render
# -----------------------------
with colR:
    if st.session_state.interactive3d:
        rgb = wavelength_to_rgb_nm(lam_obs)
        rgb_str = [f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" for r, g, b in rgb]

        vis = (lam_obs >= VISIBLE_MIN) & (lam_obs <= VISIBLE_MAX)
        bg = "rgb(7,10,22)"
        colors = np.array(rgb_str, dtype=object)
        colors[~vis] = bg

        fig = go.Figure()

        # glow-like layers (3D)
        fig.add_trace(
            go.Scatter3d(
                x=s_prime[:, 0], y=s_prime[:, 1], z=s_prime[:, 2],
                mode="markers",
                marker=dict(
                    size=float(st.session_state.base_size) * (1.0 + 2.5*float(st.session_state.glow)),
                    color=colors,
                    opacity=0.16 + 0.22*float(st.session_state.glow),
                ),
                hoverinfo="skip", showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=s_prime[:, 0], y=s_prime[:, 1], z=s_prime[:, 2],
                mode="markers",
                marker=dict(
                    size=float(st.session_state.base_size),
                    color=colors,
                    opacity=0.60 + 0.25*float(st.session_state.glow),
                ),
                hoverinfo="skip", showlegend=False
            )
        )

        # Invisible outline (optional)
        if st.session_state.show_invisible:
            fig.add_trace(
                go.Scatter3d(
                    x=s_prime[~vis, 0], y=s_prime[~vis, 1], z=s_prime[~vis, 2],
                    mode="markers",
                    marker=dict(
                        size=float(st.session_state.base_size) + 1.5,
                        color="rgba(0,0,0,0)",
                        line=dict(color="rgba(255,255,255,0.55)", width=1.5),
                        opacity=1.0
                    ),
                    hoverinfo="skip", showlegend=False
                )
            )

        # Ship marker (origin)
        fig.add_trace(
            go.Scatter3d(
                x=[0.0], y=[0.0], z=[0.0],
                mode="markers",
                marker=dict(size=6, color="rgba(255,255,255,0.95)"),
                hoverinfo="skip",
                showlegend=False
            )
        )

        # Direction markers + labels
        dirs = {
            "+x": np.array([+1.0, 0.0, 0.0]),
            "-x": np.array([-1.0, 0.0, 0.0]),
            "+y": np.array([0.0, +1.0, 0.0]),
            "-y": np.array([0.0, -1.0, 0.0]),
            "+z": np.array([0.0, 0.0, +1.0]),
            "-z": np.array([0.0, 0.0, -1.0]),
        }
        mx = [d[0] for d in dirs.values()]
        my = [d[1] for d in dirs.values()]
        mz = [d[2] for d in dirs.values()]
        mt = list(dirs.keys())
        fig.add_trace(
            go.Scatter3d(
                x=mx, y=my, z=mz,
                mode="markers+text",
                marker=dict(size=5, color="rgba(255,255,255,0.90)"),
                text=mt,
                textposition="top center",
                textfont=dict(color="white", size=14),
                hoverinfo="skip",
                showlegend=False
            )
        )

        # Layout
        scene_dict = dict(
            bgcolor="#070a16",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
            # ★ scene側にも固定のuRevision（カメラ保持を強化）
            uirevision="KEEP_3D_SCENE",
        )

        layout_kwargs = dict(
            margin=dict(l=0, r=0, t=60, b=0),
            paper_bgcolor="#070a16",
            scene=scene_dict,
            title=dict(
                text=f"v/c = {beta:.2f}（ドラッグで見回し）",
                x=0.5, y=0.98,
                xanchor="center", yanchor="top",
                font=dict(color="white", size=32),
            ),
            # ★ 全体uRevisionも固定
            uirevision="KEEP_3D",
        )

        # 初回だけ：原点寄りの初期カメラ
        if not st.session_state["camera_initialized"]:
            layout_kwargs["scene"]["camera"] = dict(
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.6, y=0.0, z=0.25),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="perspective"),
            )
            st.session_state["camera_initialized"] = True

        fig.update_layout(**layout_kwargs)

        config = {
            "displayModeBar": False,
            "scrollZoom": True,
        }

        # ★ これが肝：key固定で「同じグラフ」として扱わせ、視点を維持させる
        st.plotly_chart(fig, use_container_width=True, config=config, key="STARBOw_3D")

        st.caption("※ 3Dモードはドラッグで見回しできます。パラメータを変えても視点が維持されるよう強化済み。")

    else:
        # 2D: ring view
        yaw = float(st.session_state.yaw)
        pitch = float(st.session_state.pitch)

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
        add_glow_traces_2d(fig, X_vis, Y_vis, rgb_vis_str, int(st.session_state.base_size), float(st.session_state.glow))

        if st.session_state.show_invisible and X_inv.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=X_inv, y=Y_inv, mode="markers",
                    marker=dict(
                        size=int(st.session_state.base_size)+2,
                        color="rgba(0,0,0,0)",
                        line=dict(color="rgba(255,255,255,0.65)", width=1.2),
                        opacity=1.0,
                    ),
                    hoverinfo="skip", showlegend=False
                )
            )

        fig.add_shape(type="line", x0=-1.02, y0=0, x1=1.02, y1=0,
                      line=dict(color="rgba(120,170,255,0.35)", width=2))
        fig.add_shape(type="line", x0=0, y0=-1.02, x1=0, y1=1.02,
                      line=dict(color="rgba(120,170,255,0.35)", width=2))

        z = float(st.session_state.zoom)
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#070a16",
            plot_bgcolor="#070a16",
            xaxis=dict(visible=False, range=[-1/z, 1/z]),
            yaxis=dict(visible=False, range=[-1/z, 1/z], scaleanchor="x", scaleratio=1),
            dragmode=False,
            uirevision="KEEP_2D",
            title=dict(
                text=f"v/c = {beta:.2f}",
                x=0.5, y=0.98,
                xanchor="center", yanchor="top",
                font=dict(color="white", size=34),
            ),
        )

        config = {"displayModeBar": False, "scrollZoom": True}

        # ★ 2D側もkey固定（モード切替の安定化）
        st.plotly_chart(fig, use_container_width=True, config=config, key="STARBOw_2D")

        st.caption("※ 2Dモードはリング優先。視点はヨー/ピッチで変更。")
