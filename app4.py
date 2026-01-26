import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ----------------------------
# ページ設定 + ダークUI（文字白）
# ----------------------------
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

# ----------------------------
# 物理・表示ユーティリティ
# ----------------------------
VISIBLE_MIN = 380.0
VISIBLE_MAX = 780.0

def wavelength_to_rgb_nm(lam_nm: np.ndarray) -> np.ndarray:
    """
    380–780nm をざっくり sRGB に変換（0–1）。
    参考：一般的な近似式（可視域外は黒）
    """
    lam = lam_nm.copy()

    rgb = np.zeros((lam.size, 3), dtype=float)

    # 可視域だけ処理
    m = (lam >= 380) & (lam <= 780)
    x = lam[m]

    r = np.zeros_like(x)
    g = np.zeros_like(x)
    b = np.zeros_like(x)

    # 380-440
    k = (x >= 380) & (x < 440)
    r[k] = -(x[k] - 440) / (440 - 380)
    g[k] = 0
    b[k] = 1

    # 440-490
    k = (x >= 440) & (x < 490)
    r[k] = 0
    g[k] = (x[k] - 440) / (490 - 440)
    b[k] = 1

    # 490-510
    k = (x >= 490) & (x < 510)
    r[k] = 0
    g[k] = 1
    b[k] = -(x[k] - 510) / (510 - 490)

    # 510-580
    k = (x >= 510) & (x < 580)
    r[k] = (x[k] - 510) / (580 - 510)
    g[k] = 1
    b[k] = 0

    # 580-645
    k = (x >= 580) & (x < 645)
    r[k] = 1
    g[k] = -(x[k] - 645) / (645 - 580)
    b[k] = 0

    # 645-780
    k = (x >= 645) & (x <= 780)
    r[k] = 1
    g[k] = 0
    b[k] = 0

    # 端の減光
    a = np.ones_like(x)
    k = (x >= 380) & (x < 420)
    a[k] = 0.3 + 0.7 * (x[k] - 380) / (420 - 380)
    k = (x > 700) & (x <= 780)
    a[k] = 0.3 + 0.7 * (780 - x[k]) / (780 - 700)

    # ガンマ補正
    gamma = 0.8
    r = (a * r) ** gamma
    g = (a * g) ** gamma
    b = (a * b) ** gamma

    rgb[m, 0] = r
    rgb[m, 1] = g
    rgb[m, 2] = b
    return rgb

def rot_from_yaw_pitch(yaw_deg: float, pitch_deg: float):
    """
    視線方向（カメラ forward）を yaw/pitch で作る。
    ship座標:
      +X: 前方（進行方向）
      +Y: 右
      +Z: 上
    yaw: 上から見て右回り（+Z軸回り）
    pitch: 上を向く（+Y'軸回り）
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # forward をまず yaw で回す（x-y平面）
    fx = np.cos(yaw) * np.cos(pitch)
    fy = np.sin(yaw) * np.cos(pitch)
    fz = np.sin(pitch)
    f = np.array([fx, fy, fz], dtype=float)
    f = f / np.linalg.norm(f)

    # right = f × up_world（ただし平行回避）
    upw = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(f, upw)) > 0.95:
        upw = np.array([0.0, 1.0, 0.0])

    r = np.cross(f, upw)
    r = r / np.linalg.norm(r)

    u = np.cross(r, f)
    u = u / np.linalg.norm(u)

    return f, r, u  # forward, right, up

def aberrate_and_doppler_sources(s_xyz: np.ndarray, beta: float, lam0_nm: float):
    """
    s_xyz: 「観測者から見た星の方向（星のある方向）」= source direction unit vectors
           つまり photon direction は n = -s

    ローレンツ変換（速度 +X 方向）で船の静止系へ:
      photon 4-vector: k = (ω, ω n)
      ω' = γ(ω - β ω n_x) = γ ω (1 - β n_x) = γ ω (1 + β s_x)

    aberration は photon direction n を変換してから source direction に戻す:
      n'_x = (n_x - β)/(1 - β n_x)
      n'_⊥ = n_⊥ /(γ(1 - β n_x))
      s' = -n'
    """
    if beta <= 0:
        lam_obs = np.full(s_xyz.shape[0], lam0_nm, dtype=float)
        return s_xyz.copy(), lam_obs

    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    s = s_xyz
    n = -s  # photon propagation

    nx = n[:, 0]
    ny = n[:, 1]
    nz = n[:, 2]

    den = (1.0 - beta * nx)

    npx = (nx - beta) / den
    npy = ny / (gamma * den)
    npz = nz / (gamma * den)

    n_prime = np.stack([npx, npy, npz], axis=1)
    n_prime = n_prime / np.linalg.norm(n_prime, axis=1, keepdims=True)

    s_prime = -n_prime

    # Doppler（波長は逆比例）: ω' = γ ω (1 - β n_x) = γ ω (1 + β s_x)
    doppler = gamma * (1.0 + beta * s[:, 0])
    lam_obs = lam0_nm / doppler

    return s_prime, lam_obs

def project_azimuthal(s_cam: np.ndarray):
    """
    進行方向（カメラ forward）を中心にした「方位等距離投影（azimuthal equidistant）」:
      θ = arccos(z)（中心からの角距離）
      r = θ / (π/2) で 90° -> r=1
      x = r cosφ, y = r sinφ
    ただし z<=0（背面）は投影しない（見ている半球の外）。
    """
    x = s_cam[:, 0]
    y = s_cam[:, 1]
    z = s_cam[:, 2]

    front = z > 1e-9
    x = x[front]; y = y[front]; z = z[front]

    theta = np.arccos(np.clip(z, -1.0, 1.0))
    r = theta / (0.5 * np.pi)  # 90deg -> 1

    phi = np.arctan2(y, x)
    X = r * np.cos(phi)
    Y = r * np.sin(phi)

    return front, X, Y, theta

def add_glow_traces(fig, X, Y, rgb, base_size, glow_strength):
    """
    グロー表現：外側（大きい・薄い）+ 内側（小さい・濃い）
    """
    # 外側グロー
    glow_size = base_size * (1.0 + 3.0 * glow_strength)
    glow_opacity = 0.10 + 0.25 * glow_strength

    fig.add_trace(
        go.Scatter(
            x=X,
            y=Y,
            mode="markers",
            marker=dict(
                size=glow_size,
                color=rgb,
                opacity=glow_opacity,
                line=dict(width=0),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 本体
    core_size = base_size
    core_opacity = 0.65 + 0.25 * glow_strength

    fig.add_trace(
        go.Scatter(
            x=X,
            y=Y,
            mode="markers",
            marker=dict(
                size=core_size,
                color=rgb,
                opacity=core_opacity,
                line=dict(width=0),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

# ----------------------------
# UI
# ----------------------------
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

    st.caption("※ドラッグ/ピンチ操作が効かない端末でも、ズームは上のスライダーで調整できます。")

# ----------------------------
# 星生成（全球で一様）
# ----------------------------
rng = np.random.default_rng(int(seed))

# 一様分布（球面）
u = rng.random(nstars)
v = rng.random(nstars)
cos_theta = 2.0 * u - 1.0
phi = 2.0 * np.pi * v
sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))

sx = sin_theta * np.cos(phi)
sy = sin_theta * np.sin(phi)
sz = cos_theta
s_xyz = np.stack([sx, sy, sz], axis=1)

# ベース波長（ここは “固定” でもOKだし、後で分布にしてもOK）
lam0 = 550.0  # nm

# ローレンツ変換：方向（光行差）+ ドップラー（色）
s_prime, lam_obs = aberrate_and_doppler_sources(s_xyz, beta, lam0)

# 視線方向の基底（forward/right/up）
fwd, right, up = rot_from_yaw_pitch(yaw, pitch)

# カメラ座標へ（x=right, y=up, z=forward）
x_cam = s_prime @ right
y_cam = s_prime @ up
z_cam = s_prime @ fwd
s_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

# 投影
front_mask, X, Y, theta = project_azimuthal(s_cam)

lam_front = lam_obs[front_mask]

# 可視/不可視
vis = (lam_front >= VISIBLE_MIN) & (lam_front <= VISIBLE_MAX)

# 色（可視は波長色、不可視は描かない）
rgb_vis = wavelength_to_rgb_nm(lam_front[vis])
rgb_vis_str = [f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" for r, g, b in rgb_vis]

X_vis = X[vis]
Y_vis = Y[vis]

X_inv = X[~vis]
Y_inv = Y[~vis]

# ----------------------------
# 図（Plotly）
# ----------------------------
fig = go.Figure()

# 可視星：グロー2重描画
add_glow_traces(fig, X_vis, Y_vis, rgb_vis_str, base_size, glow)

# 不可視：表示ONなら白枠（中は透明）
if show_invisible_outline and X_inv.size > 0:
    fig.add_trace(
        go.Scatter(
            x=X_inv,
            y=Y_inv,
            mode="markers",
            marker=dict(
                size=base_size + 2,
                color="rgba(0,0,0,0)",
                line=dict(color="rgba(255,255,255,0.65)", width=1.2),
                opacity=1.0,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

# 目盛り（30/60/90°）+ 十字（前方中心）
rings = [30, 60, 90]
for deg in rings:
    r = (np.deg2rad(deg)) / (0.5 * np.pi)  # 90°->1
    fig.add_shape(type="circle", xref="x", yref="y",
                  x0=-r, y0=-r, x1=r, y1=r,
                  line=dict(color="rgba(255,255,255,0.35)" if deg < 90 else "rgba(255,255,255,0.55)",
                            width=2.0 if deg == 90 else 1.4))

# 十字
fig.add_shape(type="line", x0=-1.02, y0=0, x1=1.02, y1=0,
              line=dict(color="rgba(120,170,255,0.55)", width=2))
fig.add_shape(type="line", x0=0, y0=-1.02, x1=0, y1=1.02,
              line=dict(color="rgba(120,170,255,0.55)", width=2))

# 方向マーカー（前/後/右/左/上/下）
# ship座標での6方向ベクトルを、現在の視線基底で投影して「点」で置く
dirs = {
    "forward": np.array([+1.0, 0.0, 0.0]),
    "back":    np.array([-1.0, 0.0, 0.0]),
    "right":   np.array([0.0, +1.0, 0.0]),
    "left":    np.array([0.0, -1.0, 0.0]),
    "up":      np.array([0.0, 0.0, +1.0]),
    "down":    np.array([0.0, 0.0, -1.0]),
}

marker_pts = []
for _, d in dirs.items():
    # カメラ座標（x=right,y=up,z=fwd）
    dc = np.array([np.dot(d, right), np.dot(d, up), np.dot(d, fwd)], dtype=float)

    # 投影（front hemisphereのみ）
    if dc[2] <= 1e-9:
        continue
    th = np.arccos(np.clip(dc[2], -1, 1))
    rr = th / (0.5 * np.pi)
    ph = np.arctan2(dc[1], dc[0])
    xx = rr * np.cos(ph)
    yy = rr * np.sin(ph)
    marker_pts.append((xx, yy))

if marker_pts:
    mx = [p[0] for p in marker_pts]
    my = [p[1] for p in marker_pts]
    fig.add_trace(
        go.Scatter(
            x=mx,
            y=my,
            mode="markers",
            marker=dict(size=8, color="rgba(255,255,255,0.85)", line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
        )
    )

# 角度ラベル（30/60/90）— 文字は白
fig.add_annotation(x=(np.deg2rad(30)/(0.5*np.pi))*1.02, y=0.02, text="30°",
                   showarrow=False, font=dict(color="white", size=28))
fig.add_annotation(x=(np.deg2rad(60)/(0.5*np.pi))*1.02, y=0.02, text="60°",
                   showarrow=False, font=dict(color="white", size=28))
fig.add_annotation(x=(np.deg2rad(90)/(0.5*np.pi))*1.02, y=0.02, text="90°",
                   showarrow=False, font=dict(color="white", size=28))

# レイアウト：ダーク背景、軸非表示、ズームスライダー反映
fig.update_layout(
    margin=dict(l=10, r=10, t=40, b=10),
    paper_bgcolor="#070a16",
    plot_bgcolor="#070a16",
    xaxis=dict(visible=False, range=[-1/zoom, 1/zoom]),
    yaxis=dict(visible=False, range=[-1/zoom, 1/zoom], scaleanchor="x", scaleratio=1),
    # これが「速度を変えても視点（ドラッグ/ズーム）を維持」するポイント
    uirevision="KEEP_VIEW",
)

# 上ラベルは v/c だけ
fig.update_layout(
    title=dict(
        text=f"v/c = {beta:.2f}",
        x=0.5,
        y=0.98,
        xanchor="center",
        yanchor="top",
        font=dict(color="white", size=36),
    )
)

with colR:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
