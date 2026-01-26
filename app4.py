import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
#  基本設定
# =========================
st.set_page_config(page_title="STARBOW Simulator (Center View)", layout="wide")

# 背景（仄暗い夜空）
BG = "rgb(6, 8, 26)"          # 図の背景
PAPER_BG = "rgb(6, 8, 26)"    # 外側も同色

VISIBLE_MIN = 380.0
VISIBLE_MAX = 780.0

# -------------------------
#  便利関数
# -------------------------
def clamp(x, a, b):
    return max(a, min(b, x))

def wavelength_to_rgb_approx(lam_nm: float):
    """
    ざっくり可視域→RGB（ガンマ補正ありの簡易版）
    """
    lam = lam_nm
    if lam < 380 or lam > 780:
        return (0, 0, 0)

    if 380 <= lam < 440:
        r, g, b = -(lam - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= lam < 490:
        r, g, b = 0.0, (lam - 440) / (490 - 440), 1.0
    elif 490 <= lam < 510:
        r, g, b = 0.0, 1.0, -(lam - 510) / (510 - 490)
    elif 510 <= lam < 580:
        r, g, b = (lam - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= lam < 645:
        r, g, b = 1.0, -(lam - 645) / (645 - 580), 0.0
    else:  # 645-780
        r, g, b = 1.0, 0.0, 0.0

    # 端っこの減衰
    if 380 <= lam < 420:
        factor = 0.3 + 0.7 * (lam - 380) / (420 - 380)
    elif 420 <= lam <= 700:
        factor = 1.0
    else:  # 700-780
        factor = 0.3 + 0.7 * (780 - lam) / (780 - 700)

    gamma = 0.8
    r = (r * factor) ** gamma
    g = (g * factor) ** gamma
    b = (b * factor) ** gamma

    return (int(255 * clamp(r, 0, 1)), int(255 * clamp(g, 0, 1)), int(255 * clamp(b, 0, 1)))

def rgba_str(rgb, a):
    r, g, b = rgb
    return f"rgba({r},{g},{b},{a})"

def sample_unit_sphere(n, rng):
    """
    球面上一様（方向ベクトル）
    """
    u = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2*np.pi, size=n)
    sin_t = np.sqrt(1.0 - u*u)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = u
    return np.vstack([x, y, z]).T  # (n,3)

def aberrate_and_doppler(k_lab, beta, vhat=np.array([0.0, 0.0, 1.0])):
    """
    k_lab: (n,3) incoming photon direction in lab frame (unit)
    ship moves along +vhat with speed beta (c=1)
    Returns:
      k_ship: (n,3) direction in ship frame
      doppler_factor: (n,)  f' = doppler_factor * f
    """
    beta = float(beta)
    if beta <= 0:
        return k_lab.copy(), np.ones(k_lab.shape[0])

    gamma = 1.0 / np.sqrt(1.0 - beta*beta)

    # parallel component
    k_par = (k_lab @ vhat)  # (n,)
    k_par_vec = np.outer(k_par, vhat)  # (n,3)
    k_perp = k_lab - k_par_vec

    # aberration (lab -> ship)
    denom = (1.0 - beta * k_par)
    k_par_p = (k_par - beta) / denom
    k_perp_p = k_perp / (gamma * denom[:, None])

    k_ship = k_perp_p + np.outer(k_par_p, vhat)

    # normalize
    k_ship /= np.linalg.norm(k_ship, axis=1, keepdims=True)

    # doppler: f' = gamma(1 - beta*k_par)*f
    doppler_factor = gamma * (1.0 - beta * k_par)

    return k_ship, doppler_factor

# =========================
#  UI
# =========================
st.markdown(
    f"""
    <style>
      .stApp {{
        background: {PAPER_BG};
      }}
      h1, h2, h3, p, label, .stMarkdown {{
        color: white !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("STARBOW Simulator（中心視点）")

colL, colR = st.columns([1, 4], vertical_alignment="top")

with colL:
    beta = st.slider("速度 v/c", 0.0, 0.99, 0.50, 0.01)
    nstars = st.slider("星の数", 200, 6000, 1500, 100)
    seed = st.number_input("ランダムシード（配置固定）", value=42, step=1)
    base_lambda = st.slider("基準波長 λ0（nm）※全星共通", 380, 780, 580, 1)
    show_invisible = st.toggle("不可視光（見えない星）も“位置”を表示", value=False)
    invisible_outline = st.toggle("不可視光は『白い枠』で表示", value=True)

    st.caption("ドラッグ：視点回転 / ホイール：ズーム（端末によっては2本指）")

# =========================
#  計算
# =========================
rng = np.random.default_rng(int(seed))

# 星の「位置方向」＝星への方向 n（unit）
n_dir = sample_unit_sphere(int(nstars), rng)  # (n,3)

# 光は星→観測者へ来るので、incoming photon direction k = -n
k_lab = -n_dir

k_ship, doppler_f = aberrate_and_doppler(k_lab, beta, vhat=np.array([0.0, 0.0, 1.0]))

# 波長変化（λ' = λ / doppler_factor）
lam_obs = base_lambda / doppler_f

# 可視判定
is_visible = (lam_obs >= VISIBLE_MIN) & (lam_obs <= VISIBLE_MAX)

# 表示用：巨大半径の球に貼り付け（見た目を“遠方の星空”っぽく）
R = 1000.0
pts = R * k_ship  # (n,3)

# 色（可視：波長色 / 不可視：背景同化 or 枠だけ）
colors = []
sizes_core = []
sizes_glow = []

for i in range(pts.shape[0]):
    if is_visible[i]:
        rgb = wavelength_to_rgb_approx(float(lam_obs[i]))
        colors.append(rgb)
        sizes_core.append(5)
        sizes_glow.append(18)
    else:
        # 不可視
        colors.append((0, 0, 0))
        sizes_core.append(3)
        sizes_glow.append(12)

# =========================
#  Plotlyで描画（ドラッグで見回し）
# =========================
# 2層（グロー + コア）
visible_idx = np.where(is_visible)[0]
invisible_idx = np.where(~is_visible)[0]

fig = go.Figure()

# --- 可視：グロー
if visible_idx.size > 0:
    glow_colors = [rgba_str(colors[i], 0.18) for i in visible_idx]
    fig.add_trace(
        go.Scatter3d(
            x=pts[visible_idx, 0], y=pts[visible_idx, 1], z=pts[visible_idx, 2],
            mode="markers",
            marker=dict(size=18, color=glow_colors),
            hoverinfo="skip",
            name="glow"
        )
    )

# --- 可視：コア
if visible_idx.size > 0:
    core_colors = [rgba_str(colors[i], 0.95) for i in visible_idx]
    fig.add_trace(
        go.Scatter3d(
            x=pts[visible_idx, 0], y=pts[visible_idx, 1], z=pts[visible_idx, 2],
            mode="markers",
            marker=dict(size=5, color=core_colors),
            hoverinfo="skip",
            name="stars"
        )
    )

# --- 不可視：オプション表示（枠だけ or ほぼ見えない）
if show_invisible and invisible_idx.size > 0:
    if invisible_outline:
        # 透明塗り + 白枠
        fig.add_trace(
            go.Scatter3d(
                x=pts[invisible_idx, 0], y=pts[invisible_idx, 1], z=pts[invisible_idx, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color="rgba(0,0,0,0.0)",
                    line=dict(color="rgba(255,255,255,0.55)", width=1),
                ),
                hoverinfo="skip",
                name="invisible"
            )
        )
    else:
        # 背景同化（ほぼ見えないが薄く存在だけ）
        fig.add_trace(
            go.Scatter3d(
                x=pts[invisible_idx, 0], y=pts[invisible_idx, 1], z=pts[invisible_idx, 2],
                mode="markers",
                marker=dict(size=3, color="rgba(255,255,255,0.05)"),
                hoverinfo="skip",
                name="invisible"
            )
        )

# 視点：なるべく「中心にいる」感（eyeを中心近くに）
# ※Plotlyの仕様で完全に“内側カメラ”は難しいが、見た目はかなり近づく
camera = dict(
    eye=dict(x=0.001, y=0.001, z=0.001),  # 極小
    center=dict(x=0, y=0, z=0),
    up=dict(x=0, y=1, z=0),
)

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor=BG,
        camera=camera,
        aspectmode="data",
    ),
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=BG,
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False,
)

with colR:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

