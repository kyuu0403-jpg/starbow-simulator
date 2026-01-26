import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# ページ設定
# =========================
st.set_page_config(page_title="STARBOW Simulator", layout="wide")

# =========================
# 見た目（ダーク背景）
# =========================
DARK_BG = "#070A16"     # 夜空っぽい濃紺
PANEL_BG = "#050814"
TEXT = "white"

st.markdown(
    f"""
    <style>
      .stApp {{
        background: radial-gradient(1200px 800px at 50% 20%, #0E1433 0%, {DARK_BG} 55%, #000000 100%);
        color: {TEXT};
      }}
      h1, h2, h3, p, label, span, div {{
        color: {TEXT} !important;
      }}
      /* 左パネルっぽく見せる */
      section[data-testid="stSidebar"] > div {{
        background: {PANEL_BG};
      }}
      /* ボタン文字が見えない事故を避ける */
      .stButton>button {{
        color: #FFFFFF !important;
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
      }}
      .stButton>button:hover {{
        background: rgba(255,255,255,0.14) !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 物理：可視域 & 色
# =========================
VISIBLE_MIN = 380.0
VISIBLE_MAX = 780.0

def wavelength_to_rgb(lam_nm: float) -> str:
    """
    簡易の可視光→RGB（Plotly用の'rgb(r,g,b)'文字列）
    380-780nm 外は呼ばない前提。
    """
    lam = lam_nm
    if lam < 380: lam = 380
    if lam > 780: lam = 780

    if 380 <= lam < 440:
        r = -(lam - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= lam < 490:
        r = 0.0
        g = (lam - 440) / (490 - 440)
        b = 1.0
    elif 490 <= lam < 510:
        r = 0.0
        g = 1.0
        b = -(lam - 510) / (510 - 490)
    elif 510 <= lam < 580:
        r = (lam - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= lam < 645:
        r = 1.0
        g = -(lam - 645) / (645 - 580)
        b = 0.0
    else:  # 645-780
        r = 1.0
        g = 0.0
        b = 0.0

    # 端は暗く
    if 380 <= lam < 420:
        f = 0.3 + 0.7 * (lam - 380) / (420 - 380)
    elif 420 <= lam <= 700:
        f = 1.0
    else:
        f = 0.3 + 0.7 * (780 - lam) / (780 - 700)

    r = int(max(0, min(255, 255 * (r * f))))
    g = int(max(0, min(255, 255 * (g * f))))
    b = int(max(0, min(255, 255 * (b * f))))
    return f"rgb({r},{g},{b})"

# =========================
# 乱数で全球に一様な星配置（方向ベクトル）
# =========================
def random_unit_vectors(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

# =========================
# 相対論的：光行差（方向の変換）& ドップラー（波長）
# 進行方向 +x を基準（宇宙船が +x に進む）
# =========================
def aberrate_and_doppler(dirs: np.ndarray, beta: float, lam0_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """
    dirs: (N,3) 単位ベクトル（宇宙船静止系での星の方向）
    返り値:
      dirs_obs: (N,3) 観測者（動いてる宇宙船）から見た方向（単位ベクトル）
      lam_obs : (N,) 観測される波長[nm]
    """
    if beta <= 0:
        lam = np.full(dirs.shape[0], lam0_nm, dtype=float)
        return dirs.copy(), lam

    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    nx = dirs[:, 0]
    ny = dirs[:, 1]
    nz = dirs[:, 2]

    # 光行差（+x方向へ速度beta）
    # n'_x = (n_x + beta) / (1 + beta n_x)
    # n'_{y,z} = n_{y,z} / (gamma (1 + beta n_x))
    den = (1.0 + beta * nx)
    npx = (nx + beta) / den
    npy = ny / (gamma * den)
    npz = nz / (gamma * den)

    nprime = np.stack([npx, npy, npz], axis=1)
    nprime /= np.linalg.norm(nprime, axis=1, keepdims=True)

    # ドップラー（波長）: 観測者系での周波数比 D = gamma(1+beta*n_x)
    # 周波数 f' = D f  -> 波長 λ' = λ / D
    D = gamma * (1.0 + beta * nx)
    lam_obs = lam0_nm / D

    return nprime, lam_obs

# =========================
# “宇宙船視点”の3Dカメラ（俯瞰に戻さない最終手段）
# Plotlyは「eye から原点を見る」カメラなので、原点近くにeyeを置く。
# =========================
def ship_camera() -> dict:
    return {
        "eye": {"x": 0.001, "y": 0.001, "z": 0.001},
        "up":  {"x": 0.0, "y": 0.0, "z": 1.0},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    }

# =========================
# セッション初期化（起動時：3DモードOFF）
# =========================
if "drag3d" not in st.session_state:
    st.session_state.drag3d = False  # ←起動時OFF
if "zoom" not in st.session_state:
    st.session_state.zoom = 2.2
if "glow" not in st.session_state:
    st.session_state.glow = 0.25
if "star_size" not in st.session_state:
    st.session_state.star_size = 3
if "seed" not in st.session_state:
    st.session_state.seed = 12345
if "beta" not in st.session_state:
    st.session_state.beta = 0.50
if "n_stars" not in st.session_state:
    st.session_state.n_stars = 2500
if "show_invisible" not in st.session_state:
    st.session_state.show_invisible = False

# =========================
# UI
# =========================
st.markdown("# **STARBOW Simulator（宇宙船中心視点）**")

left, right = st.columns([0.33, 0.67], gap="large")

with left:
    st.markdown("## パラメータ")

    beta = st.slider("v/c", 0.00, 0.99, float(st.session_state.beta), 0.01)
    st.session_state.beta = beta

    n_stars = st.slider("星の数", 50, 15000, int(st.session_state.n_stars), 50)
    st.session_state.n_stars = n_stars

    seed = st.number_input("配置シード（同じ値で同じ星配置）", value=int(st.session_state.seed), step=1)
    st.session_state.seed = int(seed)

    st.session_state.drag3d = st.toggle("ドラッグで見回し（3Dモード）", value=bool(st.session_state.drag3d))
    st.markdown("---")

    st.markdown("## 表示")
    st.session_state.zoom = st.slider("ズーム（大きいほど拡大）", 1.0, 6.0, float(st.session_state.zoom), 0.05)
    st.session_state.star_size = st.slider("星の大きさ", 1, 10, int(st.session_state.star_size), 1)
    st.session_state.glow = st.slider("グロー強さ", 0.0, 1.5, float(st.session_state.glow), 0.05)

    st.session_state.show_invisible = st.toggle("不可視光を表示（白枠）", value=bool(st.session_state.show_invisible))

    st.markdown("---")

    # ✅ 文字が常に見える「初期化」ボタン
    if st.button("初期化（パラメータと視点をリセット）", use_container_width=True):
        st.session_state.beta = 0.50
        st.session_state.n_stars = 2500
        st.session_state.seed = 12345
        st.session_state.zoom = 2.2
        st.session_state.star_size = 3
        st.session_state.glow = 0.25
        st.session_state.show_invisible = False
        st.session_state.drag3d = False  # 起動時と同じでOFFに戻す
        st.rerun()

# =========================
# データ生成
# =========================
dirs = random_unit_vectors(st.session_state.n_stars, st.session_state.seed)

# ベース波長：黄色（統一）
LAMBDA0 = 550.0

dirs_obs, lam_obs = aberrate_and_doppler(dirs, st.session_state.beta, LAMBDA0)

# 可視/不可視
visible = (lam_obs >= VISIBLE_MIN) & (lam_obs <= VISIBLE_MAX)

# 色（STARBOWっぽく：観測波長で色が変わる）
colors = np.array([wavelength_to_rgb(l) if (VISIBLE_MIN <= l <= VISIBLE_MAX) else "rgba(0,0,0,0)" for l in lam_obs])

# 不可視は「白枠のみ」表示（トグルON時）
outline_color = np.where(visible, "rgba(0,0,0,0)", "rgba(255,255,255,0.9)")
outline_width = np.where(visible, 0, 2)

if not st.session_state.show_invisible:
    # 不可視は完全に消す
    keep = visible
else:
    keep = np.ones_like(visible, dtype=bool)

dirs_plot = dirs_obs[keep]
colors_plot = colors[keep]
outline_color_plot = outline_color[keep]
outline_width_plot = outline_width[keep]
lam_plot = lam_obs[keep]
vis_plot = visible[keep]

# サイズ & グロー（Plotlyは本物のグローが弱いので、同一点に薄い大きい点を重ねて擬似）
base_size = st.session_state.star_size
glow_strength = st.session_state.glow

# =========================
# 3Dモード：Plotly 3D（ドラッグ見回し）
# 重要：俯瞰に戻らないよう “毎回 ship_camera() を強制”
# =========================
with right:
    title_suffix = "（ドラッグで見回し）" if st.session_state.drag3d else ""
    st.markdown(f"## v/c = {st.session_state.beta:.2f} {title_suffix}")

    if st.session_state.drag3d:
        # 3D scatter：星は半径1の球面上
        x, y, z = dirs_plot[:, 0], dirs_plot[:, 1], dirs_plot[:, 2]

        # グロー用（薄く大きい点）
        fig = go.Figure()

        if glow_strength > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="markers",
                    marker=dict(
                        size=max(2, base_size * (3.0 + 6.0 * glow_strength)),
                        color=colors_plot,
                        opacity=min(0.35, 0.10 + 0.25 * glow_strength),
                        line=dict(color=outline_color_plot, width=outline_width_plot),
                    ),
                    hoverinfo="skip",
                    name="glow",
                )
            )

        # 本体
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(
                    size=base_size * 2.0,
                    color=colors_plot,
                    opacity=0.95,
                    line=dict(color=outline_color_plot, width=outline_width_plot),
                ),
                hoverinfo="skip",
                name="stars",
            )
        )

        # 6方向目印（±x, ±y, ±z） ※ “宇宙船の周り”の目安
        marks = {
            "+x(前)": (1.0, 0.0, 0.0),
            "-x(後)": (-1.0, 0.0, 0.0),
            "+y(右)": (0.0, 1.0, 0.0),
            "-y(左)": (0.0, -1.0, 0.0),
            "+z(上)": (0.0, 0.0, 1.0),
            "-z(下)": (0.0, 0.0, -1.0),
        }
        mx = [v[0] for v in marks.values()]
        my = [v[1] for v in marks.values()]
        mz = [v[2] for v in marks.values()]
        mt = list(marks.keys())

        fig.add_trace(
            go.Scatter3d(
                x=mx, y=my, z=mz,
                mode="markers+text",
                marker=dict(size=10, color="white", opacity=0.95),
                text=mt,
                textposition="top center",
                hoverinfo="skip",
                name="marks",
            )
        )

        # “宇宙船（原点）”マーク
        fig.add_trace(
            go.Scatter3d(
                x=[0.0], y=[0.0], z=[0.0],
                mode="markers+text",
                marker=dict(size=6, color="white", opacity=0.9),
                text=["ship"],
                textposition="bottom center",
                hoverinfo="skip",
                name="ship",
            )
        )

        # レイアウト（ダーク）
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgba(0,0,0,0)",
                aspectmode="data",
                camera=ship_camera(),  # ✅最重要：俯瞰に戻さない
            ),
        )

        # ズーム：Plotlyは“zoom値”を直接は持たないので、camera.eye を拡大縮小で近似
        # zoom↑ => 近づく（eye小さく）
        zf = float(st.session_state.zoom)
        eye_scale = max(0.0004, 0.0025 / zf)
        cam = ship_camera()
        cam["eye"] = {"x": eye_scale, "y": eye_scale, "z": eye_scale}
        fig.update_layout(scene_camera=cam)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=dict(
                scrollZoom=True,      # 2本指ズーム（トラックパッド/マウスホイール）
                displayModeBar=False,
            ),
        )

        st.caption("※ 3Dモードはドラッグで見回しできます。パラメータ変更時も“俯瞰”には戻さず、宇宙船視点に戻します。")

    # =========================
    # 2Dモード：リング（これまでの“見やすいやつ”）
    # =========================
    else:
        # 観測方向を2Dへ：進行方向(+x)を中心、角度θ = arccos(n_x)
        nx = dirs_plot[:, 0]
        ny = dirs_plot[:, 1]
        nz = dirs_plot[:, 2]

        theta = np.arccos(np.clip(nx, -1.0, 1.0))  # 0..pi
        # 半球の投影（前方90°までを円内に）
        # r = theta / (pi/2) で 0..1（θ<=90°のみ）
        front = theta <= (math.pi / 2)
        r = (theta[front] / (math.pi / 2))
        phi = np.arctan2(nz[front], ny[front])  # y-z 平面で角度

        # 画面座標（中心=前方）
        X = r * np.cos(phi)
        Y = r * np.sin(phi)

        # 色・枠
        c2 = colors_plot[front]
        oc2 = outline_color_plot[front]
        ow2 = outline_width_plot[front]

        fig2 = go.Figure()

        # グロー
        if glow_strength > 0:
            fig2.add_trace(
                go.Scatter(
                    x=X, y=Y,
                    mode="markers",
                    marker=dict(
                        size=max(4, base_size * (8.0 + 10.0 * glow_strength)),
                        color=c2,
                        opacity=min(0.35, 0.10 + 0.25 * glow_strength),
                        line=dict(color=oc2, width=ow2),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # 本体
        fig2.add_trace(
            go.Scatter(
                x=X, y=Y,
                mode="markers",
                marker=dict(
                    size=base_size * 3.0,
                    color=c2,
                    opacity=0.95,
                    line=dict(color=oc2, width=ow2),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # リング（30°, 60°, 90°）
        rings_deg = [30, 60, 90]
        for deg in rings_deg:
            rr = deg / 90.0
            t = np.linspace(0, 2*np.pi, 360)
            fig2.add_trace(
                go.Scatter(
                    x=rr*np.cos(t), y=rr*np.sin(t),
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.35)", width=2 if deg==90 else 1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=[rr], y=[0],
                    mode="text",
                    text=[f"{deg}°"],
                    textposition="middle right",
                    textfont=dict(color="white", size=22),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # 十字
        fig2.add_trace(go.Scatter(x=[-1.05, 1.05], y=[0, 0], mode="lines",
                                  line=dict(color="rgba(120,180,255,0.8)", width=3),
                                  hoverinfo="skip", showlegend=False))
        fig2.add_trace(go.Scatter(x=[0, 0], y=[-1.05, 1.05], mode="lines",
                                  line=dict(color="rgba(120,180,255,0.8)", width=3),
                                  hoverinfo="skip", showlegend=False))

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(visible=False, range=[-1.15, 1.15]),
            yaxis=dict(visible=False, range=[-1.15, 1.15], scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(
            fig2,
            use_container_width=True,
            config=dict(displayModeBar=False),
        )
        st.caption("※ 2Dモードは前方（進行方向）90°の範囲を表示。3DはOFFのときに見やすいリング表示になります。")
