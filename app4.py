import streamlit as st
import numpy as np
import plotly.graph_objects as go

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="STARBOW Simulator", layout="wide")

# 背景を暗く（CSS）
st.markdown(
    """
    <style>
      .stApp { background: radial-gradient(circle at 30% 20%, #101a33 0%, #05070f 55%, #02030a 100%); }
      h1, h2, h3, p, label, div { color: #ffffff; }
      /* スライダーの周りの文字を見やすく */
      .stMarkdown, .stText, .stCaption, .stSlider label { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

VISIBLE_MIN = 380.0
VISIBLE_MAX = 780.0

# =========================
# 便利関数
# =========================
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return v / n

def sample_isotropic_dirs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return normalize(v)

def aberrate_dirs(n: np.ndarray, beta: float) -> np.ndarray:
    """
    宇宙船が +x 方向に速度 beta で動くときの光行差
    n: (N,3) 観測前の方向（単位ベクトル）
    n': 観測後の方向
    """
    if beta <= 0:
        return n.copy()
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)

    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    denom = 1.0 + beta * nx

    npx = (nx + beta) / denom
    npy = ny / (gamma * denom)
    npz = nz / (gamma * denom)

    out = np.stack([npx, npy, npz], axis=1)
    return normalize(out)

def doppler_wavelength(lambda0_nm: float, n: np.ndarray, beta: float) -> np.ndarray:
    """
    観測者が +x 方向に beta で移動。
    周波数: nu' = gamma * nu * (1 + beta * cosθ)  （cosθ = n_x）
    波長: lambda' = lambda / (gamma*(1+beta*n_x))
    """
    if beta <= 0:
        return np.full((n.shape[0],), lambda0_nm, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    factor = gamma * (1.0 + beta * n[:, 0])
    factor = np.clip(factor, 1e-8, None)
    return lambda0_nm / factor

def wavelength_to_rgb_rough(lam_nm: np.ndarray) -> np.ndarray:
    """
    ざっくり可視光の波長→RGB（見た目用）
    """
    lam = lam_nm.copy()

    r = np.zeros_like(lam, dtype=float)
    g = np.zeros_like(lam, dtype=float)
    b = np.zeros_like(lam, dtype=float)

    # 380-440
    m = (lam >= 380) & (lam < 440)
    r[m] = -(lam[m] - 440) / (440 - 380)
    g[m] = 0
    b[m] = 1

    # 440-490
    m = (lam >= 440) & (lam < 490)
    r[m] = 0
    g[m] = (lam[m] - 440) / (490 - 440)
    b[m] = 1

    # 490-510
    m = (lam >= 490) & (lam < 510)
    r[m] = 0
    g[m] = 1
    b[m] = -(lam[m] - 510) / (510 - 490)

    # 510-580
    m = (lam >= 510) & (lam < 580)
    r[m] = (lam[m] - 510) / (580 - 510)
    g[m] = 1
    b[m] = 0

    # 580-645
    m = (lam >= 580) & (lam < 645)
    r[m] = 1
    g[m] = -(lam[m] - 645) / (645 - 580)
    b[m] = 0

    # 645-780
    m = (lam >= 645) & (lam <= 780)
    r[m] = 1
    g[m] = 0
    b[m] = 0

    # ガンマ的な補正（弱め）
    rgb = np.stack([r, g, b], axis=1)
    rgb = np.clip(rgb, 0, 1)
    rgb = rgb ** 0.9
    return rgb

def rgba_strings(rgb: np.ndarray, alpha: float) -> np.ndarray:
    a = np.clip(alpha, 0, 1)
    return np.array([f"rgba({int(255*x)}, {int(255*y)}, {int(255*z)}, {a})" for x, y, z in rgb])

def dirs_to_screen_xy(n: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """
    “宇宙船中心視点”の2D表示用：方向ベクトルを回転してから
    ステレオ投影（簡易）でスクリーン座標へ
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # yaw: z軸回り、pitch: y軸回り（わかりやすさ優先）
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,    0, 1]], dtype=float)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=float)

    v = (Ry @ (Rz @ n.T)).T  # (N,3)

    # 前方（+x）をスクリーン中心に置く
    # ステレオ投影： (y,z)/(1+x)
    denom = (1.0 + v[:, 0])
    denom = np.clip(denom, 1e-6, None)
    x2 = v[:, 1] / denom
    y2 = v[:, 2] / denom
    return np.stack([x2, y2], axis=1)

# =========================
# セッション状態（視点がリセットされないため）
# =========================
if "yaw" not in st.session_state:
    st.session_state.yaw = 0.0
if "pitch" not in st.session_state:
    st.session_state.pitch = 0.0
if "zoom" not in st.session_state:
    st.session_state.zoom = 2.2
if "mode3d" not in st.session_state:
    st.session_state.mode3d = False  # ★起動時はOFF
if "seed" not in st.session_state:
    st.session_state.seed = 12345
if "n_stars" not in st.session_state:
    st.session_state.n_stars = 2500

# =========================
# UI（左：パラメータ）
# =========================
colL, colR = st.columns([1, 2], gap="large")

with colL:
    st.markdown("## パラメータ")

    beta = st.slider("v/c", 0.0, 0.99, 0.50, 0.01)

    n_stars = st.slider("星の数", 100, 20000, int(st.session_state.n_stars), 100)
    st.session_state.n_stars = n_stars

    seed = st.number_input("配置シード（同じ値で同じ星配置）", value=int(st.session_state.seed), step=1)
    st.session_state.seed = int(seed)

    st.markdown("---")
    st.session_state.mode3d = st.toggle("ドラッグで見回し（3Dモード）", value=st.session_state.mode3d)

    st.markdown("---")
    st.markdown("## 表示")

    zoom = st.slider("ズーム（大きいほど拡大）", 0.6, 6.0, float(st.session_state.zoom), 0.05)
    st.session_state.zoom = float(zoom)

    star_size = st.slider("星の大きさ", 1.0, 10.0, 3.0, 0.5)
    glow_strength = st.slider("グロー強さ", 0.0, 1.0, 0.25, 0.01)

    show_invisible_white = st.checkbox("不可視光を表示（白枠）", value=False)

    st.markdown("---")

    # ★「文字が見えない白ボタン」問題を避けるため、普通のst.buttonで明示表示
    if st.button("初期配置に戻す（視点・ズーム）", use_container_width=True):
        st.session_state.yaw = 0.0
        st.session_state.pitch = 0.0
        st.session_state.zoom = 2.2
        st.session_state.mode3d = False
        st.rerun()

# =========================
# 星生成（等方分布→光行差→ドップラー）
# =========================
base_lambda = 580.0  # 基準波長（黄色寄り）
dirs0 = sample_isotropic_dirs(n_stars, seed=st.session_state.seed)
dirs = aberrate_dirs(dirs0, beta=beta)
lam = doppler_wavelength(base_lambda, dirs0, beta=beta)  # ここは“元の入射方向”でOK（θ基準）

visible = (lam >= VISIBLE_MIN) & (lam <= VISIBLE_MAX)

rgb = wavelength_to_rgb_rough(lam)
# 可視は色、不可視は背景色（見えない）
bg_rgb = np.array([0.03, 0.04, 0.10], dtype=float)  # 背景に馴染む暗色
rgb_visible = rgb.copy()
rgb_visible[~visible] = bg_rgb

# =========================
# 目印（前後上下左右）
# =========================
# 進行方向 +x を「前」
markers_dirs = np.array([
    [ 1, 0, 0],  # +x 前
    [-1, 0, 0],  # -x 後
    [ 0, 1, 0],  # +y 右（定義）
    [ 0,-1, 0],  # -y 左
    [ 0, 0, 1],  # +z 上
    [ 0, 0,-1],  # -z 下
], dtype=float)
markers_text = ["+x(前)", "-x(後)", "+y(右)", "-y(左)", "+z(上)", "-z(下)"]

# =========================
# 描画
# =========================
with colR:
    title_suffix = "（ドラッグで見回し）" if st.session_state.mode3d else "（スライダーで向き）"
    st.markdown(f"# STARBOW Simulator（宇宙船中心視点）")
    st.markdown(f"## v/c = {beta:.2f} {title_suffix}")

    if st.session_state.mode3d:
        # -------------------------
        # 3Dモード（Plotly scene）
        # -------------------------
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

        # グロー（大きい点＋薄い）
        glow_colors = rgba_strings(rgb_visible, alpha=0.12 + 0.35 * glow_strength)
        core_colors = rgba_strings(rgb_visible, alpha=0.45 + 0.45 * (1.0 - 0.3 * glow_strength))

        fig = go.Figure()

        # グロー層
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                name="glow",
                marker=dict(
                    size=star_size * (2.8 + 3.0 * glow_strength),
                    color=glow_colors,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # コア層
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                name="stars",
                marker=dict(
                    size=star_size,
                    color=core_colors,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # 不可視光の白枠（単一トグル）
        if show_invisible_white:
            inv = ~visible
            if np.any(inv):
                fig.add_trace(
                    go.Scatter3d(
                        x=x[inv], y=y[inv], z=z[inv],
                        mode="markers",
                        name="invisible",
                        marker=dict(
                            size=star_size * 1.05,
                            color="rgba(0,0,0,0)",  # 中身透明
                            line=dict(color="rgba(255,255,255,0.75)", width=2),  # ★widthは必ず数値1個
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # 目印（白点＋文字）
        md = markers_dirs
        fig.add_trace(
            go.Scatter3d(
                x=md[:, 0], y=md[:, 1], z=md[:, 2],
                mode="markers+text",
                text=markers_text,
                textposition="top center",
                marker=dict(size=10, color="rgba(255,255,255,0.95)"),
                textfont=dict(color="rgba(255,255,255,0.9)", size=14),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # ★「俯瞰に戻る」対策：uirevision固定
        # ★「宇宙船視点に近い」初期：eyeを原点近くに
        #  zoomスライダーで eye距離だけ調整
        eye_dist = 0.20 / max(0.1, st.session_state.zoom)  # zoom大→近く
        camera = dict(
            eye=dict(x=eye_dist, y=0.0, z=0.0),
            center=dict(x=0.0, y=0.0, z=0.0),
            up=dict(x=0.0, y=0.0, z=1.0),
        )

        fig.update_layout(
            uirevision="KEEP_VIEW",  # これが肝
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="cube",
                camera=camera,
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        # ★スマホ操作：1本指ドラッグ=回転 になってほしいので orbit を明示
        config = dict(
            displaylogo=False,
            scrollZoom=True,   # 2本指ピンチ/ホイールズームを効かせやすい
        )
        fig.update_layout(scene_dragmode="orbit")

        st.plotly_chart(fig, use_container_width=True, config=config)

        st.caption("※ 3Dモードはドラッグで見回しできます。パラメータを変えても視点を維持するよう調整済み。")

    else:
        # -------------------------
        # 2Dモード（スライダーで向き）
        # -------------------------
        # “見ている方向”はスライダーで（3D OFF のときだけ）
        st.session_state.yaw = st.slider("ヨー（左右）", -180.0, 180.0, float(st.session_state.yaw), 1.0)
        st.session_state.pitch = st.slider("ピッチ（上下）", -89.0, 89.0, float(st.session_state.pitch), 1.0)

        xy = dirs_to_screen_xy(dirs, st.session_state.yaw, st.session_state.pitch)

        # ズーム：xyを拡大表示
        xy = xy * st.session_state.zoom

        # 可視の点
        vis_xy = xy[visible]
        vis_rgb = rgb[visible]
        vis_col = rgba_strings(vis_rgb, alpha=0.55 + 0.35 * (1.0 - 0.3 * glow_strength))
        glow_col = rgba_strings(vis_rgb, alpha=0.10 + 0.35 * glow_strength)

        fig2 = go.Figure()

        # グロー
        fig2.add_trace(
            go.Scatter(
                x=vis_xy[:, 0],
                y=vis_xy[:, 1],
                mode="markers",
                marker=dict(
                    size=star_size * (5.0 + 6.0 * glow_strength),
                    color=glow_col,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # コア
        fig2.add_trace(
            go.Scatter(
                x=vis_xy[:, 0],
                y=vis_xy[:, 1],
                mode="markers",
                marker=dict(size=star_size, color=vis_col),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # 不可視光の白枠（単一トグル）
        if show_invisible_white:
            inv_xy = xy[~visible]
            if inv_xy.shape[0] > 0:
                fig2.add_trace(
                    go.Scatter(
                        x=inv_xy[:, 0],
                        y=inv_xy[:, 1],
                        mode="markers",
                        marker=dict(
                            size=star_size * 1.2,
                            color="rgba(0,0,0,0)",
                            line=dict(color="rgba(255,255,255,0.75)", width=2),
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # 目印（2D表示でも置く）
        mk_xy = dirs_to_screen_xy(markers_dirs, st.session_state.yaw, st.session_state.pitch) * st.session_state.zoom
        fig2.add_trace(
            go.Scatter(
                x=mk_xy[:, 0],
                y=mk_xy[:, 1],
                mode="markers+text",
                text=markers_text,
                textposition="top center",
                marker=dict(size=10, color="rgba(255,255,255,0.95)"),
                textfont=dict(color="rgba(255,255,255,0.9)", size=14),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig2.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(fig2, use_container_width=True, config=dict(displaylogo=False))

        st.caption("※ 2Dモードはスライダーで視点（ヨー/ピッチ）を調整できます。3Dモードにするとドラッグで見回し。")
