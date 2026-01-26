import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="STARBOW Simulator", layout="wide")

BG = "#060a14"         # 背景（夜空っぽい暗色）
PANEL_BG = "#060a14"   # Streamlitの周辺も暗く見せたい
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
      /* スライダーのラベル等 */
      [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background: {PANEL_BG};
      }}
      /* ボタンの文字が見えない問題対策 */
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
    """一様分布の単位ベクトルを n 個生成（球面一様）"""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def aberrate_directions(n: np.ndarray, beta: float) -> np.ndarray:
    """
    光行差：+x 方向へ速度 beta で進む観測者(宇宙船)から見た方向 n' を返す
    n は「元の静止系での光の到来方向（単位ベクトル）」として扱う。
    """
    if beta <= 0:
        return n.copy()

    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    nx = n[:, 0]
    ny = n[:, 1]
    nz = n[:, 2]

    denom = (1.0 - beta * nx)
    # 安全策：ゼロ割り回避
    denom = np.clip(denom, 1e-12, None)

    npx = (nx - beta) / denom
    npy = ny / (gamma * denom)
    npz = nz / (gamma * denom)

    npv = np.stack([npx, npy, npz], axis=1)
    npv /= np.linalg.norm(npv, axis=1, keepdims=True)
    return npv

def doppler_factor(n_prime: np.ndarray, beta: float) -> np.ndarray:
    """
    ドップラー因子 D = gamma(1 - beta * cosθ) の逆っぽい表現が色々あるけど、
    ここでは「観測波長が短くなる（青方偏移）ほど色が前方で強く出る」ように
    直感的に使えるスカラーを返す。
    """
    if beta <= 0:
        return np.ones(len(n_prime))

    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    # cosθ'（宇宙船進行方向 +x に対する角）
    cos_th = n_prime[:, 0]
    # 観測周波数比（よく使う形）：nu' / nu = gamma*(1 - beta*cosθ_static) と表現差があるが
    # ここでは前方で >1 になる形が欲しいので、以下を採用：
    # D = gamma * (1 + beta*cosθ')  （前方 cosθ'=1 で大きくなる）
    D = gamma * (1.0 + beta * cos_th)
    return D

def rot_yaw_pitch(v: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """方向ベクトル v を yaw(左右) と pitch(上下) で回転"""
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # yaw: z軸回り回転（左右）
    Rz = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # pitch: y軸回り回転（上下）
    Ry = np.array([
        [ cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp]
    ])

    R = Ry @ Rz
    return v @ R.T

def to_view_plane_coords(n_ship: np.ndarray, yaw_deg: float, pitch_deg: float):
    """
    2D表示用：視線方向が +x になるように座標を回してから、
    前方半球(theta<=90°)を「角度座標(θ,φ)」として (u,v) を返す。
    """
    # 視線回転（ユーザーが見たい方向を +x に持ってくる）
    # つまり、星方向ベクトルを逆向きに回す（見回し相当）
    n2 = rot_yaw_pitch(n_ship, -yaw_deg, -pitch_deg)

    # theta: 視線(+x)からの角度
    cos_th = np.clip(n2[:, 0], -1.0, 1.0)
    theta = np.arccos(cos_th)  # [0..pi]
    theta_deg = np.rad2deg(theta)

    # 前方半球
    mask = theta_deg <= 90.0

    # phi: 視線軸周り（y,z）平面の角
    phi = np.arctan2(n2[:, 2], n2[:, 1])  # [-pi, pi]
    u = theta_deg * np.cos(phi)
    v = theta_deg * np.sin(phi)
    return u, v, mask

def wavelength_to_rgb(wl_nm: np.ndarray) -> np.ndarray:
    """
    可視域(380-780nm)の簡易色。範囲外は黒。
    （見えないものは基本描画しないので、ここは保険）
    """
    wl = wl_nm
    rgb = np.zeros((len(wl), 3), dtype=float)

    # ざっくり分割（簡易）
    # 380-440: violet->blue
    m = (wl >= 380) & (wl < 440)
    rgb[m, 0] = -(wl[m] - 440) / (440 - 380)
    rgb[m, 2] = 1.0

    # 440-490: blue->cyan
    m = (wl >= 440) & (wl < 490)
    rgb[m, 1] = (wl[m] - 440) / (490 - 440)
    rgb[m, 2] = 1.0

    # 490-510: cyan->green
    m = (wl >= 490) & (wl < 510)
    rgb[m, 1] = 1.0
    rgb[m, 2] = -(wl[m] - 510) / (510 - 490)

    # 510-580: green->yellow
    m = (wl >= 510) & (wl < 580)
    rgb[m, 0] = (wl[m] - 510) / (580 - 510)
    rgb[m, 1] = 1.0

    # 580-645: yellow->red
    m = (wl >= 580) & (wl < 645)
    rgb[m, 0] = 1.0
    rgb[m, 1] = -(wl[m] - 645) / (645 - 580)

    # 645-780: red
    m = (wl >= 645) & (wl <= 780)
    rgb[m, 0] = 1.0

    # ガンマっぽく少し強調
    rgb = np.clip(rgb, 0, 1) ** 0.8
    return rgb

def rgb_to_hex(rgb: np.ndarray) -> list[str]:
    rgb255 = (np.clip(rgb, 0, 1) * 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb255]

# =========================
# UI
# =========================
st.markdown("# **STARBOW Simulator（宇宙船中心視点）**")

colL, colR = st.columns([1.05, 2.2], gap="large")

with colL:
    st.markdown("## パラメータ")

    beta = st.slider("v/c", 0.0, 0.99, 0.50, 0.01)
    n_stars = st.slider("星の数", 1000, 20000, 11500, 500)

    seed = st.number_input("配置シード（同じ値で同じ星配置）", value=12345, step=1)

    st.markdown("---")
    drag_3d = st.toggle("ドラッグで見回し（3Dモード）", value=False)  # デフォルトOFF

    st.markdown("## 表示")
    zoom = st.slider("ズーム（大きいほど拡大）", 0.8, 6.0, 2.2, 0.05)
    star_size = st.slider("星の大きさ", 0.5, 6.0, 1.0, 0.1)
    glow = st.slider("グロー強さ", 0.0, 1.0, 0.0, 0.02)

    show_invisible_as_ring = st.toggle("不可視光を表示（白枠）", value=False)

    st.markdown("---")
    if "reset_clicked" not in st.session_state:
        st.session_state.reset_clicked = 0

    if st.button("初期化（向き・ズームを戻す）"):
        st.session_state.reset_clicked += 1
        # 2D用向き
        st.session_state.yaw2d = 0.0
        st.session_state.pitch2d = 0.0
        # 3Dカメラ
        st.session_state.cam3d = None
        # ズーム（スライダーの値自体は戻さない。戻したいならここでst.session_stateにキーを持たせる）
        st.rerun()

with colR:
    if not drag_3d:
        st.markdown(f"## v/c = {beta:.2f} （スライダーで向き）")
        # 2D向き（セッションで保持）
        if "yaw2d" not in st.session_state:
            st.session_state.yaw2d = 0.0
        if "pitch2d" not in st.session_state:
            st.session_state.pitch2d = 0.0

        yaw2d = st.slider("ヨー（左右）", -180.0, 180.0, float(st.session_state.yaw2d), 1.0)
        pitch2d = st.slider("ピッチ（上下）", -89.0, 89.0, float(st.session_state.pitch2d), 1.0)
        st.session_state.yaw2d = yaw2d
        st.session_state.pitch2d = pitch2d
    else:
        st.markdown(f"## v/c = {beta:.2f} （ドラッグで見回し）")
        st.caption("※3Dモードではドラッグで見回し・ホイール/ピンチでズームできます。パラメータを変えても視点を維持します。")

# =========================
# データ生成（星配置・相対論）
# =========================
base_dirs = random_unit_vectors(int(n_stars), int(seed))

# ベース波長：黄色寄りを基準にして、ドップラーで虹化するのが“STARBOWらしい”
base_lambda_nm = 560.0  # 黄色
# 光行差
dirs_ship = aberrate_directions(base_dirs, beta)
# ドップラー因子
D = doppler_factor(dirs_ship, beta)
obs_lambda = base_lambda_nm / D

# 可視判定（可視域のみ基本表示）
visible = (obs_lambda >= 380.0) & (obs_lambda <= 780.0)

# 可視の色
colors_rgb = wavelength_to_rgb(obs_lambda)
colors = rgb_to_hex(colors_rgb)

# =========================
# 目印（方向マーカー）
# =========================
# 宇宙船進行方向系： +x 前, -x 後, +y 右, -y 左, +z 上, -z 下
markers = np.array([
    [ 1.0, 0.0, 0.0],  # +x
    [-1.0, 0.0, 0.0],  # -x
    [ 0.0, 1.0, 0.0],  # +y
    [ 0.0,-1.0, 0.0],  # -y
    [ 0.0, 0.0, 1.0],  # +z
    [ 0.0, 0.0,-1.0],  # -z
], dtype=float)

marker_text = ["+x(前)", "-x(後)", "+y(右)", "-y(左)", "+z(上)", "-z(下)"]

# =========================
# 2Dモード描画
# =========================
def build_2d_fig() -> go.Figure:
    u, v, hemi = to_view_plane_coords(dirs_ship, st.session_state.yaw2d, st.session_state.pitch2d)

    # 2Dは前方半球のみ
    mask2d = hemi

    # 不可視光：OFFなら描画しない（完全に消す）
    if show_invisible_as_ring:
        # 可視は塗り、不可視は白枠
        vis_mask = mask2d & visible
        inv_mask = mask2d & (~visible)
    else:
        vis_mask = mask2d & visible
        inv_mask = np.zeros_like(vis_mask, dtype=bool)

    # ズーム：θ座標(度)なので、表示範囲を縮める
    lim = 90.0 / zoom

    fig = go.Figure()

    # 可視点
    if np.any(vis_mask):
        fig.add_trace(
            go.Scattergl(
                x=u[vis_mask],
                y=v[vis_mask],
                mode="markers",
                marker=dict(
                    size=star_size * 3.0,
                    color=np.array(colors)[vis_mask],
                    opacity=0.9,
                ),
                hoverinfo="skip",
                name="visible",
            )
        )

    # グロー（可視のみ・疑似）
    if glow > 0 and np.any(vis_mask):
        fig.add_trace(
            go.Scattergl(
                x=u[vis_mask],
                y=v[vis_mask],
                mode="markers",
                marker=dict(
                    size=star_size * 9.0 * (1.0 + glow * 2.5),
                    color=np.array(colors)[vis_mask],
                    opacity=min(0.25, 0.08 + glow * 0.25),
                ),
                hoverinfo="skip",
                name="glow",
            )
        )

    # 不可視（白枠）
    if np.any(inv_mask):
        fig.add_trace(
            go.Scattergl(
                x=u[inv_mask],
                y=v[inv_mask],
                mode="markers",
                marker=dict(
                    size=star_size * 3.2,
                    color="rgba(0,0,0,0)",
                    line=dict(color="rgba(255,255,255,0.9)", width=1.2),
                    opacity=0.9,
                ),
                hoverinfo="skip",
                name="invisible",
            )
        )

    # 目印：2Dでは視線座標に回してから、前方半球だけ出す（分かりやすさ優先）
    m_rot = rot_yaw_pitch(markers, -st.session_state.yaw2d, -st.session_state.pitch2d)
    cos_th_m = np.clip(m_rot[:, 0], -1, 1)
    th_m = np.rad2deg(np.arccos(cos_th_m))
    m_mask = th_m <= 90.0
    phi_m = np.arctan2(m_rot[:, 2], m_rot[:, 1])
    mu = th_m * np.cos(phi_m)
    mv = th_m * np.sin(phi_m)

    fig.add_trace(
        go.Scattergl(
            x=mu[m_mask],
            y=mv[m_mask],
            mode="markers+text",
            marker=dict(size=10, color="white"),
            text=np.array(marker_text)[m_mask],
            textposition="top center",
            textfont=dict(color="white", size=12),
            hoverinfo="skip",
            name="markers",
        )
    )

    # リング（30/60/90）
    for rr, w in [(30, 1.2), (60, 1.4), (90, 2.2)]:
        t = np.linspace(0, 2*np.pi, 400)
        fig.add_trace(
            go.Scatter(
                x=rr*np.cos(t),
                y=rr*np.sin(t),
                mode="lines",
                line=dict(color="rgba(255,255,255,0.20)", width=w),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # 十字
    fig.add_trace(
        go.Scatter(
            x=[-90, 90], y=[0, 0],
            mode="lines",
            line=dict(color="rgba(140,180,255,0.35)", width=2),
            hoverinfo="skip",
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0], y=[-90, 90],
            mode="lines",
            line=dict(color="rgba(140,180,255,0.35)", width=2),
            hoverinfo="skip",
            showlegend=False
        )
    )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            visible=False,
            range=[-lim, lim],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            visible=False,
            range=[-lim, lim],
        ),
        showlegend=False,
        # 2Dはズームスライダーを優先したいので uirevision は zoom 依存にする
        uirevision=f"2d-{zoom:.2f}-{st.session_state.reset_clicked}",
    )

    return fig

# =========================
# 3Dモード描画
# =========================
def build_3d_fig() -> go.Figure:
    # 3D表示：単位球面上の点として出す（宇宙船中心視点）
    # 可視/不可視の描画制御
    if show_invisible_as_ring:
        vis_mask = visible
        inv_mask = ~visible
    else:
        vis_mask = visible
        inv_mask = np.zeros_like(visible, dtype=bool)

    # 目印を少し外側へ
    m3 = markers * 1.15

    fig = go.Figure()

    # グロー（3D）: plotly 3Dでブラーはできないので「大きい半透明点」で疑似
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

    # 可視点
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

    # 不可視点（白枠）※OFFならそもそも作らない
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
                    line=dict(color="rgba(255,255,255,0.95)", width=2),  # ←必ずスカラー
                ),
                hoverinfo="skip",
                name="invisible",
            )
        )

    # 目印
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

    # 「宇宙船位置」(原点) も目印として追加
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

    # レイアウト：カメラ維持が肝（uirevisionを固定）
    # 3Dのズームは plotly 側操作も許可しつつ、スライダーzoomで「見た目のスケール」を合わせる
    # → ここでは球の半径表示範囲を zoom に応じて狭める
    lim = 1.25 / zoom

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
            # ここが重要：視点を「外から俯瞰」じゃなく、なるべく中心に近い感じにする
            camera=dict(
                eye=dict(x=0.25, y=0.25, z=0.25),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
        showlegend=False,
        # これでパラメータ変更してもPlotly側のカメラ状態が維持される
        uirevision="3d-keep-camera",
    )

    return fig

# =========================
# 描画
# =========================
if drag_3d:
    fig = build_3d_fig()
    # on_select は使わない（Streamlit版によってAPI例外になるため）
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=dict(
            scrollZoom=True,  # ピンチ/ホイールズーム
            displaylogo=False,
        ),
    )
else:
    fig = build_2d_fig()
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=dict(
            scrollZoom=False,  # 2Dはズームスライダーで統制
            displaylogo=False,
        ),
    )

st.caption("※不可視光は「不可視光を表示（白枠）」OFF のとき完全に描画しません。ONで白枠として表示します。")
