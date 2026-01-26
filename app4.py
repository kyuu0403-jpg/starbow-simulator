import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# Utilities
# =========================
def rot_yaw_pitch(vecs, yaw_deg: float, pitch_deg: float):
    """Rotate vectors by yaw (around z) then pitch (around y)."""
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)

    Rz = np.array([[cz, -sz, 0.0],
                   [sz,  cz, 0.0],
                   [0.0, 0.0, 1.0]])
    Ry = np.array([[ cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]])

    R = Ry @ Rz
    return vecs @ R.T


def sample_uniform_sphere(n: int, rng: np.random.Generator):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def aberrate_dirs(n_rest: np.ndarray, beta: float):
    """
    Aberration for observer moving +x (beta along +x).
    Using standard formula:
      n'_x = (n_x + beta) / (1 + beta n_x)
      n'_{y,z} = n_{y,z} / (gamma (1 + beta n_x))
    """
    if beta <= 0:
        return n_rest.copy()

    bx = beta
    nx, ny, nz = n_rest[:, 0], n_rest[:, 1], n_rest[:, 2]
    den = (1.0 + bx * nx)
    gamma = 1.0 / np.sqrt(1.0 - bx * bx)

    npx = (nx + bx) / den
    npy = ny / (gamma * den)
    npz = nz / (gamma * den)

    nprime = np.stack([npx, npy, npz], axis=1)
    nprime /= np.linalg.norm(nprime, axis=1, keepdims=True)
    return nprime


def doppler_factor(n_rest: np.ndarray, beta: float):
    """Doppler factor D = gamma (1 + beta n_x) for observer moving +x."""
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    return gamma * (1.0 + beta * n_rest[:, 0])


def wavelength_to_rgb_approx(lam_nm: np.ndarray):
    """
    Rough visible-spectrum mapping (380–780nm).
    Returns RGB in [0,255] and a visibility mask.
    """
    lam = lam_nm.copy()
    vis = (lam >= 380.0) & (lam <= 780.0)

    # Normalize to [0,1] within visible
    t = np.clip((lam - 380.0) / (780.0 - 380.0), 0.0, 1.0)

    # Simple piecewise hue-ish mapping (not physically perfect, but looks good)
    r = np.zeros_like(t)
    g = np.zeros_like(t)
    b = np.zeros_like(t)

    # 380-440: violet -> blue
    m = (lam >= 380) & (lam < 440)
    r[m] = (440 - lam[m]) / (440 - 380)
    g[m] = 0
    b[m] = 1

    # 440-490: blue -> cyan
    m = (lam >= 440) & (lam < 490)
    r[m] = 0
    g[m] = (lam[m] - 440) / (490 - 440)
    b[m] = 1

    # 490-510: cyan -> green
    m = (lam >= 490) & (lam < 510)
    r[m] = 0
    g[m] = 1
    b[m] = (510 - lam[m]) / (510 - 490)

    # 510-580: green -> yellow
    m = (lam >= 510) & (lam < 580)
    r[m] = (lam[m] - 510) / (580 - 510)
    g[m] = 1
    b[m] = 0

    # 580-645: yellow -> red
    m = (lam >= 580) & (lam < 645)
    r[m] = 1
    g[m] = (645 - lam[m]) / (645 - 580)
    b[m] = 0

    # 645-780: red
    m = (lam >= 645) & (lam <= 780)
    r[m] = 1
    g[m] = 0
    b[m] = 0

    # Intensity falloff near edges (nice look)
    intensity = np.ones_like(t)
    m = (lam >= 380) & (lam < 420)
    intensity[m] = 0.3 + 0.7 * (lam[m] - 380) / (420 - 380)
    m = (lam > 700) & (lam <= 780)
    intensity[m] = 0.3 + 0.7 * (780 - lam[m]) / (780 - 700)

    r = np.clip(r * intensity, 0, 1)
    g = np.clip(g * intensity, 0, 1)
    b = np.clip(b * intensity, 0, 1)

    rgb = np.stack([r, g, b], axis=1) * 255.0
    return rgb.astype(np.int32), vis


def rgba_strings(rgb255: np.ndarray, alpha: float):
    a = np.clip(alpha, 0.0, 1.0)
    return [f"rgba({r},{g},{b},{a})" for r, g, b in rgb255]


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="STARBOW Simulator", layout="wide")

# --- CSS: dark sky-like background & white text ---
st.markdown(
    """
    <style>
      .stApp { background: radial-gradient(circle at 30% 20%, #0b1630 0%, #05060f 55%, #000 100%); }
      h1, h2, h3, p, label, span, div { color: #ffffff !important; }
      .stSlider label { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("STARBOW Simulator（宇宙船中心視点）")

# Defaults in session_state
if "mode3d" not in st.session_state:
    st.session_state.mode3d = False  # 最初OFF
if "cam" not in st.session_state:
    # 3Dカメラ：宇宙船中心（原点）から“外側を見回す”ように近距離に置く
    st.session_state.cam = dict(
        eye=dict(x=0.0, y=-0.01, z=0.0),  # ほぼ原点（Plotlyは完全0だと不安定なので少しだけ）
        center=dict(x=0.0, y=0.0, z=0.0),
        up=dict(x=0.0, y=0.0, z=1.0)
    )

# Layout columns
left, right = st.columns([0.32, 0.68], gap="large")

with left:
    st.header("パラメータ")

    beta = st.slider("v/c", 0.0, 0.99, 0.50, 0.01)
    nstars = st.slider("星の数", 200, 20000, 11500, 100)
    seed = st.number_input("配置シード（同じ値で同じ星配置）", value=12345, step=1)

    st.session_state.mode3d = st.toggle("ドラッグで見回し（3Dモード）", value=st.session_state.mode3d)

    st.header("表示")
    zoom = st.slider("ズーム（大きいほど拡大）", 0.8, 6.0, 2.2, 0.05)
    star_size = st.slider("星の大きさ", 0.5, 6.0, 1.0, 0.1)
    glow = st.slider("グロー強さ", 0.0, 1.0, 0.45, 0.01)
    show_invisible_white = st.toggle("不可視光を白枠で表示", value=False)

    # 2D view direction sliders (used when 3D OFF)
    st.header("視点（2D：スライダーで向き）")
    yaw2d = st.slider("ヨー（左右）", -180.0, 180.0, 0.0, 1.0, disabled=st.session_state.mode3d)
    pitch2d = st.slider("ピッチ（上下）", -89.0, 89.0, 0.0, 1.0, disabled=st.session_state.mode3d)

    # Reset button (always visible text)
    if st.button("初期位置に戻す（視点・ズーム）", use_container_width=True):
        st.session_state.cam = dict(
            eye=dict(x=0.0, y=-0.01, z=0.0),
            center=dict(x=0.0, y=0.0, z=0.0),
            up=dict(x=0.0, y=0.0, z=1.0)
        )
        # also reset sliders via session state keys if needed
        # (Streamlit slider reset is tricky without keys; user can move them back easily)

with right:
    title_suffix = "（ドラッグで見回し）" if st.session_state.mode3d else "（スライダーで向き）"
    st.subheader(f"v/c = {beta:.2f}  {title_suffix}")

# =========================
# Data generation (cached-ish)
# =========================
rng = np.random.default_rng(int(seed))
n_rest = sample_uniform_sphere(int(nstars), rng=rng)

# base wavelength (unified yellow-ish, per your request earlier)
lam0 = 580.0  # nm

# Doppler shift (observer frame): lambda' = lambda0 / D
D = doppler_factor(n_rest, beta)
lam_obs = lam0 / D

rgb, visible = wavelength_to_rgb_approx(lam_obs)

# If invisible: background color or optional white outline
# We'll draw visible as filled glow dots; invisible optionally as transparent fill + white outline
alpha_visible = 0.72
alpha_glow = np.clip(glow, 0.0, 1.0) * 0.55

# Apply aberration to directions
n_obs = aberrate_dirs(n_rest, beta)

# =========================
# Render
# =========================
bg_plot = "rgba(0,0,0,0)"  # let page background show through

# ---- 3D mode ----
if st.session_state.mode3d:
    # NOTE: 3D is already "perfect" per you, so we keep behavior and ONLY ensure camera persists.
    # Use plotly 3D, render points on unit sphere directions (center viewpoint vibe)
    # We'll map direction vectors directly (unit sphere).
    x, y, z = n_obs[:, 0], n_obs[:, 1], n_obs[:, 2]

    # Glow layer (bigger & more transparent)
    glow_colors = rgba_strings(rgb, alpha_glow)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        name="glow",
        marker=dict(
            size=star_size * 6.0,
            color=glow_colors,
            line=dict(width=0)
        ),
        hoverinfo="skip"
    ))

    # Core layer
    core_colors = rgba_strings(rgb, alpha_visible)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        name="stars",
        marker=dict(
            size=star_size * 2.6,
            color=core_colors,
            line=dict(width=0)
        ),
        hoverinfo="skip"
    ))

    # Invisible stars as white-outline only (optional)
    if show_invisible_white:
        inv = ~visible
        if np.any(inv):
            fig.add_trace(go.Scatter3d(
                x=x[inv], y=y[inv], z=z[inv],
                mode="markers",
                name="invisible",
                marker=dict(
                    size=star_size * 2.8,
                    color="rgba(0,0,0,0)",
                    line=dict(color="rgba(255,255,255,0.75)", width=2)
                ),
                hoverinfo="skip"
            ))

    # Direction markers (front/back/up/down/left/right + center)
    # Define in ship frame: +x front, -x back, +y right, -y left, +z up, -z down
    mpos = np.array([
        [ 1.0,  0.0,  0.0],  # +x
        [-1.0,  0.0,  0.0],  # -x
        [ 0.0,  1.0,  0.0],  # +y
        [ 0.0, -1.0,  0.0],  # -y
        [ 0.0,  0.0,  1.0],  # +z
        [ 0.0,  0.0, -1.0],  # -z
        [ 0.0,  0.0,  0.0],  # origin
    ])
    mtxt = ["+x(前)", "-x(後)", "+y(右)", "-y(左)", "+z(上)", "-z(下)", "原点"]

    fig.add_trace(go.Scatter3d(
        x=mpos[:, 0], y=mpos[:, 1], z=mpos[:, 2],
        mode="markers+text",
        text=mtxt,
        textposition="top center",
        marker=dict(size=8, color="rgba(255,255,255,0.95)"),
        hoverinfo="skip",
        showlegend=False
    ))

    # Layout: important bits
    fig.update_layout(
        paper_bgcolor=bg_plot,
        plot_bgcolor=bg_plot,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
            camera=st.session_state.cam,
            bgcolor="rgba(0,0,0,0)",
        ),
        # uirevision keeps user camera when other params change
        uirevision="KEEP_3D_CAMERA"
    )

    # Capture camera changes to session_state (prevents reset when sliders move)
    event = st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True}, on_select=None)

    # Streamlit doesn't directly return relayout events here,
    # but plotly uirevision already keeps camera stable across reruns.

    with right:
        st.caption("※3Dモードはドラッグで見回し。パラメータ変更しても視点は維持されます。")

# ---- 2D mode ----
else:
    # 2D fisheye projection of front hemisphere after applying a viewing rotation.
    # Key fix: DO NOT allow Plotly autoscale; axis ranges are fixed by zoom ONLY.
    v = rot_yaw_pitch(n_obs, yaw2d, pitch2d)

    # front hemisphere relative to current view direction (+x forward? we use +x as "front" in marker labels,
    # but for 2D we want screen center = +x (forward). Let's set "forward" to +x.
    # So project around +x axis: angle from +x.
    # Build coordinates in a basis where forward is +x, right is +y, up is +z.
    # For fisheye: theta = arccos(vx), r = theta/(pi/2) for theta<=pi/2
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    theta = np.arccos(np.clip(vx, -1.0, 1.0))
    front = theta <= (np.pi / 2)

    # azimuth around +x axis
    phi = np.arctan2(vz, vy)  # using (right=+y, up=+z)

    r = theta / (np.pi / 2)  # 0..1
    x2 = r * np.cos(phi)
    y2 = r * np.sin(phi)

    # keep only front
    x2, y2 = x2[front], y2[front]
    rgb2, vis2 = rgb[front], visible[front]

    # Colors
    glow_colors = np.array(rgba_strings(rgb2, alpha_glow))
    core_colors = np.array(rgba_strings(rgb2, alpha_visible))

    fig2 = go.Figure()

    # Glow
    fig2.add_trace(go.Scattergl(
        x=x2, y=y2,
        mode="markers",
        marker=dict(size=star_size * 18.0, color=glow_colors, line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False
    ))

    # Core
    fig2.add_trace(go.Scattergl(
        x=x2, y=y2,
        mode="markers",
        marker=dict(size=star_size * 6.5, color=core_colors, line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False
    ))

    # Invisible stars (optional white-outline)
    if show_invisible_white:
        inv = ~vis2
        if np.any(inv):
            fig2.add_trace(go.Scattergl(
                x=x2[inv], y=y2[inv],
                mode="markers",
                marker=dict(
                    size=star_size * 7.0,
                    color="rgba(0,0,0,0)",
                    line=dict(color="rgba(255,255,255,0.75)", width=2)
                ),
                hoverinfo="skip",
                showlegend=False
            ))

    # Direction markers in 2D (for current view):
    # forward is center (0,0). Place six markers around circle at r=1.
    # Using our phi definition (x=r cos phi, y=r sin phi):
    # +y(right) corresponds to phi=0 -> (1,0)
    # +z(up) corresponds to phi=pi/2 -> (0,1)
    # -y(left) corresponds to phi=pi -> (-1,0)
    # -z(down) corresponds to phi=-pi/2 -> (0,-1)
    # Back (-x) is not in front hemisphere; we can still show a hint near edge with text.
    m2 = np.array([
        [ 1.0,  0.0],   # +y right
        [-1.0,  0.0],   # -y left
        [ 0.0,  1.0],   # +z up
        [ 0.0, -1.0],   # -z down
    ])
    t2 = ["+y(右)", "-y(左)", "+z(上)", "-z(下)"]

    fig2.add_trace(go.Scattergl(
        x=m2[:, 0], y=m2[:, 1],
        mode="markers+text",
        text=t2,
        textposition="top center",
        marker=dict(size=10, color="rgba(255,255,255,0.9)"),
        hoverinfo="skip",
        showlegend=False
    ))

    # Forward marker at center
    fig2.add_trace(go.Scattergl(
        x=[0.0], y=[0.0],
        mode="markers+text",
        text=["+x(前)"],
        textposition="bottom center",
        marker=dict(size=10, color="rgba(255,255,255,0.95)"),
        hoverinfo="skip",
        showlegend=False
    ))

    # Circle boundary at 90deg (r=1)
    ang = np.linspace(0, 2*np.pi, 400)
    fig2.add_trace(go.Scattergl(
        x=np.cos(ang), y=np.sin(ang),
        mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=2),
        hoverinfo="skip",
        showlegend=False
    ))

    # --- THE FIX: 2D zoom is ONLY controlled by 'zoom' slider, not by beta ---
    # Set axis ranges explicitly; disable autorange
    lim = 1.05 / zoom  # zoom↑ => lim↓ => magnify
    fig2.update_layout(
        paper_bgcolor=bg_plot,
        plot_bgcolor=bg_plot,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            visible=False,
            range=[-lim, lim],
            autorange=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            visible=False,
            range=[-lim, lim],
            autorange=False,
        ),
        # This prevents Plotly from "helpfully" changing scale when points concentrate at high beta
        uirevision="KEEP_2D_VIEW"
    )

    # Put title only as v/c
    fig2.update_layout(title=dict(text=f"v/c = {beta:.2f}", x=0.5, y=0.98, xanchor="center", font=dict(size=28)))

    with right:
        st.plotly_chart(fig2, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})
        st.caption("※2Dモードは速度を変えても勝手にズームしません（ズームはスライダーのみで決まります）。")
