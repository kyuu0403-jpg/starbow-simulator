# =========================
# 2D描画
# =========================
def build_2d_fig() -> go.Figure:
    u, v, hemi = to_view_plane_coords(
        dirs_ship, st.session_state.yaw2d, st.session_state.pitch2d
    )

    mask2d = hemi

    if show_invisible_as_ring:
        vis_mask = mask2d & visible
        inv_mask = mask2d & (~visible)
    else:
        vis_mask = mask2d & visible
        inv_mask = np.zeros_like(mask2d, dtype=bool)  # ←完全に描画しない

    # 2Dズーム：スライダーだけで決まる「固定レンジ」
    # zoomが大きいほど拡大 → 表示範囲は小さく
    lim = 90.0 / float(zoom)

    fig = go.Figure()

    # 可視
    if np.any(vis_mask):
        fig.add_trace(
            go.Scattergl(
                x=u[vis_mask],
                y=v[vis_mask],
                mode="markers",
                marker=dict(
                    size=float(star_size * 3.0),
                    color=np.array(colors)[vis_mask],
                    opacity=0.9,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # グロー
    if glow > 0 and np.any(vis_mask):
        fig.add_trace(
            go.Scattergl(
                x=u[vis_mask],
                y=v[vis_mask],
                mode="markers",
                marker=dict(
                    size=float(star_size * 9.0 * (1.0 + glow * 2.5)),
                    color=np.array(colors)[vis_mask],
                    opacity=float(min(0.25, 0.08 + glow * 0.25)),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 不可視（白枠ONのときだけ）
    if np.any(inv_mask):
        fig.add_trace(
            go.Scattergl(
                x=u[inv_mask],
                y=v[inv_mask],
                mode="markers",
                marker=dict(
                    size=float(star_size * 3.2),
                    color="rgba(0,0,0,0)",
                    line=dict(color="rgba(255,255,255,0.9)", width=1.2),
                    opacity=0.9,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 目印（前方半球のみ）
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
            showlegend=False,
        )
    )

    # リング（30/60/90）
    for rr, w in [(30, 1.2), (60, 1.4), (90, 2.2)]:
        t = np.linspace(0, 2 * np.pi, 400)
        fig.add_trace(
            go.Scatter(
                x=rr * np.cos(t),
                y=rr * np.sin(t),
                mode="lines",
                line=dict(color="rgba(255,255,255,0.20)", width=w),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 十字
    fig.add_trace(
        go.Scatter(
            x=[-90, 90],
            y=[0, 0],
            mode="lines",
            line=dict(color="rgba(140,180,255,0.35)", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-90, 90],
            mode="lines",
            line=dict(color="rgba(140,180,255,0.35)", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # ★ここが肝：オートスケールを絶対させない（ズームはrangeだけ）
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(
            visible=False,
            range=[-lim, lim],
            autorange=False,
            fixedrange=True,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            visible=False,
            range=[-lim, lim],
            autorange=False,
            fixedrange=True,
        ),
        # ズーム変更時は必ずレンジを更新したいので zoom を含める
        uirevision=f"2d-{float(zoom):.2f}-{st.session_state.reset_clicked}",
    )

    return fig
