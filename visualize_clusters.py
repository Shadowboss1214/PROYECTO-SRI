"""
CineMatch — visualize_clusters.py
===================================
Visualización interactiva de clusters K-Means en tiempo real.

- Usuarios    → puntos de colores (uno por cluster)
- Centroides  → diamantes ROJOS con etiqueta
- Reducción:  PCA 2D o 3D

Uso:
    python visualize_clusters.py              → 2D estático
    python visualize_clusters.py --3d         → 3D interactivo
    python visualize_clusters.py --live       → dashboard tiempo real (30s)
    python visualize_clusters.py --3d --live  → 3D en tiempo real
    python visualize_clusters.py --live --interval 60  → actualiza cada 60s

Requisitos:
    pip install plotly dash scikit-learn numpy
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection

GENRE_KEYS = [
    "Action", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy",
    "Foreign", "History", "Horror", "Music", "Mystery",
    "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"
]

CLUSTER_COLORS = [
    "#4299e1","#48bb78","#ed8936","#9f7aea","#f56565",
    "#38b2ac","#ecc94b","#fc8181","#68d391","#76e4f7",
    "#b794f4","#fbb6ce","#90cdf4","#9ae6b4","#faf089",
    "#feb2b2","#bee3f8","#c6f6d5","#fefcbf","#fed7d7",
    "#e9d8fd","#d6bcfa","#c3dafe","#b2f5ea","#feebc8",
    "#e2e8f0","#cbd5e0","#a0aec0","#718096","#4a5568","#2d3748",
]


# =============================================================================
#  LECTURA DE DATOS
# =============================================================================

def fetch_data(conn) -> dict:
    cur = conn.cursor()

    cur.execute("""
        SELECT
            cs.session_id,
            cs.taste_vector,
            cs.country_code,
            cs.cold_start_done,
            cs.last_seen,
            sc.cluster_id,
            sc.distance_to_centroid,
            c.cluster_number
        FROM cookie_sessions cs
        LEFT JOIN session_cluster sc ON sc.session_id = cs.session_id
        LEFT JOIN clusters c ON c.cluster_id = sc.cluster_id AND c.is_active = TRUE
        WHERE cs.taste_vector != '{}'::jsonb
          AND cs.taste_vector IS NOT NULL
        ORDER BY cs.last_seen DESC
    """)
    user_rows = cur.fetchall()

    cur.execute("""
        SELECT cluster_id, cluster_number, centroid_vector,
               member_count, version, computed_at
        FROM   clusters
        WHERE  is_active = TRUE
        ORDER  BY cluster_number
    """)
    centroid_rows = cur.fetchall()

    cur.execute("SELECT COUNT(*) AS c FROM cookie_sessions")
    total_sessions = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) AS c FROM ratings")
    total_ratings = cur.fetchone()["c"]

    cur.execute("SELECT COALESCE(MAX(version),0) AS v FROM clusters WHERE is_active=TRUE")
    cluster_version = cur.fetchone()["v"]

    return {
        "users":           user_rows,
        "centroids":       centroid_rows,
        "total_sessions":  total_sessions,
        "total_ratings":   total_ratings,
        "cluster_version": cluster_version,
        "fetched_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# =============================================================================
#  CONSTRUCCIÓN DE MATRICES
# =============================================================================

def build_matrices(data: dict) -> dict:
    user_vectors   = []
    user_labels    = []
    user_ids       = []
    user_countries = []
    user_distances = []
    user_is_real   = []

    for row in data["users"]:
        tv = row["taste_vector"]
        if isinstance(tv, str):
            try:    tv = json.loads(tv)
            except: continue
        if not tv:
            continue
        vec = np.array([tv.get(g, 0.0) for g in GENRE_KEYS], dtype=np.float32)
        if vec.sum() == 0:
            continue
        user_vectors.append(vec)
        user_labels.append(row["cluster_number"] if row["cluster_number"] is not None else -1)
        user_ids.append(str(row["session_id"])[:8] + "...")
        user_countries.append(row["country_code"] or "??")
        user_distances.append(round(float(row["distance_to_centroid"] or 0), 4))
        user_is_real.append(bool(row["cold_start_done"]))

    centroid_vectors = []
    centroid_labels  = []
    centroid_counts  = []
    centroid_genres  = []

    for row in data["centroids"]:
        cv = row["centroid_vector"]
        if isinstance(cv, str):
            try:    cv = json.loads(cv)
            except: continue
        vec = np.array([cv.get(g, 0.0) for g in GENRE_KEYS], dtype=np.float32)
        centroid_vectors.append(vec)
        centroid_labels.append(row["cluster_number"])
        centroid_counts.append(row["member_count"])
        top3 = sorted(cv.items(), key=lambda x: x[1], reverse=True)[:3]
        centroid_genres.append(", ".join(f"{g}({v:.2f})" for g, v in top3))

    return {
        "user_matrix":     np.array(user_vectors)    if user_vectors    else np.empty((0, 21)),
        "user_labels":     user_labels,
        "user_ids":        user_ids,
        "user_countries":  user_countries,
        "user_distances":  user_distances,
        "user_is_real":    user_is_real,
        "centroid_matrix": np.array(centroid_vectors) if centroid_vectors else np.empty((0, 21)),
        "centroid_labels": centroid_labels,
        "centroid_counts": centroid_counts,
        "centroid_genres": centroid_genres,
    }


# =============================================================================
#  PCA
# =============================================================================

def apply_pca(matrices: dict, n_components: int = 2) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    u_mat = matrices["user_matrix"]
    c_mat = matrices["centroid_matrix"]

    parts = [m for m in [u_mat, c_mat] if len(m) > 0]
    if not parts:
        return {
            "user_proj":          np.empty((0, n_components)),
            "centroid_proj":      np.empty((0, n_components)),
            "explained_variance": [0.0] * n_components,
            "n_components":       n_components,
        }

    combined      = np.vstack(parts)
    # Evitar dividir por cero en normalize
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1
    combined_norm = combined / norms

    real_components = min(n_components, combined_norm.shape[0], combined_norm.shape[1])
    pca = PCA(n_components=real_components, random_state=42)
    pca.fit(combined_norm)

    ev = list(pca.explained_variance_ratio_)
    while len(ev) < n_components:
        ev.append(0.0)

    def project(mat):
        if len(mat) == 0:
            return np.empty((0, n_components))
        n = np.linalg.norm(mat, axis=1, keepdims=True)
        n[n == 0] = 1
        proj = pca.transform(mat / n)
        # Rellenar si real_components < n_components
        if proj.shape[1] < n_components:
            pad  = np.zeros((proj.shape[0], n_components - proj.shape[1]))
            proj = np.hstack([proj, pad])
        return proj

    print(f"   → PCA varianza explicada: "
          f"{', '.join(f'PC{i+1}={v*100:.1f}%' for i,v in enumerate(ev[:n_components]))}")
    print(f"   → Total: {sum(ev[:n_components])*100:.1f}%")

    return {
        "user_proj":          project(u_mat),
        "centroid_proj":      project(c_mat),
        "explained_variance": ev,
        "n_components":       n_components,
    }


# =============================================================================
#  FIGURA PLOTLY
# =============================================================================

def build_figure(data: dict, matrices: dict, pca_result: dict, mode_3d: bool = False):
    import plotly.graph_objects as go

    u_proj = pca_result["user_proj"]
    c_proj = pca_result["centroid_proj"]
    ev     = pca_result["explained_variance"]

    traces = []
    k      = len(matrices["centroid_labels"])

    # ── Usuarios por cluster ──────────────────────────────────────────────────
    if len(u_proj) > 0:
        for cluster_num in sorted(set(matrices["user_labels"])):
            mask  = [i for i, l in enumerate(matrices["user_labels"]) if l == cluster_num]
            if not mask:
                continue
            color = CLUSTER_COLORS[cluster_num % len(CLUSTER_COLORS)] if cluster_num >= 0 else "#a0aec0"
            label = f"Cluster {cluster_num}" if cluster_num >= 0 else "Sin cluster"

            hover = [
                f"<b>Usuario</b>: {matrices['user_ids'][i]}<br>"
                f"País: {matrices['user_countries'][i]}<br>"
                f"Cluster: {cluster_num}<br>"
                f"Dist. centroide: {matrices['user_distances'][i]}<br>"
                f"Tipo: {'✓ Real' if matrices['user_is_real'][i] else 'Demo'}"
                for i in mask
            ]

            if mode_3d and u_proj.shape[1] >= 3:
                traces.append(go.Scatter3d(
                    x=[u_proj[i,0] for i in mask],
                    y=[u_proj[i,1] for i in mask],
                    z=[u_proj[i,2] for i in mask],
                    mode="markers", name=label,
                    marker=dict(size=4, color=color, opacity=0.75,
                                line=dict(width=0.5, color="white")),
                    text=hover, hoverinfo="text",
                ))
            else:
                traces.append(go.Scatter(
                    x=[u_proj[i,0] for i in mask],
                    y=[u_proj[i,1] for i in mask],
                    mode="markers", name=label,
                    marker=dict(size=8, color=color, opacity=0.75,
                                line=dict(width=0.5, color="white")),
                    text=hover, hoverinfo="text",
                ))

    # ── Centroides ────────────────────────────────────────────────────────────
    if len(c_proj) > 0:
        c_hover = [
            f"<b>⬟ CENTROIDE {cn}</b><br>"
            f"Miembros: {matrices['centroid_counts'][i]}<br>"
            f"Top géneros: {matrices['centroid_genres'][i]}"
            for i, cn in enumerate(matrices["centroid_labels"])
        ]

        if mode_3d and c_proj.shape[1] >= 3:
            traces.append(go.Scatter3d(
                x=c_proj[:,0], y=c_proj[:,1], z=c_proj[:,2],
                mode="markers+text", name="⬟ Centroides",
                marker=dict(size=10, color="red", symbol="diamond",
                            line=dict(width=2, color="darkred")),
                text=[f"C{cn}" for cn in matrices["centroid_labels"]],
                textposition="top center",
                textfont=dict(size=11, color="red"),
                hovertext=c_hover, hoverinfo="text",
            ))
        else:
            traces.append(go.Scatter(
                x=c_proj[:,0], y=c_proj[:,1],
                mode="markers+text", name="⬟ Centroides",
                marker=dict(size=18, color="red", symbol="diamond",
                            line=dict(width=2, color="darkred")),
                text=[f"C{cn}" for cn in matrices["centroid_labels"]],
                textposition="top center",
                textfont=dict(size=11, color="red"),
                hovertext=c_hover, hoverinfo="text",
            ))

    # ── Layout ────────────────────────────────────────────────────────────────
    n_users = len(u_proj)
    subtitle = (
        f"Usuarios: {n_users} | Centroides: {len(c_proj)} (rojo ⬟) | "
        f"K={k} | Versión: {data['cluster_version']} | "
        f"Ratings: {data['total_ratings']:,} | {data['fetched_at']}"
    )
    axis = [
        f"PC1 ({ev[0]*100:.1f}%)" if len(ev) > 0 else "PC1",
        f"PC2 ({ev[1]*100:.1f}%)" if len(ev) > 1 else "PC2",
        f"PC3 ({ev[2]*100:.1f}%)" if len(ev) > 2 else "PC3",
    ]

    dark = dict(paper_bgcolor="#0a0a0f", plot_bgcolor="#0f0f17",
                font=dict(color="#f0ede8"))

    if mode_3d:
        layout = go.Layout(
            **dark,
            title=dict(text=f"<b>CineMatch — Clusters K-Means (3D)</b><br><sup>{subtitle}</sup>",
                       font=dict(size=15)),
            scene=dict(
                xaxis=dict(title=axis[0], gridcolor="#2d2d3d", color="#aaa"),
                yaxis=dict(title=axis[1], gridcolor="#2d2d3d", color="#aaa"),
                zaxis=dict(title=axis[2], gridcolor="#2d2d3d", color="#aaa"),
                bgcolor="#0f0f17",
            ),
            legend=dict(bgcolor="#16161f", bordercolor="#333", borderwidth=1),
            margin=dict(l=0, r=0, t=80, b=0),
        )
    else:
        layout = go.Layout(
            **dark,
            title=dict(text=f"<b>CineMatch — Clusters K-Means (2D)</b><br><sup>{subtitle}</sup>",
                       font=dict(size=15)),
            xaxis=dict(title=axis[0], gridcolor="#2d2d3d", zeroline=False, color="#aaa"),
            yaxis=dict(title=axis[1], gridcolor="#2d2d3d", zeroline=False, color="#aaa"),
            legend=dict(bgcolor="#16161f", bordercolor="#333", borderwidth=1),
            hovermode="closest",
            margin=dict(l=60, r=20, t=80, b=60),
        )

    return go.Figure(data=traces, layout=layout)


def empty_figure(message: str):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0f0f17",
        font=dict(color="#f0ede8"),
        annotations=[dict(
            text=message, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color="#7a7888")
        )]
    )
    return fig


# =============================================================================
#  PIPELINE COMPLETO (fetch → matrices → PCA → figura)
# =============================================================================

def build_full_figure(mode_3d: bool) -> tuple:
    """
    Conecta a Supabase, lee datos y construye la figura.
    Devuelve (figure, status_text).
    """
    try:
        conn      = get_connection()
        data      = fetch_data(conn)
        matrices  = build_matrices(data)
        conn.close()

        n_users     = len(matrices["user_matrix"])
        n_centroids = len(matrices["centroid_matrix"])

        if n_users == 0 and n_centroids == 0:
            return (
                empty_figure("Sin datos — ejecuta retrain_clusters.py primero"),
                "⚠️  Sin datos para graficar"
            )

        n_comp  = 3 if mode_3d else 2
        pca_res = apply_pca(matrices, n_components=n_comp)
        fig     = build_figure(data, matrices, pca_res, mode_3d=mode_3d)

        status = (
            f"✓ {data['fetched_at']} | "
            f"Usuarios: {n_users} | "
            f"Centroides: {n_centroids} | "
            f"Ratings: {data['total_ratings']:,} | "
            f"Versión cluster: {data['cluster_version']}"
        )
        return fig, status

    except Exception as e:
        import traceback
        traceback.print_exc()
        return empty_figure(f"Error: {str(e)[:120]}"), f"❌ {str(e)[:100]}"


# =============================================================================
#  MODO LIVE — Dashboard Dash
# =============================================================================

def run_live_dashboard(mode_3d: bool = False, interval_sec: int = 30):
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
    except ImportError:
        print("❌ Instala dash:  pip install dash")
        sys.exit(1)

    app = dash.Dash(__name__, title="CineMatch Clusters")
    app.layout = html.Div([
        html.Div([
            html.H2(
                "🎬 CineMatch — Clusters K-Means en Tiempo Real",
                style={"color":"#e8c547","fontFamily":"sans-serif",
                       "margin":"0","padding":"16px 24px"}
            ),
            html.Span(
                f"Actualización automática cada {interval_sec}s  |  "
                f"{'3D' if mode_3d else '2D'}  |  "
                "Rojo ⬟ = Centroide  |  Colores = Clusters de usuarios",
                style={"color":"#7a7888","paddingLeft":"24px","fontSize":"13px"}
            ),
        ], style={"background":"#111118","borderBottom":"1px solid #222",
                  "paddingBottom":"12px"}),

        dcc.Graph(
            id="cluster-graph",
            style={"height":"85vh"},
            config={"displayModeBar":True,"scrollZoom":True}
        ),

        dcc.Interval(
            id="auto-update",
            interval=interval_sec * 1000,
            n_intervals=0
        ),

        html.Div(
            id="status-bar",
            style={"background":"#111118","color":"#7a7888",
                   "padding":"8px 24px","fontSize":"12px","fontFamily":"monospace"}
        ),
    ], style={"background":"#0a0a0f","minHeight":"100vh","margin":"0"})

    @app.callback(
        [Output("cluster-graph","figure"),
         Output("status-bar","children")],
        [Input("auto-update","n_intervals")]
    )
    def update(n):
        fig, status = build_full_figure(mode_3d)
        return fig, f"{status}  |  Actualización #{n+1}"

    print(f"\n🚀 Dashboard en:  http://localhost:8050")
    print(f"   Actualización cada {interval_sec}s | Ctrl+C para detener\n")
    app.run(debug=False, host="0.0.0.0", port=8050)


# =============================================================================
#  MODO ESTÁTICO
# =============================================================================

def run_static(mode_3d: bool = False):
    import plotly.io as pio

    print("\n📊 CineMatch — Visualización de Clusters")
    print("=" * 50)
    print("\n[1] Conectando y leyendo datos...")
    fig, status = build_full_figure(mode_3d)
    print(f"\n{status}")
    print("\n[2] Abriendo en el navegador...")
    pio.show(fig)
    print("\n   Rojo ⬟ = Centroide | Colores = Clusters de usuarios")
    print("   Pasa el cursor sobre los puntos para ver detalles.\n")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CineMatch — Visualización interactiva de clusters K-Means"
    )
    parser.add_argument("--3d",      dest="mode_3d",  action="store_true",
                        help="Gráfica 3D (PC1, PC2, PC3)")
    parser.add_argument("--live",    action="store_true",
                        help="Dashboard en tiempo real en localhost:8050")
    parser.add_argument("--interval",type=int, default=30,
                        help="Segundos entre actualizaciones en --live (default: 30)")
    args = parser.parse_args()

    if args.live:
        run_live_dashboard(mode_3d=args.mode_3d, interval_sec=args.interval)
    else:
        run_static(mode_3d=args.mode_3d)


if __name__ == "__main__":
    main()