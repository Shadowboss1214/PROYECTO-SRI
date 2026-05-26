"""
CineMatch — clean_demo_data.py
================================
Elimina todos los datos del dataset MovieLens (demo)
y conserva únicamente los datos reales de usuarios de la app.

Qué elimina:
  - Sesiones demo creadas por load_movielens.py (ml_user_1..610)
  - Ratings de esas sesiones (cascade automático)
  - Asignaciones de cluster de esas sesiones (cascade)
  - Clusters y jobs anteriores (quedan desactualizados sin los datos demo)

Qué conserva:
  - Películas (catálogo intacto)
  - Sesiones reales (usuarios que usaron la app)
  - Ratings reales
  - Búsquedas reales

Después de limpiar ejecuta:
    python retrain_clusters.py

Uso:
    python clean_demo_data.py            → muestra preview y pide confirmación
    python clean_demo_data.py --force    → ejecuta sin pedir confirmación
    python clean_demo_data.py --dry-run  → solo muestra qué se borraría
"""

import sys
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection

# UUID namespace — mismo que load_movielens.py
UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
MAX_MOVIELENS_USERS = 1000   # margen sobre los 610 del dataset


# =============================================================================
#  IDENTIFICACIÓN DE SESIONES
# =============================================================================

def get_demo_session_ids(conn) -> list[str]:
    """
    Genera los UUIDs deterministas de load_movielens.py
    y filtra solo los que realmente existen en la BD.
    """
    # Generar todos los posibles UUIDs demo
    all_demo = {
        str(uuid.uuid5(UUID_NAMESPACE, f"ml_user_{i}"))
        for i in range(1, MAX_MOVIELENS_USERS + 1)
    }

    # Intersectar con los que existen en la BD
    cur = conn.cursor()
    cur.execute("SELECT session_id::text FROM cookie_sessions")
    existing = {row["session_id"] for row in cur.fetchall()}

    return list(all_demo & existing)


# =============================================================================
#  PREVIEW
# =============================================================================

def get_stats(conn, demo_ids: list[str]) -> dict:
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS c FROM cookie_sessions")
    total_sessions = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) AS c FROM ratings")
    total_ratings = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) AS c FROM movies")
    total_movies = cur.fetchone()["c"]

    demo_ratings = 0
    if demo_ids:
        cur.execute(
            "SELECT COUNT(*) AS c FROM ratings WHERE session_id = ANY(%s::uuid[])",
            (demo_ids,)
        )
        demo_ratings = cur.fetchone()["c"]

    return {
        "total_sessions":  total_sessions,
        "real_sessions":   total_sessions - len(demo_ids),
        "demo_sessions":   len(demo_ids),
        "total_ratings":   total_ratings,
        "demo_ratings":    demo_ratings,
        "real_ratings":    total_ratings - demo_ratings,
        "total_movies":    total_movies,
    }


def print_preview(stats: dict) -> None:
    print(f"""
╔══════════════════════════════════════════════╗
║      CineMatch — Preview de limpieza         ║
╠══════════════════════════════════════════════╣
║  ESTADO ACTUAL                               ║
║    Sesiones totales    {stats['total_sessions']:>8,}              ║
║    Ratings totales     {stats['total_ratings']:>8,}              ║
║    Películas           {stats['total_movies']:>8,}              ║
╠══════════════════════════════════════════════╣
║  SE ELIMINARÁ  (datos demo de MovieLens)     ║
║    Sesiones demo       {stats['demo_sessions']:>8,}              ║
║    Ratings demo        {stats['demo_ratings']:>8,}              ║
╠══════════════════════════════════════════════╣
║  SE CONSERVARÁ  (datos reales de la app)     ║
║    Sesiones reales     {stats['real_sessions']:>8,}              ║
║    Ratings reales      {stats['real_ratings']:>8,}              ║
║    Películas           {stats['total_movies']:>8,}  (intactas)  ║
╚══════════════════════════════════════════════╝
""")


# =============================================================================
#  LIMPIEZA
# =============================================================================

def delete_demo_sessions(conn, demo_ids: list[str]) -> int:
    """
    Elimina las sesiones demo en lotes de 500.
    El ON DELETE CASCADE del schema borra automáticamente:
      - ratings de esas sesiones
      - session_cluster de esas sesiones
      - search_log de esas sesiones
    """
    cur     = conn.cursor()
    deleted = 0
    total   = len(demo_ids)

    for i in range(0, total, 500):
        batch = demo_ids[i:i+500]
        cur.execute(
            "DELETE FROM cookie_sessions WHERE session_id = ANY(%s::uuid[])",
            (batch,)
        )
        deleted += cur.rowcount
        conn.commit()

        pct = min(100, (i+500) * 100 // total)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r     [{bar}] {deleted:,}/{total:,} sesiones eliminadas",
              end="", flush=True)

    print()
    return deleted


def delete_stale_clusters(conn) -> None:
    """
    Elimina clusters y jobs anteriores — quedan desactualizados
    sin los datos demo. El usuario correrá retrain_clusters.py después.
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM session_cluster")
    cur.execute("DELETE FROM clusters")
    cur.execute("DELETE FROM cluster_job_log")
    conn.commit()
    print("   ✓ Clusters anteriores eliminados.")


def recalculate_movie_stats(conn) -> None:
    """
    Recalcula avg_rating, rating_count y popularity de todas las películas
    ahora que los ratings demo ya no existen.
    """
    cur = conn.cursor()

    print("   🔄 Recalculando estadísticas de películas...")
    cur.execute("""
        UPDATE movies SET
            avg_rating   = COALESCE(
                (SELECT AVG(score)  FROM ratings WHERE movie_id = movies.movie_id), 0),
            rating_count = COALESCE(
                (SELECT COUNT(*)    FROM ratings WHERE movie_id = movies.movie_id), 0)
    """)
    conn.commit()

    cur.execute("""
        UPDATE movies SET
            popularity = CASE
                WHEN rating_count > 0
                THEN ROUND(CAST(avg_rating * LOG(rating_count + 1) AS numeric), 4)
                ELSE 0
            END
    """)
    conn.commit()
    print("   ✓ Estadísticas y popularidad actualizadas.")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CineMatch — Limpiar datos demo")
    parser.add_argument("--force",   action="store_true",
                        help="Ejecuta sin pedir confirmación")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo muestra qué se borraría")
    args = parser.parse_args()

    print("\n🧹 CineMatch — Limpieza de datos demo (MovieLens)")
    print("=" * 50)

    print("\n[1] Conectando a Supabase...")
    conn = get_connection()
    print("   ✓ Conectado.")

    print("\n[2] Identificando sesiones demo...")
    demo_ids = get_demo_session_ids(conn)
    stats    = get_stats(conn, demo_ids)
    print_preview(stats)

    if stats["demo_sessions"] == 0:
        print("   ✓ No hay datos demo. La BD ya está limpia.")
        conn.close()
        return

    if args.dry_run:
        print("   [DRY RUN] No se ejecutó ningún cambio.\n")
        conn.close()
        return

    if not args.force:
        print("⚠️  Esta operación es IRREVERSIBLE.")
        confirm = input("   Escribe 'CONFIRMAR' para continuar: ").strip()
        if confirm != "CONFIRMAR":
            print("   Operación cancelada.")
            conn.close()
            return

    start = datetime.now(timezone.utc)

    print(f"\n[3] Eliminando {stats['demo_sessions']:,} sesiones demo...")
    deleted = delete_demo_sessions(conn, demo_ids)
    print(f"   ✓ {deleted:,} sesiones eliminadas (ratings en cascada).")

    print("\n[4] Eliminando clusters desactualizados...")
    delete_stale_clusters(conn)

    print("\n[5] Recalculando estadísticas de películas...")
    recalculate_movie_stats(conn)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    # Resumen final
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM cookie_sessions")
    final_sessions = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) AS c FROM ratings")
    final_ratings = cur.fetchone()["c"]

    print(f"""
╔══════════════════════════════════════════════╗
║           Limpieza completada ✓              ║
╠══════════════════════════════════════════════╣
║  Sesiones eliminadas   {stats['demo_sessions']:>8,}              ║
║  Ratings eliminados    {stats['demo_ratings']:>8,}              ║
║  Sesiones restantes    {final_sessions:>8,}              ║
║  Ratings restantes     {final_ratings:>8,}              ║
║  Tiempo                {elapsed:>7.1f}s              ║
╚══════════════════════════════════════════════╝

Siguiente paso:
    python retrain_clusters.py
""")

    conn.close()


if __name__ == "__main__":
    main()