import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Asegúrate de que el módulo db que tienes es accesible
sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
# Endpoint de TMDB para buscar películas mediante IDs externos (como imdb_id)
TMDB_FIND_URL = "https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

MAX_WORKERS = 10  # Número de peticiones simultáneas a la API para acelerar el proceso

def fetch_poster_from_tmdb(imdb_id: str):
    """Consulta la API de TMDB para obtener el poster oficial usando el imdb_id."""
    if not imdb_id:
        return None

    url = TMDB_FIND_URL.format(imdb_id=imdb_id, api_key=TMDB_API_KEY)
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            movie_results = data.get("movie_results", [])
            # Verificamos que se encontró la película y que tiene póster
            if movie_results and movie_results[0].get("poster_path"):
                return TMDB_IMAGE_BASE + movie_results[0]["poster_path"]
    except Exception as e:
        print(f"Error al consultar TMDB para {imdb_id}: {e}")
    
    return None

def process_movie(movie):
    """Obtiene el póster de la API y devuelve los datos listos para el UPDATE."""
    movie_id, imdb_id, title = movie
    poster_url = fetch_poster_from_tmdb(imdb_id)
    return movie_id, imdb_id, title, poster_url

def main():
    if not TMDB_API_KEY:
        print("❌ ERROR: No se encontró TMDB_API_KEY en el archivo .env")
        sys.exit(1)

    try:
        conn = get_connection()
    except Exception as e:
        print(f"❌ Error conectando a la BD: {e}")
        sys.exit(1)

    cur = conn.cursor()

    print("🎬 Obteniendo lista de películas desde Supabase...")
    # Obtenemos todas las películas que tengan un imdb_id válido
    cur.execute("SELECT movie_id, imdb_id, title FROM movies WHERE imdb_id IS NOT NULL")
    movies = cur.fetchall()
    
    # Si psycopg2 devuelve diccionarios, convertimos a tuplas
    if movies and isinstance(movies[0], dict):
        movies = [(m["movie_id"], m["imdb_id"], m["title"]) for m in movies]

    total_movies = len(movies)
    print(f"🔍 Encontradas {total_movies} películas para procesar.\n")

    updates = []
    processed = 0

    # Usamos ThreadPoolExecutor para hacer peticiones en paralelo (mucho más rápido)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Enviamos todas las tareas al pool
        futures = {executor.submit(process_movie, m): m for m in movies}

        for future in as_completed(futures):
            movie_id, imdb_id, title, poster_url = future.result()
            processed += 1

            if poster_url:
                updates.append((poster_url, movie_id))
                # print para monitorear el progreso
                print(f"[{processed}/{total_movies}] ✅ Éxito: {title[:40]}") 
            else:
                # Si no hay imagen en TMDB, le asignamos NULL para limpiar posibles errores previos
                updates.append((None, movie_id))
                print(f"[{processed}/{total_movies}] ⚠️ Sin poster: {title[:40]} ({imdb_id})")

    print(f"\n💾 Guardando {len(updates)} actualizaciones en la base de datos...")
    
    # Hacemos el UPDATE masivo en la tabla movies
    try:
        cur.executemany(
            """
            UPDATE movies 
            SET poster_url = %s 
            WHERE movie_id = %s
            """,
            updates
        )
        conn.commit()
        print(f"✅ ¡Operación finalizada! Las imágenes ahora coinciden con los títulos.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error al guardar en la BD: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()