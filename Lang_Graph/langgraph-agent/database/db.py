"""PostgreSQL connection helpers and schema initialization."""
import os
from contextlib import contextmanager
from psycopg import connect
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool  # Pool import kiya

from config import get_settings

settings = get_settings()

# Pure application ke liye ek single pool configure kiya
# Yeh background mein 1 se lekar 10 tak connections open rakhega automatically
pool = ConnectionPool(
    conninfo=settings.database_url,
    min_size=1,
    max_size=10,
    kwargs={"row_factory": dict_row}
)

@contextmanager
def get_connection():
    """Retrieve an efficient managed connection from the pool."""
    with pool.connection() as conn:
        yield conn

def init_db():
    """Execute structural migration layout logic on startup hooks."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
    print("Database connection pool initialized and schema verified successfully.")