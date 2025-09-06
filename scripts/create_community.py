#!/usr/bin/env python3
import os, sys
import psycopg
from dotenv import load_dotenv

def get_dsn():
    load_dotenv()
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5433")
        name = os.getenv("DB_NAME", "hoa_ai")
        user = os.getenv("DB_USER", "hoa_user")
        pwd  = os.getenv("DB_PASS", "hoa_pass")
        dsn  = f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
    return dsn

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_community.py \"Community Name\"")
        sys.exit(1)
    name = sys.argv[1]
    dsn = get_dsn()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO communities (name) VALUES (%s) RETURNING id", (name,))
            cid = cur.fetchone()[0]
            print(f"OK: community created id={cid}, name={name}")

if __name__ == "__main__":
    main()
