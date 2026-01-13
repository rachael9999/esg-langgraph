import os
from urllib.parse import urlparse
import pymysql
from dotenv import load_dotenv

load_dotenv()

def create_database():
    db_url = os.getenv("DATABASE_URL")
    if not db_url or not db_url.startswith("mysql"):
        print("DATABASE_URL is not set to MySQL. Skipping creation.")
        return

    # Parse URL: mysql+pymysql://user:pass@host/dbname
    # Removing 'mysql+pymysql://'
    url_str = db_url.replace("mysql+pymysql://", "mysql://")
    parsed = urlparse(url_str)
    
    user = parsed.username
    password = parsed.password
    host = parsed.hostname
    port = parsed.port or 3306
    dbname = parsed.path.lstrip('/')

    print(f"Connecting to MySQL at {host} as {user}...")
    
    try:
        # Connect without selecting a DB
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            port=port,
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        
        print(f"Creating database '{dbname}' if it does not exist...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {dbname} CHARACTER SET utf8mb4;")
        conn.commit()
        
        print(f"Database '{dbname}' created successfully (or already existed).")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")

if __name__ == "__main__":
    create_database()
