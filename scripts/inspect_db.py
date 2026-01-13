import os
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv

load_dotenv()

def inspect_database():
    db_url = os.getenv("DATABASE_URL")
    print(f"Connecting to: {db_url}")

    conn = None
    try:
        engine = create_engine(db_url)
        conn = engine.connect()
        print("Connection successful!")

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if not tables:
            print("No tables found in the database.")
            return

        print(f"Tables found: {tables}")

        for table_name in tables:
            print(f"\n--- Data in table: {table_name} (Limit 5) ---")
            try:
                # Use text() for safe SQL execution
                query = text(f"SELECT * FROM {table_name} LIMIT 5")
                result = conn.execute(query)
                rows = result.fetchall()
                if not rows:
                    print("(Table is empty)")
                else:
                    # Print headers
                    print(result.keys()) 
                    for row in rows:
                        print(row)
            except Exception as e:
                print(f"Error reading table {table_name}: {e}")

    except Exception as e:
        print(f"Database connection error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    inspect_database()
