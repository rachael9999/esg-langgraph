from backend.session_store import SESSION_STORE
import pprint

# Access the private cache to see what's currently loaded in memory
# Note: This only shows sessions active in THIS process, 
# and since we are running this as a script, it will start empty unless we load something.

print("--- Inspecting In-Memory SESSION_STORE ---")

# Let's verify by trying to 'get' a known session to see what the object looks like
session_id = '1' 
print(f"Loading session '{session_id}'...")
session_data = SESSION_STORE.get(session_id)

print(f"\n[SessionData Object for '{session_id}']")
print(f"Type: {type(session_data)}")
print(f"session_id: {session_data.session_id}")
print(f"vectorstore: {session_data.vectorstore}")

if session_data.vectorstore:
    print(f"\n[Vectorstore Details]")
    print(f"Type: {type(session_data.vectorstore)}")
    print(f"Index: {session_data.vectorstore.index}")
    print(f"Number of documents in index: {session_data.vectorstore.index.ntotal}")

print("\n[Internal Cache State]")
pprint.pprint(SESSION_STORE._cache)
