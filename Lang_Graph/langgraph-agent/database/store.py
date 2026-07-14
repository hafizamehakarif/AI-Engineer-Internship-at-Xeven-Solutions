import uuid
# We import the modern context manager connection from your existing db.py
from database.db import get_connection

def initialize_session(user_id: str, session_id: str = None) -> dict:
    """
    Validates users and sessions against your 3-table schema using psycopg3.
    - Ensures user exists in 'users' table.
    - Generates a dynamic session key if session_id is not passed.
    - If session_id is passed but doesn't exist, creates it on the fly!
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                # Step 1: Ensure the user exists in the 'users' table first.
                cursor.execute(
                    """
                    INSERT INTO users (user_id) 
                    VALUES (%s) 
                    ON CONFLICT (user_id) DO NOTHING;
                    """,
                    (user_id,)
                )

                # Scenario A: User wants a brand new auto-generated session
                if not session_id:
                    new_session_id = f"sess_{str(uuid.uuid4())[:8]}"
                    default_title = f"Chat with Agent - {new_session_id}"
                    
                    cursor.execute(
                        """
                        INSERT INTO sessions (session_id, user_id, title) 
                        VALUES (%s, %s, %s);
                        """,
                        (new_session_id, user_id, default_title)
                    )
                    conn.commit()
                    
                    return {
                        "user_id": user_id,
                        "session_id": new_session_id,
                        "status": "new_session_created",
                        "message": "Backend generated a fresh dynamic session key successfully."
                    }
                    
                # Scenario B: User passed a custom/specific session_id
                else:
                    # Check if this specific session already exists
                    cursor.execute(
                        "SELECT session_id FROM sessions WHERE user_id = %s AND session_id = %s;",
                        (user_id, session_id)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        # If session exists, update updated_at timestamp
                        cursor.execute(
                            "UPDATE sessions SET updated_at = NOW() WHERE session_id = %s;",
                            (session_id,)
                        )
                        conn.commit()
                        
                        return {
                            "user_id": user_id,
                            "session_id": session_id,
                            "status": "existing_session",
                            "message": "Welcome back! The session was validated and successfully restored."
                        }
                    else:
                        # If the session does NOT exist, CREATE IT dynamically!
                        default_title = f"Chat Session - {session_id}"
                        cursor.execute(
                            """
                            INSERT INTO sessions (session_id, user_id, title) 
                            VALUES (%s, %s, %s);
                            """,
                            (session_id, user_id, default_title)
                        )
                        conn.commit()
                        
                        return {
                            "user_id": user_id,
                            "session_id": session_id,
                            "status": "new_session_created",
                            "message": f"Custom session '{session_id}' was successfully created and registered."
                        }
                        
            except Exception as e:
                conn.rollback()
                raise e

def save_chat_message(session_id: str, role: str, content: str):
    """
    Saves a single message block (user, assistant, or tool) into the 'messages' table.
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                # Inserts into your provided 'messages' table layout
                cursor.execute(
                    """
                    INSERT INTO messages (session_id, role, content) 
                    VALUES (%s, %s, %s);
                    """,
                    (session_id, role, content)
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e

def get_sliding_window_history(session_id: str, limit: int = 5) -> list:
    """
    Fetches only the last 'k' rows from the 'messages' table to optimize memory overhead.
    Uses your dict_row factory format for modern clean mapping.
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                # Grabs the last 'limit' messages sorting backwards, then orders them chronologically
                query = """
                    SELECT role, content FROM (
                        SELECT role, content, message_id FROM messages
                        WHERE session_id = %s
                        ORDER BY created_at DESC, message_id DESC
                        LIMIT %s
                    ) subquery ORDER BY message_id ASC;
                """
                cursor.execute(query, (session_id, limit))
                rows = cursor.fetchall() # This returns a list of dictionaries, e.g., [{"role": "user", "content": "hi"}]
                
                # Format the database dictionary rows into structured shapes for the Gemini model layer
                formatted_history = []
                for row in rows:
                    formatted_history.append({
                        "role": row["role"],
                        "parts": [{"text": row["content"]}]
                    })
                    
                return formatted_history
                
            except Exception as e:
                raise e
            
def delete_session(session_id: str) -> bool:
    """
    Deletes an entire session from the database.
    Due to 'ON DELETE CASCADE' in your schema, all messages linked 
    to this session will be automatically and safely purged as well.
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                # First, check if the session exists
                cursor.execute("SELECT session_id FROM sessions WHERE session_id = %s;", (session_id,))
                if not cursor.fetchone():
                    return False  # Session didn't exist in the first place
                
                # If it exists, execute the delete command
                cursor.execute("DELETE FROM sessions WHERE session_id = %s;", (session_id,))
                conn.commit()
                return True
                
            except Exception as e:
                conn.rollback()
                raise e            