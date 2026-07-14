DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS users CASCADE;
-- LangGraph Agent Database Schema

CREATE TABLE IF NOT EXISTS users (
    user_id   TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

CREATE TABLE IF NOT EXISTS messages (
    message_id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at);
