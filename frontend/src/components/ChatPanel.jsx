import { useState, useRef, useEffect } from 'react';

export default function ChatPanel({
  sessions, activeSessionId, messages, streaming, activeDoc,
  onSelectSession, onNewSession, onSend,
  onRenameSession, onDeleteSession,
}) {
  const [input, setInput] = useState('');
  const [search, setSearch] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const editRef = useRef(null);

  const filteredSessions = search
    ? sessions.filter((s) => s.title.toLowerCase().includes(search.toLowerCase()))
    : sessions;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!streaming) inputRef.current?.focus();
  }, [streaming]);

  useEffect(() => {
    if (editingId) editRef.current?.focus();
  }, [editingId]);

  function handleSubmit(e) {
    e.preventDefault();
    if (!input.trim() || streaming) return;
    onSend(input.trim());
    setInput('');
  }

  function startRename(session) {
    setEditingId(session.id);
    setEditTitle(session.title);
  }

  function confirmRename() {
    if (editingId && editTitle.trim()) {
      onRenameSession(editingId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle('');
  }

  function cancelRename() {
    setEditingId(null);
    setEditTitle('');
  }

  return (
    <section className="chat-panel">
      <div className="session-bar">
        <button className="btn btn-primary btn-sm" onClick={onNewSession}>
          + New Chat
        </button>
        <div className="session-search-wrapper">
          <svg className="session-search-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            className="session-search-input"
            placeholder="Search sessions..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="session-list">
          {filteredSessions.length === 0 && (
            <div className="session-list-empty">{search ? 'No matching sessions' : 'No sessions yet'}</div>
          )}
          {filteredSessions.map((s) => (
            <div
              key={s.id}
              className={`session-item${s.id === activeSessionId ? ' active' : ''}`}
              onClick={() => onSelectSession(s.id)}
            >
              {editingId === s.id ? (
                <input
                  ref={editRef}
                  className="session-edit-input"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onBlur={confirmRename}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') confirmRename();
                    if (e.key === 'Escape') cancelRename();
                  }}
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <span className="session-title" title={s.title}>
                  {s.title}
                </span>
              )}
              <div className="session-actions">
                <button
                  className="session-action-btn"
                  title="Rename"
                  onClick={(e) => { e.stopPropagation(); startRename(s); }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                  </svg>
                </button>
                <button
                  className="session-action-btn danger"
                  title="Delete"
                  onClick={(e) => { e.stopPropagation(); onDeleteSession(s.id); }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>AI RAG System</h2>
            <p>Upload documents and ask questions about them.</p>
            <p className="hint">Your data stays local — no API costs.</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role} ${m.streaming ? 'streaming' : ''}`}>
            <div className="msg-content">
              {m.content}
              {m.streaming && <span className="cursor" />}
            </div>
            {m.role === 'assistant' && m.sources?.length > 0 && (
              <details className="msg-sources" open>
                <summary>Sources ({m.sources.length})</summary>
                <ul>
                  {m.sources.map((s, j) => (
                    <li key={j}>
                      <span>{s.source || 'unknown'}</span>
                      <span className="score">{(s.score || 0).toFixed(3)}</span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="input-bar">
        {activeDoc && (
          <div className="doc-filter">
            Filtering: {activeDoc.filename}
            <button className="clear-filter" onClick={() => onSelectDoc(null)}>&times;</button>
          </div>
        )}
        <form className="input-row" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="text"
            className="chat-input"
            placeholder="Ask a question about your documents..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={streaming}
            autoComplete="off"
          />
          <button
            type="submit"
            className="btn btn-primary send-btn"
            disabled={!input.trim() || streaming}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </form>
      </div>
    </section>
  );
}