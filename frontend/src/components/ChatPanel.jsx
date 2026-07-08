import { useState, useRef, useEffect } from 'react';

export default function ChatPanel({
  sessions, activeSessionId, messages, streaming, activeDoc,
  onSelectSession, onNewSession, onSend,
}) {
  const [input, setInput] = useState('');
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!streaming) inputRef.current?.focus();
  }, [streaming]);

  function handleSubmit(e) {
    e.preventDefault();
    if (!input.trim() || streaming) return;
    onSend(input.trim());
    setInput('');
  }

  return (
    <section className="chat-panel">
      <div className="session-bar">
        <button className="btn btn-primary btn-sm" onClick={onNewSession}>
          + New Chat
        </button>
        <select
          value={activeSessionId || ''}
          onChange={(e) => onSelectSession(e.target.value)}
        >
          {sessions.length === 0 && <option value="">No sessions</option>}
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>{s.title}</option>
          ))}
        </select>
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
