import { useState, useEffect, useCallback } from 'react';
import { get, post, postForm, del, postStream } from '../api.js';
import TopBar from '../components/TopBar.jsx';
import DocPanel from '../components/DocPanel.jsx';
import ChatPanel from '../components/ChatPanel.jsx';

export default function ChatPage({ health }) {
  const [docs, setDocs] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [activeDoc, setActiveDoc] = useState(null);

  useEffect(() => {
    loadDocs();
    loadSessions();
  }, []);

  async function loadDocs() {
    try {
      const d = await get('/api/documents/');
      setDocs(d);
    } catch (e) {
      console.error('Failed to load docs:', e);
    }
  }

  async function loadSessions() {
    try {
      const s = await get('/api/chat/sessions');
      setSessions(s);
    } catch {
      // no sessions yet
    }
  }

  async function switchSession(id) {
    setActiveSessionId(id);
    try {
      const msgs = await get(`/api/chat/sessions/${id}/messages`);
      setMessages(msgs);
    } catch {
      setMessages([]);
    }
  }

  async function handleNewSession() {
    try {
      const s = await post('/api/chat/sessions', { title: 'New Chat' });
      await loadSessions();
      await switchSession(s.session_id);
    } catch (e) {
      setMessages((prev) => [...prev, { role: 'error', content: `Failed: ${e.message}` }]);
    }
  }

  async function handleUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', 'general');
    formData.append('tags', '');
    try {
      await postForm('/api/documents/upload', formData);
      await loadDocs();
    } catch (e) {
      alert(`Upload failed: ${e.message}`);
    }
  }

  async function handleDeleteDoc(id) {
    if (!confirm('Delete this document?')) return;
    try {
      await del(`/api/documents/${id}`);
      await loadDocs();
      if (activeDoc?.id === id) setActiveDoc(null);
    } catch (e) {
      alert(`Delete failed: ${e.message}`);
    }
  }

  const handleSend = useCallback(
    async (query) => {
      if (streaming || !query.trim()) return;

      setStreaming(true);
      setMessages((prev) => [...prev, { role: 'user', content: query }]);

      const body = {
        query,
        session_id: activeSessionId || undefined,
        stream: true,
        document_filter: activeDoc?.filename || undefined,
      };

      try {
        const resp = await postStream('/api/chat/ask/stream', body);
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullAnswer = '';
        let sources = [];
        let sessionId = activeSessionId;

        setMessages((prev) => [...prev, { role: 'assistant', content: '', streaming: true }]);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let idx;
          while ((idx = buffer.indexOf('\n\n')) !== -1) {
            const block = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            const eventMatch = block.match(/^event: (.+)$/m);
            const dataMatch = block.match(/^data: (.+)$/m);
            if (!eventMatch || !dataMatch) continue;
            const event = eventMatch[1];
            const data = JSON.parse(dataMatch[1]);

            if (event === 'sources') {
              sources = data;
            } else if (event === 'token') {
              fullAnswer += data.token;
              setMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last?.streaming) {
                  next[next.length - 1] = { ...last, content: fullAnswer };
                }
                return next;
              });
            } else if (event === 'done') {
              sessionId = data.session_id;
              setMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last?.streaming) {
                  next[next.length - 1] = {
                    role: 'assistant',
                    content: fullAnswer,
                    sources,
                  };
                }
                return next;
              });
              if (!activeSessionId) {
                setActiveSessionId(sessionId);
                await loadSessions();
              }
            }
          }
        }
      } catch (e) {
        setMessages((prev) => [...prev, { role: 'error', content: `Error: ${e.message}` }]);
      }

      setStreaming(false);
    },
    [streaming, activeSessionId, activeDoc],
  );

  return (
    <div className="app">
      <TopBar health={health} />
      <div className="main">
        <DocPanel
          docs={docs}
          activeDoc={activeDoc}
          onSelectDoc={setActiveDoc}
          onUpload={handleUpload}
          onDeleteDoc={handleDeleteDoc}
        />
        <ChatPanel
          sessions={sessions}
          activeSessionId={activeSessionId}
          messages={messages}
          streaming={streaming}
          activeDoc={activeDoc}
          onSelectSession={switchSession}
          onNewSession={handleNewSession}
          onSend={handleSend}
        />
      </div>
    </div>
  );
}