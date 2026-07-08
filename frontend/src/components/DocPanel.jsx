import { useState, useRef } from 'react';

function fmtSize(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let size = bytes;
  while (size >= 1024 && i < units.length - 1) { size /= 1024; i++; }
  return size.toFixed(i > 0 ? 1 : 0) + ' ' + units[i];
}

export default function DocPanel({ docs, activeDoc, onSelectDoc, onUpload, onDeleteDoc }) {
  const [dragover, setDragover] = useState(false);
  const fileRef = useRef(null);

  function handleFile(file) {
    if (file) onUpload(file);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragover(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  return (
    <aside className="doc-panel">
      <div className="panel-header">
        <h2>Documents</h2>
        <span className="badge">{docs.length}</span>
      </div>

      <div className="upload-area">
        <div
          className={`drop-zone ${dragover ? 'dragover' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
          onDragLeave={() => setDragover(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
        >
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <p>Drop a file or <span className="link">browse</span></p>
        </div>
        <input
          ref={fileRef}
          type="file"
          hidden
          onChange={(e) => { const f = e.target.files[0]; if (f) handleFile(f); e.target.value = ''; }}
        />
      </div>

      <div className="doc-list">
        {docs.length === 0 && (
          <div className="empty">No documents uploaded.</div>
        )}
        {docs.map((d) => (
          <div
            key={d.id}
            className={`doc-item ${activeDoc?.id === d.id ? 'active' : ''}`}
            onClick={() => onSelectDoc(activeDoc?.id === d.id ? null : d)}
          >
            <div className="doc-info">
              <div className="doc-name">{d.filename}</div>
              <div className="doc-meta">
                {fmtSize(d.file_size_bytes)} &middot; {(d.char_count || 0).toLocaleString()} chars
                {d.category ? ` · ${d.category}` : ''}
              </div>
            </div>
            <button
              className="doc-delete"
              title="Delete"
              onClick={(e) => { e.stopPropagation(); onDeleteDoc(d.id); }}
            >
              &times;
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
