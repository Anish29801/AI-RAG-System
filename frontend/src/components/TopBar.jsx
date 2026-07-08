export default function TopBar({ health }) {
  const status = health?.status;
  const dotClass =
    status === 'healthy' ? 'dot healthy' : status === 'degraded' || status === 'running' ? 'dot degraded' : 'dot error';
  const label =
    status === 'healthy'
      ? `Healthy · ${health.components?.llm?.model || 'OK'}`
      : status === 'running'
        ? 'Running (LLM offline)'
        : status === 'degraded'
          ? 'Degraded (LLM unavailable)'
          : 'Backend unreachable';

  return (
    <header className="topbar">
      <div className="logo">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="3" />
          <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
        </svg>
        <span>AI RAG System</span>
      </div>
      <div className="health">
        <span className={dotClass} />
        <span className="health-label">{label}</span>
      </div>
    </header>
  );
}
