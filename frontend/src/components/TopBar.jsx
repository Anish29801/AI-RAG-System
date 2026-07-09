import { NavLink } from 'react-router-dom';

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
      <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
        <div className="logo">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
          </svg>
          <span>AI RAG System</span>
        </div>
        <nav className="nav-links">
          <NavLink to="/chat" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            Chat
          </NavLink>
          <NavLink to="/dashboard" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 20V10M18 20V4M6 20v-4" />
            </svg>
            Dashboard
          </NavLink>
        </nav>
      </div>
      <div className="health">
        <span className={dotClass} />
        <span className="health-label">{label}</span>
      </div>
    </header>
  );
}