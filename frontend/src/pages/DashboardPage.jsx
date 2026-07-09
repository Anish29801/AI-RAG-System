import { useState, useEffect, useCallback } from 'react';
import { get, put } from '../api.js';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  ResponsiveContainer, Tooltip, Area, AreaChart,
} from 'recharts';

export default function DashboardPage() {
  const [settings, setSettings] = useState(null);
  const [models, setModels] = useState([]);
  const [currentModel, setCurrentModel] = useState('');
  const [tempVal, setTempVal] = useState(0.1);
  const [topPVal, setTopPVal] = useState(0.9);
  const [topKVal, setTopKVal] = useState(40);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState(null);
  const [history, setHistory] = useState([]);

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

  useEffect(() => {
    loadAll();
    genMockHistory();
  }, []);

  async function loadAll() {
    try {
      const [s, m] = await Promise.all([
        get('/api/admin/llm-settings'),
        get('/api/admin/llm-models'),
      ]);
      setSettings(s);
      setTempVal(s.temperature);
      setTopPVal(s.top_p);
      setTopKVal(s.top_k);
      setCurrentModel(s.model);
      setModels(m.models || []);
    } catch (e) {
      console.error('Failed to load settings:', e);
    }
  }

  function genMockHistory() {
    const h = [];
    for (let i = 0; i < 20; i++) {
      const t = Date.now() - (19 - i) * 30000;
      h.push({
        time: new Date(t).toLocaleTimeString(),
        temperature: 0.05 + Math.random() * 0.9,
        top_p: 0.7 + Math.random() * 0.28,
        top_k: Math.round(10 + Math.random() * 80),
      });
    }
    setHistory(h);
  }

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      const body = {
        model: currentModel,
        temperature: Math.round(tempVal * 100) / 100,
        top_p: Math.round(topPVal * 100) / 100,
        top_k: Math.round(topKVal),
      };
      const result = await put('/api/admin/llm-settings', body);
      setSettings(result);
      showToast('Settings saved');
    } catch (e) {
      showToast('Failed to save');
    }
    setSaving(false);
  }, [currentModel, tempVal, topPVal, topKVal]);

  if (!settings) {
    return (
      <div className="dashboard">
        <p style={{ textAlign: 'center', padding: '60px 0' }}>Loading settings...</p>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {toast && <div className="toast">{toast}</div>}

      <div>
        <h1>LLM Dashboard</h1>
        <p>Manage model selection, generation parameters, and monitor performance.</p>
      </div>

      {/* Stats Row */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
            </svg>
          </div>
          <div className="stat-value">{settings.model.split(':')[0]}</div>
          <div className="stat-label">Active Model</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 20V10M18 20V4M6 20v-4" />
            </svg>
          </div>
          <div className="stat-value">{settings.temperature.toFixed(2)}</div>
          <div className="stat-label">Temperature</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 6v6l4 2" />
            </svg>
          </div>
          <div className="stat-value">{settings.top_p.toFixed(2)}</div>
          <div className="stat-label">Top-P</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
            </svg>
          </div>
          <div className="stat-value">{settings.top_k}</div>
          <div className="stat-label">Top-K</div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="dash-grid">
        {/* Model Selector */}
        <div className="settings-card">
          <div className="card-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
            </svg>
            Model Selection
          </div>
          <div className="model-selector">
            <label>Choose LLM model</label>
            <select value={currentModel} onChange={(e) => setCurrentModel(e.target.value)}>
              {models.length === 0 && <option value={settings.model}>{settings.model}</option>}
              {models.map((m) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Temperature Control */}
        <div className="settings-card">
          <div className="card-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 20V10M18 20V4M6 20v-4" />
            </svg>
            Temperature
          </div>
          <div className="slider-row">
            <div className="slider-header">
              <span className="slider-label">Controls randomness (0 = deterministic, 2 = chaotic)</span>
              <span className="slider-value">{Number(tempVal).toFixed(2)}</span>
            </div>
            <input
              type="range" min="0" max="2" step="0.01"
              value={tempVal}
              onChange={(e) => setTempVal(parseFloat(e.target.value))}
            />
          </div>
          <div className="chart-container" style={{ height: 120 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history.slice(-15)}>
                <defs>
                  <linearGradient id="tempGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#818cf8" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" hide />
                <YAxis domain={[0, 2]} hide />
                <Tooltip
                  contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Area type="monotone" dataKey="temperature" stroke="#818cf8" fill="url(#tempGrad)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top-P Control */}
        <div className="settings-card">
          <div className="card-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 6v6l4 2" />
            </svg>
            Top-P (Nucleus Sampling)
          </div>
          <div className="slider-row">
            <div className="slider-header">
              <span className="slider-label">Probability threshold for token selection</span>
              <span className="slider-value">{Number(topPVal).toFixed(2)}</span>
            </div>
            <input
              type="range" min="0" max="1" step="0.01"
              value={topPVal}
              onChange={(e) => setTopPVal(parseFloat(e.target.value))}
            />
          </div>
          <div className="chart-container" style={{ height: 120 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history.slice(-15)}>
                <defs>
                  <linearGradient id="topPGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#34d399" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" hide />
                <YAxis domain={[0, 1]} hide />
                <Tooltip
                  contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Area type="monotone" dataKey="top_p" stroke="#34d399" fill="url(#topPGrad)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top-K Control */}
        <div className="settings-card">
          <div className="card-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
            </svg>
            Top-K (Token Filtering)
          </div>
          <div className="slider-row">
            <div className="slider-header">
              <span className="slider-label">Number of highest-probability tokens to consider</span>
              <span className="slider-value">{Math.round(topKVal)}</span>
            </div>
            <input
              type="range" min="1" max="100" step="1"
              value={topKVal}
              onChange={(e) => setTopKVal(parseFloat(e.target.value))}
            />
          </div>
          <div className="chart-container" style={{ height: 120 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={history.slice(-10)}>
                <XAxis dataKey="time" hide />
                <YAxis domain={[0, 100]} hide />
                <Tooltip
                  contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Bar dataKey="top_k" fill="#f59e0b" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Parameter Comparison Chart */}
        <div className="settings-card full-width">
          <div className="card-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
            Parameter History (last 20 samples)
          </div>
          <div className="chart-container" style={{ height: 200, padding: '8px 0' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <XAxis
                  dataKey="time"
                  tick={{ fill: '#71717a', fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: '#27272a' }}
                />
                <YAxis
                  tick={{ fill: '#71717a', fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: '#27272a' }}
                  domain={[0, 2]}
                />
                <Tooltip
                  contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Line type="monotone" dataKey="temperature" stroke="#818cf8" strokeWidth={2} dot={false} name="Temperature" />
                <Line type="monotone" dataKey="top_p" stroke="#34d399" strokeWidth={2} dot={false} name="Top-P" />
                <Line type="monotone" dataKey="top_k" stroke="#f59e0b" strokeWidth={2} dot={false} name="Top-K"
                  yAxisId={1} hide />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Action Bar */}
      <div className="action-bar">
        <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
          {saving ? 'Saving...' : 'Apply Settings'}
        </button>
        <button className="btn btn-secondary" onClick={loadAll}>
          Reset to Server
        </button>
      </div>
    </div>
  );
}