import { useState, useEffect, lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { get } from './api.js';
import ChatPage from './pages/ChatPage.jsx';

const DashboardPage = lazy(() => import('./pages/DashboardPage.jsx'));

export default function App() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  async function checkHealth() {
    try {
      const h = await get('/api/admin/health');
      setHealth(h);
    } catch {
      try {
        const h = await get('/api/health');
        setHealth({ status: 'running', components: { llm: { available: false, model: '?' } } });
      } catch {
        setHealth(null);
      }
    }
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/chat" element={<ChatPage health={health} />} />
        <Route path="/dashboard" element={
          <Suspense fallback={<div className="dashboard"><p style={{textAlign:'center',padding:'60px 0'}}>Loading...</p></div>}>
            <DashboardPage />
          </Suspense>
        } />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </BrowserRouter>
  );
}