import { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [activeItem, setActiveItem] = useState(null);
  const scrollRef = useRef(null);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, brand: 'General' }),
      });
      const data = await res.json();
      const entry = { ...data, id: Date.now() };
      setHistory(prev => [entry, ...prev]);
      setActiveItem(entry);
      setQuery('');
    } catch (err) {
      alert("Backend unreachable. Ensure your FastAPI server is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* Sidebar for Interaction History [cite: 65] */}
      <aside className="sidebar">
        <div className="sidebar-header">History</div>
        <div className="history-list">
          {history.map(item => (
            <div 
              key={item.id} 
              className={`history-pill ${activeItem?.id === item.id ? 'active' : ''}`}
              onClick={() => setActiveItem(item)}
            >
              {item.query}
            </div>
          ))}
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="workspace">
        {!activeItem && !loading ? (
          <div className="empty-state">
            <h1>Decision Intelligence</h1>
            <p>Compare RAG vs ML Priority in Real-time</p>
          </div>
        ) : (
          <div className="results-container">
            {/* Comparison Table (The "Whole Point") [cite: 49, 76] */}
            <section className="metrics-card">
              <h3>Comparison Metrics</h3>
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>System</th>
                      <th>Priority</th>
                      <th>Latency</th>
                      <th>Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="ml-row">
                      <td>Trained ML (Baseline)</td>
                      <td>{activeItem?.ml_priority.label === 1 ? '🔴 High' : '🟢 Low'}</td>
                      <td>{activeItem?.ml_priority.latency}ms</td>
                      <td>$0.00</td>
                    </tr>
                    <tr className="llm-row">
                      <td>LLM Zero-Shot</td>
                      <td>{activeItem?.llm_priority.label === 1 ? '🔴 High' : '🟢 Low'}</td>
                      <td>{activeItem?.llm_priority.latency}ms</td>
                      <td>${activeItem?.llm_priority.cost}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            {/* Answer Comparison [cite: 35, 51-52] */}
            <div className="answers-grid">
              <div className="answer-box rag">
                <label>RAG GROUNDED ANSWER</label>
                <p>{activeItem?.rag_answer.answer}</p>
                <div className="box-meta">Latency: {activeItem?.rag_answer.latency}ms</div>
              </div>
              <div className="answer-box">
                <label>LLM ONLY ANSWER</label>
                <p>{activeItem?.non_rag_answer.answer}</p>
              </div>
            </div>

            {/* Source Panel [cite: 75] */}
            <section className="sources-section">
              <label>Retrieved Knowledge Base</label>
              <div className="sources-row">
                {activeItem?.sources.map((s, i) => (
                  <div key={i} className="source-item">
                    <div className="source-meta">Dist: {s.distance.toFixed(3)}</div>
                    <p>{s.text}</p>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}

        {/* Floating Input Box [cite: 73] */}
        <div className="input-area">
          <form onSubmit={handlePredict} className="input-wrapper">
            <input 
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Ask about a customer issue..."
              autoFocus
            />
            <button type="submit" disabled={loading}>
              {loading ? "..." : "Analyze"}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;