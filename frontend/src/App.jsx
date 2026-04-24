import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  MessageSquare, BarChart2, Database, Plus, Send, 
  Cpu, Zap, Brain, LayoutGrid, Clock, DollarSign 
} from 'lucide-react';

const API_BASE = "http://localhost:8000/api";

const App = () => {
  const [activeTab, setActiveTab] = useState('chat');
  const [mode, setMode] = useState('all'); // 'rag', 'ml', 'llm-only', 'zero-shot', 'all'
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => { scrollToBottom() }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      // Determine which endpoint to hit based on the mode selector
      const endpoint = mode === 'all' ? '/predict/all' : `/predict/${mode}`;
      const response = await axios.post(`${API_BASE}${endpoint}`, { question: input });
      
      const assistantMsg = { 
        role: 'assistant', 
        type: mode, 
        data: response.data 
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      console.error("API Error:", error);
      setMessages(prev => [...prev, { role: 'assistant', content: "Error: Is the backend running?" }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-ef-bg text-ef-fg overflow-hidden">
      {/* Sidebar */}
      <aside className="w-72 bg-ef-bg-soft border-r border-ef-gray flex flex-col p-4">
        <button className="flex items-center gap-2 bg-ef-gray hover:bg-opacity-80 p-3 rounded-xl transition-all mb-8">
          <Plus size={18} /> <span className="font-medium">New Case</span>
        </button>
        
        <nav className="flex-1 space-y-4">
          <div className="space-y-1">
            <TabButton icon={<MessageSquare size={18}/>} label="Assistant" active={activeTab === 'chat'} onClick={() => setActiveTab('chat')} />
            <TabButton icon={<BarChart2 size={18}/>} label="Evaluation" active={activeTab === 'evaluate'} onClick={() => setActiveTab('evaluate')} />
            <TabButton icon={<Database size={18}/>} label="Ingestion" active={activeTab === 'ingest'} onClick={() => setActiveTab('ingest')} />
          </div>
        </nav>

        <div className="pt-4 border-t border-ef-gray">
          <div className="text-[10px] uppercase font-bold text-ef-gray tracking-widest mb-2">Hardware Status</div>
          <div className="flex items-center gap-2 text-xs text-ef-green">
            <div className="w-2 h-2 rounded-full bg-ef-green animate-pulse" />
            RTX 3060 - CUDA Ready
          </div>
        </div>
      </aside>

      {/* Main Container */}
      <main className="flex-1 flex flex-col relative">
        <header className="p-4 border-b border-ef-gray flex justify-between items-center">
          <h2 className="text-sm font-bold tracking-tight uppercase text-ef-blue">
            Decision Intelligence <span className="text-ef-gray mx-2">/</span> {activeTab}
          </h2>
          
          {activeTab === 'chat' && (
            <div className="flex bg-ef-bg-soft p-1 rounded-lg border border-ef-gray text-xs">
              {['rag', 'ml', 'all'].map(m => (
                <button key={m} onClick={() => setMode(m)} className={`px-3 py-1 rounded-md transition-all ${mode === m ? 'bg-ef-blue text-ef-bg font-bold' : 'hover:text-ef-blue'}`}>
                  {m.toUpperCase()}
                </button>
              ))}
            </div>
          )}
        </header>

        <section className="flex-1 overflow-y-auto p-6 space-y-6">
          {activeTab === 'chat' ? (
            <div className="max-w-5xl mx-auto space-y-6">
              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] ${msg.role === 'user' ? 'bg-ef-gray p-4 rounded-2xl' : 'w-full'}`}>
                    {msg.role === 'user' ? msg.content : <AssistantResponse msg={msg} />}
                  </div>
                </div>
              ))}
              {loading && <div className="text-ef-blue animate-pulse text-sm">Thinking...</div>}
              <div ref={chatEndRef} />
            </div>
          ) : activeTab === 'evaluate' ? (
            <AdminTab title="Evaluation Report" script="run-evaluation" />
          ) : (
            <AdminTab title="Vector DB Management" script="run-ingestion" />
          )}
        </section>

        {/* Input Bar */}
        {activeTab === 'chat' && (
          <footer className="p-6 bg-gradient-to-t from-ef-bg to-transparent">
            <div className="max-w-4xl mx-auto relative">
              <input 
                value={input} 
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask about a support ticket..." 
                className="w-full bg-ef-bg-soft border border-ef-gray p-4 pr-16 rounded-2xl focus:outline-none focus:border-ef-blue transition-all"
              />
              <button onClick={handleSend} className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-ef-blue text-ef-bg rounded-xl hover:opacity-90 transition-all">
                <Send size={20} />
              </button>
            </div>
          </footer>
        )}
      </main>
    </div>
  );
};

// --- Helper Components ---

const TabButton = ({ icon, label, active, onClick }) => (
  <button onClick={onClick} className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all ${active ? 'bg-ef-blue text-ef-bg font-bold shadow-lg' : 'hover:bg-ef-gray'}`}>
    {icon} <span>{label}</span>
  </button>
);

const AssistantResponse = ({ msg }) => {
  if (msg.type === 'all') {
    const { rag, ml, zero_shot, llm_only } = msg.data;
    return (
      <div className="grid grid-cols-2 gap-4">
        <MetricCard title="RAG (Context)" icon={<Brain size={16}/>} content={rag.answer} latency={rag.latency_ms} cost={rag.cost_usd} />
        <MetricCard title="ML (Local)" icon={<Cpu size={16}/>} content={`Priority: ${ml.priority}`} latency={ml.latency_ms} cost={0} />
        <MetricCard title="LLM (Zero-Shot)" icon={<Zap size={16}/>} content={zero_shot.reasoning} latency={zero_shot.latency_ms} cost={zero_shot.cost_usd} />
        <MetricCard title="LLM (No Context)" icon={<LayoutGrid size={16}/>} content={llm_only.answer} latency={llm_only.latency_ms} cost={llm_only.cost_usd} />
      </div>
    );
  }
  return <div className="bg-ef-bg-soft p-5 rounded-2xl border border-ef-gray">{msg.data.answer || msg.data.reasoning || "Result received."}</div>;
};

const MetricCard = ({ title, icon, content, latency, cost }) => (
  <div className="bg-ef-bg-soft p-4 rounded-xl border border-ef-gray flex flex-col gap-3">
    <div className="flex justify-between items-center">
      <span className="text-xs font-bold text-ef-blue flex items-center gap-2">{icon} {title}</span>
      <div className="flex gap-2">
        <span className="text-[10px] bg-ef-gray px-2 py-0.5 rounded flex items-center gap-1"><Clock size={10}/> {latency}ms</span>
        <span className="text-[10px] bg-ef-gray px-2 py-0.5 rounded flex items-center gap-1"><DollarSign size={10}/> {cost}</span>
      </div>
    </div>
    <p className="text-xs line-clamp-4 leading-relaxed opacity-80">{content}</p>
  </div>
);

const AdminTab = ({ title, script }) => {
  const [status, setStatus] = useState('idle');
  const trigger = async () => {
    setStatus('running');
    try { await axios.post(`${API_BASE}/admin/${script}`); setStatus('success'); } 
    catch { setStatus('error'); }
  };
  return (
    <div className="flex flex-col items-center justify-center h-full space-y-4">
      <h3 className="text-2xl font-bold">{title}</h3>
      <button onClick={trigger} className="bg-ef-green text-ef-bg px-6 py-3 rounded-xl font-bold hover:scale-105 transition-all">
        {status === 'running' ? 'Running Script...' : `Run ${title}`}
      </button>
      {status === 'success' && <p className="text-ef-green text-sm">Triggered successfully! Check backend logs.</p>}
    </div>
  );
};

export default App;