import React, { useState } from 'react';
import { MessageSquare, BarChart2, Database, Plus, Search } from 'lucide-react';

const App = () => {
  const [activeTab, setActiveTab] = useState('chat'); // 'chat', 'evaluate', 'ingest'
  const [conversations, setConversations] = useState([{ id: 1, title: 'Network Issue Case' }]);

  return (
    <div className="flex h-screen bg-efBg text-efFg font-sans">
      {/* Sidebar - Chat History */}
      <aside className="w-64 bg-efBgSoft border-r border-efGray flex flex-col">
        <div className="p-4">
          <button className="w-full flex items-center justify-center gap-2 py-2 px-4 border border-efGray rounded-lg hover:bg-efGray transition-colors">
            <Plus size={18} /> New Chat
          </button>
        </div>
        
        <nav className="flex-1 overflow-y-auto px-2 space-y-1">
          <p className="text-xs uppercase text-efGray px-2 font-bold mb-2">Recent Chats</p>
          {conversations.map(conv => (
            <div key={conv.id} className="p-2 rounded-md hover:bg-efGray cursor-pointer flex items-center gap-2">
              <MessageSquare size={14} className="text-efBlue" />
              <span className="truncate text-sm">{conv.title}</span>
            </div>
          ))}
        </nav>

        {/* Tab Switcher - Bottom of Sidebar */}
        <div className="p-4 border-t border-efGray space-y-2">
          <button onClick={() => setActiveTab('chat')} className={`w-full flex items-center gap-3 p-2 rounded ${activeTab === 'chat' ? 'bg-efGray' : ''}`}>
            <MessageSquare size={20} /> Assistant
          </button>
          <button onClick={() => setActiveTab('evaluate')} className={`w-full flex items-center gap-3 p-2 rounded ${activeTab === 'evaluate' ? 'bg-efGray' : ''}`}>
            <BarChart2 size={20} /> Evaluate
          </button>
          <button onClick={() => setActiveTab('ingest')} className={`w-full flex items-center gap-3 p-2 rounded ${activeTab === 'ingest' ? 'bg-efGray' : ''}`}>
            <Database size={20} /> Ingestion
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="p-4 border-b border-efGray flex justify-between items-center bg-efBg">
          <h1 className="text-xl font-semibold capitalize">{activeTab} Mode</h1>
          <div className="flex items-center gap-4">
             {/* User Info from Charbel's profile could go here */}
             <span className="text-xs text-efBlue">RTX 3060 - CUDA Active</span>
          </div>
        </header>

        {/* Dynamic Tab Content */}
        <section className="flex-1 overflow-y-auto p-6">
          {activeTab === 'chat' && <div className="max-w-4xl mx-auto">Chat Interface & All-Comparison Cards go here...</div>}
          {activeTab === 'evaluate' && <div className="text-center py-20">Evaluation Metrics Dashboard...</div>}
          {activeTab === 'ingest' && <div className="text-center py-20">Vector DB Management...</div>}
        </section>
      </main>
    </div>
  );
};

export default App;