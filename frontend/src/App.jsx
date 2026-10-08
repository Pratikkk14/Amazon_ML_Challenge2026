import React, { useState, useEffect } from 'react';
import { 
  Building2, 
  Layers, 
  ArrowRightLeft, 
  Sparkles, 
  Sliders, 
  Trash2, 
  Plus, 
  FileText, 
  CheckCircle2, 
  HelpCircle,
  ChevronDown,
  ChevronRight,
  GitMerge,
  Info,
  ExternalLink,
  RotateCw
} from 'lucide-react';

const API_BASE = "http://127.0.0.1:5000/api";

export default function App() {
  const [records, setRecords] = useState([
    { id: 'S1-001', name: 'Infosys Technologies Private Limited', addr: 'Electronics City, Hosur Road, Bengaluru' },
    { id: 'S2-045', name: 'INFOSYS LTD.', addr: 'Electronic City Phase 1, Bangalore' },
    { id: 'S3-102', name: 'Infosys Tech Pvt. Ltd.', addr: 'Hosur Rd, Electronic City, Bangalore' },
    { id: 'S1-002', name: 'Tata Consultancy Services Limited', addr: 'TCS House, Raveline Street, Fort, Mumbai' },
    { id: 'S2-088', name: 'TCS Ltd', addr: 'Fort, Mumbai, Maharashtra' },
    { id: 'S1-003', name: 'Wipro Enterprises Private Limited', addr: 'Doddakannelli, Sarjapur Road, Bangalore' },
    { id: 'S2-119', name: 'Wipro Ltd', addr: 'Sarjapur Rd, Doddakannelli, Bengaluru' },
    { id: 'S1-004', name: 'Reliance Retail Ventures Limited', addr: 'Maker Chambers IV, Nariman Point, Mumbai' }
  ]);

  const [threshold, setThreshold] = useState(0.72);
  const [robertaThreshold, setRobertaThreshold] = useState(0.68);
  const [loading, setLoading] = useState(false);
  const [resolutionResult, setResolutionResult] = useState(null);
  const [activeTab, setActiveTab] = useState('clusters'); // 'clusters' | 'pairwise' | 'input'
  const [expandedEntities, setExpandedEntities] = useState({});
  const [selectedPair, setSelectedPair] = useState(null);

  // New record input state
  const [newSource, setNewSource] = useState('S1');
  const [newName, setNewName] = useState('');
  const [newAddr, setNewAddr] = useState('');

  // Auto-resolve on initial mount
  useEffect(() => {
    handleResolve();
  }, []);

  const handleResolve = async () => {
    if (records.length < 2) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          records: records,
          threshold: threshold,
          roberta_threshold: robertaThreshold
        })
      });
      const data = await res.json();
      if (res.ok) {
        setResolutionResult(data);
        // Expand all merged clusters by default
        const initExpanded = {};
        data.entities.forEach(ent => {
          if (ent.is_merged) initExpanded[ent.entity_id] = true;
        });
        setExpandedEntities(initExpanded);
        if (data.pairwise_links && data.pairwise_links.length > 0) {
          setSelectedPair(data.pairwise_links[0]);
        }
      }
    } catch (err) {
      console.error("Failed to connect to backend", err);
    } finally {
      setLoading(false);
    }
  };

  const loadSample20 = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/sample-data`);
      const data = await res.json();
      if (data.records && data.records.length > 0) {
        setRecords(data.records);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const addCustomRecord = (e) => {
    e.preventDefault();
    if (!newName.trim()) return;
    const count = records.length + 1;
    const formattedId = `${newSource}-${String(count).padStart(3, '0')}`;
    setRecords([...records, { id: formattedId, name: newName.trim(), addr: newAddr.trim() }]);
    setNewName('');
    setNewAddr('');
  };

  const removeRecord = (idx) => {
    const updated = records.filter((_, i) => i !== idx);
    setRecords(updated);
  };

  const toggleExpand = (id) => {
    setExpandedEntities(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const getSourceBadge = (id = '') => {
    if (id.startsWith('S1')) return <span className="badge badge-source1">Source 1</span>;
    if (id.startsWith('S2')) return <span className="badge badge-source2">Source 2</span>;
    if (id.startsWith('S3')) return <span className="badge badge-source3">Source 3</span>;
    return <span className="badge badge-source1">{id}</span>;
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Minimal Navigation Bar */}
      <header style={{
        borderBottom: '1px solid var(--border-subtle)',
        padding: '14px 32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(19, 21, 27, 0.7)',
        backdropFilter: 'blur(12px)',
        position: 'sticky',
        top: 0,
        zIndex: 50
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, #6366f1, #a855f7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff'
          }}>
            <GitMerge size={18} />
          </div>
          <div>
            <h1 style={{ fontSize: '15px', fontWeight: '600', color: 'var(--text-main)', letterSpacing: '-0.2px' }}>
              EntityResolver
            </h1>
            <p style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
              LightGBM + Sublinear TF-IDF Multi-Source Resolution
            </p>
          </div>
        </div>

        {/* Global Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-input)', padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <Sliders size={14} color="var(--text-muted)" />
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>LGBM τ:</span>
            <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', fontWeight: '600', color: 'var(--accent-primary)', minWidth: '32px' }}>
              {threshold.toFixed(2)}
            </span>
            <input 
              type="range" 
              min="0.40" 
              max="0.95" 
              step="0.02" 
              value={threshold} 
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ width: '70px', accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-input)', padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <Sparkles size={14} color="#ec4899" />
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>RoBERTa τ:</span>
            <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', fontWeight: '600', color: '#ec4899', minWidth: '32px' }}>
              {robertaThreshold.toFixed(2)}
            </span>
            <input 
              type="range" 
              min="0.40" 
              max="0.95" 
              step="0.02" 
              value={robertaThreshold} 
              onChange={(e) => setRobertaThreshold(parseFloat(e.target.value))}
              style={{ width: '70px', accentColor: '#ec4899', cursor: 'pointer' }}
            />
          </div>

          <button 
            className="btn-ghost"
            onClick={loadSample20}
            title="Load 20-record realistic multi-source sample dataset"
          >
            <FileText size={14} />
            <span>Load 20-Entity Sample</span>
          </button>

          <button 
            className="btn-primary"
            onClick={handleResolve}
            disabled={loading || records.length < 2}
          >
            <RotateCw size={14} className={loading ? 'animate-spin' : ''} />
            <span>{loading ? 'Resolving...' : 'Run Resolution'}</span>
          </button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <main style={{ flex: 1, padding: '24px 32px', display: 'grid', gridTemplateColumns: '380px 1fr', gap: '24px', maxWidth: '1600px', width: '100%', margin: '0 auto' }}>
        
        {/* LEFT COLUMN: Input Entities Stream */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div className="glass-panel" style={{ padding: '18px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={16} color="var(--accent-primary)" />
                <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-main)' }}>
                  Active Records ({records.length})
                </span>
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                Multi-Source Input
              </span>
            </div>

            {/* Quick Add Form */}
            <form onSubmit={addCustomRecord} style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: 'var(--bg-input)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', gap: '8px' }}>
                <select 
                  value={newSource} 
                  onChange={(e) => setNewSource(e.target.value)}
                  style={{ background: 'var(--bg-card)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '6px 8px', fontSize: '12px', outline: 'none' }}
                >
                  <option value="S1">Source 1 (Ref)</option>
                  <option value="S2">Source 2</option>
                  <option value="S3">Source 3</option>
                </select>
                <input 
                  type="text" 
                  placeholder="Business Name..." 
                  value={newName} 
                  onChange={(e) => setNewName(e.target.value)}
                  style={{ flex: 1, background: 'var(--bg-card)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '6px 10px', fontSize: '12px', outline: 'none' }}
                />
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input 
                  type="text" 
                  placeholder="Address (e.g., Electronics City, Bangalore)..." 
                  value={newAddr} 
                  onChange={(e) => setNewAddr(e.target.value)}
                  style={{ flex: 1, background: 'var(--bg-card)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '6px 10px', fontSize: '12px', outline: 'none' }}
                />
                <button type="submit" className="btn-primary" style={{ padding: '6px 12px' }}>
                  <Plus size={14} />
                  <span>Add</span>
                </button>
              </div>
            </form>

            {/* Records List */}
            <div className="subtle-scroll" style={{ maxHeight: 'calc(100vh - 360px)', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', paddingRight: '4px' }}>
              {records.map((rec, i) => (
                <div 
                  key={i}
                  style={{
                    background: 'var(--bg-card-subtle)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    padding: '10px 12px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    gap: '10px',
                    transition: 'border-color 0.15s ease'
                  }}
                >
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', overflow: 'hidden' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {getSourceBadge(rec.id)}
                      <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                        {rec.id}
                      </span>
                    </div>
                    <span style={{ fontSize: '12px', fontWeight: '500', color: 'var(--text-main)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {rec.name}
                    </span>
                    {rec.addr && (
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {rec.addr}
                      </span>
                    )}
                  </div>

                  <button 
                    onClick={() => removeRecord(i)}
                    style={{ background: 'transparent', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', padding: '4px', borderRadius: '4px' }}
                    title="Remove record"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* RIGHT COLUMN: Interactive Resolution Canvas */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          {/* Metrics & Overview Ribbon */}
          {resolutionResult && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
              <div className="glass-panel" style={{ padding: '14px 18px' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Total Records</span>
                <div style={{ fontSize: '22px', fontWeight: '700', color: 'var(--text-main)', marginTop: '4px' }}>
                  {resolutionResult.total_records}
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '14px 18px', borderLeft: '3px solid var(--accent-primary)' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Unique Real Entities</span>
                <div style={{ fontSize: '22px', fontWeight: '700', color: 'var(--accent-primary)', marginTop: '4px' }}>
                  {resolutionResult.unique_entities_count}
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '14px 18px', borderLeft: '3px solid var(--status-match)' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Merged Clusters</span>
                <div style={{ fontSize: '22px', fontWeight: '700', color: 'var(--status-match)', marginTop: '4px' }}>
                  {resolutionResult.merged_clusters_count}
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '14px 18px' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Singletons (No Merges)</span>
                <div style={{ fontSize: '22px', fontWeight: '700', color: 'var(--text-muted)', marginTop: '4px' }}>
                  {resolutionResult.singletons_count}
                </div>
              </div>
            </div>
          )}

          {/* View Tab Selector */}
          <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
            <button 
              onClick={() => setActiveTab('clusters')}
              style={{
                background: activeTab === 'clusters' ? 'var(--bg-card-subtle)' : 'transparent',
                color: activeTab === 'clusters' ? 'var(--text-main)' : 'var(--text-dim)',
                border: activeTab === 'clusters' ? '1px solid var(--border-active)' : '1px solid transparent',
                borderRadius: '6px',
                padding: '6px 14px',
                fontSize: '13px',
                fontWeight: '500',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <GitMerge size={14} />
              <span>Resolved Entity Clusters</span>
            </button>

            <button 
              onClick={() => setActiveTab('pairwise')}
              style={{
                background: activeTab === 'pairwise' ? 'var(--bg-card-subtle)' : 'transparent',
                color: activeTab === 'pairwise' ? 'var(--text-main)' : 'var(--text-dim)',
                border: activeTab === 'pairwise' ? '1px solid var(--border-active)' : '1px solid transparent',
                borderRadius: '6px',
                padding: '6px 14px',
                fontSize: '13px',
                fontWeight: '500',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <ArrowRightLeft size={14} />
              <span>Pairwise Link Inspection ({resolutionResult?.pairwise_links?.length || 0})</span>
            </button>
          </div>

          {/* TAB 1: BEAUTIFIED CLUSTERS & MERGED ENTITY CARDS */}
          {activeTab === 'clusters' && resolutionResult && (
            <div className="subtle-scroll" style={{ maxHeight: 'calc(100vh - 280px)', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {resolutionResult.entities.map((entity, idx) => {
                const isExpanded = expandedEntities[entity.entity_id];
                return (
                  <div 
                    key={entity.entity_id}
                    className="glass-panel"
                    style={{
                      border: entity.is_merged ? '1px solid var(--status-match-border)' : '1px solid var(--border-subtle)',
                      background: entity.is_merged ? 'linear-gradient(180deg, rgba(16, 185, 129, 0.04) 0%, rgba(19, 21, 27, 0.95) 100%)' : 'var(--bg-card)',
                      overflow: 'hidden',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    {/* Header Banner */}
                    <div 
                      onClick={() => toggleExpand(entity.entity_id)}
                      style={{
                        padding: '14px 18px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        cursor: 'pointer',
                        userSelect: 'none'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <div style={{
                          width: '28px',
                          height: '28px',
                          borderRadius: '6px',
                          background: entity.is_merged ? 'var(--status-match-bg)' : 'var(--bg-card-subtle)',
                          color: entity.is_merged ? 'var(--status-match)' : 'var(--text-dim)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}>
                          {entity.is_merged ? <CheckCircle2 size={16} /> : <Building2 size={16} />}
                        </div>

                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-main)' }}>
                              {entity.canonical_name}
                            </span>
                            <span style={{
                              fontSize: '10px',
                              padding: '2px 6px',
                              borderRadius: '4px',
                              fontWeight: '600',
                              background: entity.is_merged ? 'rgba(16, 185, 129, 0.15)' : 'rgba(148, 163, 184, 0.1)',
                              color: entity.is_merged ? 'var(--status-match)' : 'var(--text-dim)'
                            }}>
                              {entity.is_merged ? `${entity.record_count} Records Merged` : 'Singleton'}
                            </span>
                          </div>
                          {entity.canonical_address && (
                            <span style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'block', marginTop: '2px' }}>
                              📍 {entity.canonical_address}
                            </span>
                          )}
                        </div>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                          {entity.entity_id}
                        </span>
                        {isExpanded ? <ChevronDown size={16} color="var(--text-muted)" /> : <ChevronRight size={16} color="var(--text-muted)" />}
                      </div>
                    </div>

                    {/* Merged Inner Records Drawer */}
                    {isExpanded && (
                      <div style={{ borderTop: '1px solid var(--border-subtle)', background: 'rgba(11, 12, 16, 0.5)', padding: '14px 18px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <span style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                          Linked Multi-Source Fragments:
                        </span>
                        
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '10px' }}>
                          {entity.records.map((r, rIdx) => (
                            <div 
                              key={rIdx}
                              style={{
                                background: 'var(--bg-card)',
                                border: '1px solid var(--border-subtle)',
                                borderRadius: '8px',
                                padding: '10px 12px',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: '4px'
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                {getSourceBadge(r.id)}
                                <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                                  ID: {r.id}
                                </span>
                              </div>
                              <span style={{ fontSize: '12px', fontWeight: '500', color: 'var(--text-main)', marginTop: '2px' }}>
                                {r.name}
                              </span>
                              {r.addr ? (
                                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                                  {r.addr}
                                </span>
                              ) : (
                                <span style={{ fontSize: '11px', color: 'var(--text-dim)', fontStyle: 'italic' }}>
                                  (No address specified)
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* TAB 2: DETAILED PAIRWISE FEATURE INSPECTOR */}
          {activeTab === 'pairwise' && resolutionResult && (
            <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '16px', maxHeight: 'calc(100vh - 280px)' }}>
              
              {/* Pairwise Links Scrollable List */}
              <div className="glass-panel subtle-scroll" style={{ overflowY: 'auto', padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {resolutionResult.pairwise_links.map((link, lIdx) => {
                  const isSelected = selectedPair === link;
                  return (
                    <div 
                      key={lIdx}
                      onClick={() => setSelectedPair(link)}
                      style={{
                        padding: '10px',
                        borderRadius: '6px',
                        background: isSelected ? 'var(--bg-card-subtle)' : 'transparent',
                        border: isSelected ? '1px solid var(--accent-primary)' : '1px solid transparent',
                        cursor: 'pointer',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '4px'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                          {link.source_id} ↔ {link.target_id}
                        </span>
                        <span style={{
                          fontSize: '11px',
                          fontWeight: '700',
                          fontFamily: 'var(--font-mono)',
                          color: link.is_match ? 'var(--status-match)' : 'var(--text-dim)'
                        }}>
                          {(link.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <span style={{ fontSize: '12px', color: 'var(--text-main)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {link.source_name}
                      </span>
                    </div>
                  );
                })}
              </div>

              {/* Selected Pair Deep Inspector Panel */}
              {selectedPair ? (
                <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '18px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '14px' }}>
                    <div>
                      <h3 style={{ fontSize: '15px', fontWeight: '600', color: 'var(--text-main)' }}>
                        Pairwise Link Analysis
                      </h3>
                      <p style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
                        LightGBM Tree-Level Feature Vector Decomposition
                      </p>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{ textAlign: 'right' }}>
                        <span style={{ fontSize: '11px', color: 'var(--text-dim)', display: 'block' }}>Match Probability</span>
                        <span style={{ fontSize: '18px', fontWeight: '700', fontFamily: 'var(--font-mono)', color: selectedPair.is_match ? 'var(--status-match)' : 'var(--text-dim)' }}>
                          {(selectedPair.probability * 100).toFixed(2)}%
                        </span>
                      </div>
                      <span style={{
                        padding: '4px 10px',
                        borderRadius: '6px',
                        fontWeight: '700',
                        fontSize: '11px',
                        background: selectedPair.is_match ? 'var(--status-match-bg)' : 'rgba(148, 163, 184, 0.1)',
                        color: selectedPair.is_match ? 'var(--status-match)' : 'var(--text-dim)'
                      }}>
                        {selectedPair.is_match ? 'MATCH (Merged)' : 'DISTINCT'}
                      </span>
                    </div>
                  </div>

                  {/* Side by Side Comparison */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
                    <div style={{ background: 'var(--bg-input)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                        {getSourceBadge(selectedPair.source_id)}
                        <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>{selectedPair.source_id}</span>
                      </div>
                      <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-main)' }}>{selectedPair.source_name}</div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>{selectedPair.source_addr || '(No address)'}</div>
                    </div>

                    <div style={{ background: 'var(--bg-input)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                        {getSourceBadge(selectedPair.target_id)}
                        <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>{selectedPair.target_id}</span>
                      </div>
                      <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-main)' }}>{selectedPair.target_name}</div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>{selectedPair.target_addr || '(No address)'}</div>
                    </div>
                  </div>

                  {/* Feature Breakdown Bars */}
                  <div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
                      {/* LightGBM Card */}
                      <div style={{ background: 'var(--bg-card-subtle)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                          <span style={{ fontSize: '12px', fontWeight: '600', color: 'var(--accent-primary)' }}>LightGBM Classifier</span>
                          <span style={{ fontSize: '11px', padding: '2px 6px', borderRadius: '4px', background: selectedPair.lgbm_is_match ? 'var(--status-match-bg)' : 'rgba(148,163,184,0.1)', color: selectedPair.lgbm_is_match ? 'var(--status-match)' : 'var(--text-dim)' }}>
                            {selectedPair.lgbm_is_match ? 'MATCH' : 'NO MATCH'}
                          </span>
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: '700', fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}>
                          {((selectedPair.lgbm_prob || selectedPair.probability) * 100).toFixed(1)}%
                        </div>
                      </div>

                      {/* RoBERTa Transformer Card */}
                      <div style={{ background: 'var(--bg-card-subtle)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                          <span style={{ fontSize: '12px', fontWeight: '600', color: '#ec4899' }}>RoBERTa Embedding</span>
                          <span style={{ fontSize: '11px', padding: '2px 6px', borderRadius: '4px', background: selectedPair.roberta_is_match ? 'rgba(236,72,153,0.15)' : 'rgba(148,163,184,0.1)', color: selectedPair.roberta_is_match ? '#ec4899' : 'var(--text-dim)' }}>
                            {selectedPair.roberta_is_match ? 'MATCH' : 'NO MATCH'}
                          </span>
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: '700', fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}>
                          {((selectedPair.roberta_sim || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    <span style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                      Computed NLP & Semantic Metrics:
                    </span>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '10px' }}>
                      
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                          <span style={{ color: 'var(--text-muted)' }}>TF-IDF Cosine Similarity</span>
                          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}>{selectedPair.features.tfidf_sim}</span>
                        </div>
                        <div style={{ height: '6px', background: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${selectedPair.features.tfidf_sim * 100}%`, height: '100%', background: 'var(--accent-primary)' }}></div>
                        </div>
                      </div>

                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Name Token Jaccard</span>
                          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}>{selectedPair.features.name_jaccard}</span>
                        </div>
                        <div style={{ height: '6px', background: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${selectedPair.features.name_jaccard * 100}%`, height: '100%', background: '#a855f7' }}></div>
                        </div>
                      </div>

                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Character 3-Gram Overlap</span>
                          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}>{selectedPair.features.name_3gram}</span>
                        </div>
                        <div style={{ height: '6px', background: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${selectedPair.features.name_3gram * 100}%`, height: '100%', background: '#ec4899' }}></div>
                        </div>
                      </div>

                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Address Token Jaccard</span>
                          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}>{selectedPair.features.addr_jaccard}</span>
                        </div>
                        <div style={{ height: '6px', background: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${selectedPair.features.addr_jaccard * 100}%`, height: '100%', background: '#f59e0b' }}></div>
                        </div>
                      </div>

                    </div>
                  </div>

                </div>
              ) : (
                <div className="glass-panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)' }}>
                  Select a pairwise comparison from the left
                </div>
              )}
            </div>
          )}

        </section>
      </main>
    </div>
  );
}
