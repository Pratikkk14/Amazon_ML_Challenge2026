import React, { useState, useEffect } from 'react';
import { 
  Building2, 
  Layers, 
  ArrowRightLeft, 
  Sliders, 
  Trash2, 
  Plus, 
  FileText, 
  Check, 
  Minus,
  ChevronDown, 
  ChevronRight, 
  GitMerge, 
  RotateCw,
  Cpu,
  ShieldCheck,
  AlertTriangle
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
  const [ensembleStrategy, setEnsembleStrategy] = useState('weighted'); // 'weighted' | 'consensus' | 'cascade' | 'voting'
  const [loading, setLoading] = useState(false);
  const [resolutionResult, setResolutionResult] = useState(null);
  const [activeTab, setActiveTab] = useState('clusters'); // 'clusters' | 'pairwise'
  const [expandedEntities, setExpandedEntities] = useState({});
  const [selectedPair, setSelectedPair] = useState(null);

  // New record input
  const [newSource, setNewSource] = useState('S1');
  const [newName, setNewName] = useState('');
  const [newAddr, setNewAddr] = useState('');

  useEffect(() => {
    handleResolve();
  }, [threshold, robertaThreshold, ensembleStrategy]);

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
          roberta_threshold: robertaThreshold,
          strategy: ensembleStrategy
        })
      });
      const data = await res.json();
      if (res.ok) {
        setResolutionResult(data);
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
      console.error("Backend connection error:", err);
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
    const nextNum = records.length + 1;
    const newId = `${newSource}-${String(nextNum).padStart(3, '0')}`;
    setRecords([...records, { id: newId, name: newName.trim(), addr: newAddr.trim() }]);
    setNewName('');
    setNewAddr('');
  };

  const removeRecord = (indexToRemove) => {
    if (records.length <= 2) return;
    setRecords(records.filter((_, idx) => idx !== indexToRemove));
  };

  const toggleExpand = (id) => {
    setExpandedEntities(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const getSourceBadge = (id) => {
    const isS1 = id.startsWith('S1');
    const isS2 = id.startsWith('S2');
    const badgeClass = isS1 ? 'sq-badge-src1' : isS2 ? 'sq-badge-src2' : 'sq-badge-src3';
    return (
      <span className={`sq-badge ${badgeClass}`}>
        {isS1 ? 'SRC-1 (REF)' : isS2 ? 'SRC-2' : 'SRC-3'}
      </span>
    );
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-page)' }}>
      
      {/* 2D Minimalist AWS Builder Center Header */}
      <header className="sq-card sq-border-b" style={{ padding: '12px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: 'none', borderLeft: 'none', borderRight: 'none' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: '28px', height: '28px', background: '#ffffff', color: '#000000', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: '800', fontSize: '13px', fontFamily: 'var(--font-mono)' }}>
            ER
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: '700', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
              Entity Resolution System
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
              Dual Model Ensemble · LightGBM + RoBERTa Reranker
            </div>
          </div>
        </div>

        {/* Action Controls & Parameter Sliders */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          
          {/* Ensemble Strategy Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: '#0a0a0a', border: '1px solid var(--border-light)', padding: '4px 8px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>Ensemble:</span>
            <select 
              value={ensembleStrategy} 
              onChange={(e) => setEnsembleStrategy(e.target.value)}
              style={{ background: '#000000', color: '#ffffff', border: '1px solid var(--border-medium)', fontSize: '11px', padding: '3px 6px', outline: 'none', fontFamily: 'var(--font-mono)' }}
            >
              <option value="weighted">Weighted Fusion (65/35)</option>
              <option value="cascade">Cascading Disambiguation</option>
              <option value="consensus">Strict Agreement (Precision)</option>
              <option value="voting">Union / Majority (Recall)</option>
            </select>
          </div>

          {/* LightGBM Threshold */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: '#0a0a0a', border: '1px solid var(--border-light)', padding: '4px 8px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>LGBM τ:</span>
            <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', fontWeight: '700', minWidth: '28px' }}>
              {threshold.toFixed(2)}
            </span>
            <input 
              type="range" 
              min="0.40" 
              max="0.95" 
              step="0.02" 
              value={threshold} 
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ width: '60px' }}
            />
          </div>

          {/* RoBERTa Threshold */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: '#0a0a0a', border: '1px solid var(--border-light)', padding: '4px 8px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>RoBERTa τ:</span>
            <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', fontWeight: '700', minWidth: '28px' }}>
              {robertaThreshold.toFixed(2)}
            </span>
            <input 
              type="range" 
              min="0.40" 
              max="0.95" 
              step="0.02" 
              value={robertaThreshold} 
              onChange={(e) => setRobertaThreshold(parseFloat(e.target.value))}
              style={{ width: '60px' }}
            />
          </div>

          <button className="sq-btn-secondary" onClick={loadSample20}>
            <FileText size={12} />
            <span>Load 20-Sample</span>
          </button>

          <button className="sq-btn-primary" onClick={handleResolve} disabled={loading}>
            <RotateCw size={12} className={loading ? 'animate-spin' : ''} />
            <span>{loading ? 'Evaluating...' : 'Resolve'}</span>
          </button>
        </div>
      </header>

      {/* Main 2D Grid Layout */}
      <main style={{ flex: 1, padding: '20px 24px', display: 'grid', gridTemplateColumns: '360px 1fr', gap: '20px', maxWidth: '1680px', width: '100%', margin: '0 auto' }}>
        
        {/* Left Column: Data Records Feed */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div className="sq-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-light)', paddingBottom: '8px' }}>
              <div style={{ fontSize: '12px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Active Records ({records.length})
              </div>
              <span style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                MULTI-SOURCE STREAM
              </span>
            </div>

            {/* Quick Add Form */}
            <form onSubmit={addCustomRecord} style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ display: 'flex', gap: '6px' }}>
                <select 
                  value={newSource} 
                  onChange={(e) => setNewSource(e.target.value)}
                  style={{ background: '#000000', color: '#ffffff', border: '1px solid var(--border-medium)', padding: '5px 8px', fontSize: '11px', outline: 'none', fontFamily: 'var(--font-mono)' }}
                >
                  <option value="S1">S1</option>
                  <option value="S2">S2</option>
                  <option value="S3">S3</option>
                </select>
                <input 
                  type="text" 
                  placeholder="Entity name..." 
                  value={newName} 
                  onChange={(e) => setNewName(e.target.value)}
                  style={{ flex: 1, background: '#000000', color: '#ffffff', border: '1px solid var(--border-medium)', padding: '5px 8px', fontSize: '11px', outline: 'none' }}
                />
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <input 
                  type="text" 
                  placeholder="Business address..." 
                  value={newAddr} 
                  onChange={(e) => setNewAddr(e.target.value)}
                  style={{ flex: 1, background: '#000000', color: '#ffffff', border: '1px solid var(--border-medium)', padding: '5px 8px', fontSize: '11px', outline: 'none' }}
                />
                <button type="submit" className="sq-btn-secondary" style={{ padding: '5px 10px' }}>
                  <Plus size={12} />
                  <span>Add</span>
                </button>
              </div>
            </form>

            {/* Records Stream */}
            <div className="sq-scroll" style={{ maxHeight: 'calc(100vh - 350px)', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px', paddingRight: '2px' }}>
              {records.map((rec, i) => (
                <div 
                  key={i}
                  style={{
                    background: '#0d0d0d',
                    border: '1px solid var(--border-light)',
                    padding: '8px 10px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    gap: '8px'
                  }}
                >
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', overflow: 'hidden' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {getSourceBadge(rec.id)}
                      <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
                        {rec.id}
                      </span>
                    </div>
                    <div style={{ fontSize: '12px', fontWeight: '600', color: '#ffffff', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {rec.name}
                    </div>
                    {rec.addr && (
                      <div style={{ fontSize: '11px', color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {rec.addr}
                      </div>
                    )}
                  </div>

                  <button 
                    onClick={() => removeRecord(i)}
                    style={{ background: 'transparent', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer', padding: '2px' }}
                    title="Remove record"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Right Column: 2D Resolution Canvas */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          
          {/* Top Summary Metrics Bar (Square 2D Cards) */}
          {resolutionResult && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
              <div className="sq-card" style={{ padding: '12px 16px' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-secondary)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  Total Inputs
                </div>
                <div style={{ fontSize: '20px', fontWeight: '800', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                  {resolutionResult.total_records}
                </div>
              </div>

              <div className="sq-card" style={{ padding: '12px 16px', borderLeft: '3px solid #ffffff' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-secondary)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  Resolved Unique
                </div>
                <div style={{ fontSize: '20px', fontWeight: '800', fontFamily: 'var(--font-mono)', marginTop: '2px', color: '#ffffff' }}>
                  {resolutionResult.unique_entities_count}
                </div>
              </div>

              <div className="sq-card" style={{ padding: '12px 16px', borderLeft: '3px solid var(--status-match-border)' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-secondary)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  Merged Clusters
                </div>
                <div style={{ fontSize: '20px', fontWeight: '800', fontFamily: 'var(--font-mono)', marginTop: '2px', color: 'var(--status-match-text)' }}>
                  {resolutionResult.merged_clusters_count}
                </div>
              </div>

              <div className="sq-card" style={{ padding: '12px 16px' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-secondary)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  Singletons
                </div>
                <div style={{ fontSize: '20px', fontWeight: '800', fontFamily: 'var(--font-mono)', marginTop: '2px', color: 'var(--text-secondary)' }}>
                  {resolutionResult.singletons_count}
                </div>
              </div>
            </div>
          )}

          {/* Tab Selector */}
          <div style={{ display: 'flex', gap: '0px', borderBottom: '1px solid var(--border-light)' }}>
            <button 
              onClick={() => setActiveTab('clusters')}
              style={{
                background: activeTab === 'clusters' ? '#171717' : 'transparent',
                color: activeTab === 'clusters' ? '#ffffff' : 'var(--text-secondary)',
                border: '1px solid var(--border-light)',
                borderBottom: activeTab === 'clusters' ? '1px solid #171717' : '1px solid var(--border-light)',
                padding: '6px 16px',
                fontSize: '11px',
                fontWeight: '600',
                cursor: 'pointer',
                textTransform: 'uppercase',
                fontFamily: 'var(--font-mono)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <GitMerge size={12} />
              <span>Resolved Clusters</span>
            </button>

            <button 
              onClick={() => setActiveTab('pairwise')}
              style={{
                background: activeTab === 'pairwise' ? '#171717' : 'transparent',
                color: activeTab === 'pairwise' ? '#ffffff' : 'var(--text-secondary)',
                border: '1px solid var(--border-light)',
                borderLeft: 'none',
                borderBottom: activeTab === 'pairwise' ? '1px solid #171717' : '1px solid var(--border-light)',
                padding: '6px 16px',
                fontSize: '11px',
                fontWeight: '600',
                cursor: 'pointer',
                textTransform: 'uppercase',
                fontFamily: 'var(--font-mono)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <ArrowRightLeft size={12} />
              <span>Pairwise Link Inspection ({resolutionResult?.pairwise_links?.length || 0})</span>
            </button>
          </div>

          {/* TAB 1: 2D Square Clusters */}
          {activeTab === 'clusters' && resolutionResult && (
            <div className="sq-scroll" style={{ maxHeight: 'calc(100vh - 280px)', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {resolutionResult.entities.map((entity) => {
                const isExpanded = expandedEntities[entity.entity_id];
                return (
                  <div 
                    key={entity.entity_id}
                    className="sq-card"
                    style={{
                      borderLeft: entity.is_merged ? '3px solid var(--status-match-border)' : '1px solid var(--border-light)',
                      background: entity.is_merged ? '#0e1711' : '#0a0a0a'
                    }}
                  >
                    <div 
                      onClick={() => toggleExpand(entity.entity_id)}
                      style={{
                        padding: '12px 14px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        cursor: 'pointer'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '13px', fontWeight: '700', color: '#ffffff' }}>
                              {entity.canonical_name}
                            </span>
                            <span className="sq-badge" style={{
                              borderColor: entity.is_merged ? 'var(--status-match-border)' : 'var(--border-medium)',
                              color: entity.is_merged ? 'var(--status-match-text)' : 'var(--text-secondary)'
                            }}>
                              {entity.is_merged ? `${entity.record_count} Records Merged` : 'Singleton'}
                            </span>
                          </div>
                          {entity.canonical_address && (
                            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                              {entity.canonical_address}
                            </div>
                          )}
                        </div>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
                          {entity.entity_id}
                        </span>
                        {isExpanded ? <ChevronDown size={14} color="var(--text-secondary)" /> : <ChevronRight size={14} color="var(--text-secondary)" />}
                      </div>
                    </div>

                    {isExpanded && (
                      <div style={{ borderTop: '1px solid var(--border-light)', background: '#000000', padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        <div style={{ fontSize: '10px', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
                          Linked Multi-Source Records:
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '6px' }}>
                          {entity.records.map((r, rIdx) => (
                            <div key={rIdx} style={{ background: '#0a0a0a', border: '1px solid var(--border-light)', padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                {getSourceBadge(r.id)}
                                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>ID: {r.id}</span>
                              </div>
                              <div style={{ fontSize: '11px', fontWeight: '600', color: '#ffffff', marginTop: '2px' }}>{r.name}</div>
                              {r.addr ? (
                                <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>{r.addr}</div>
                              ) : (
                                <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontStyle: 'italic' }}>(No address specified)</div>
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

          {/* TAB 2: Pairwise Link Inspection */}
          {activeTab === 'pairwise' && resolutionResult && (
            <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '12px', maxHeight: 'calc(100vh - 280px)' }}>
              
              {/* Pairwise Links Left List */}
              <div className="sq-card sq-scroll" style={{ overflowY: 'auto', padding: '6px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {resolutionResult.pairwise_links.map((link, lIdx) => {
                  const isSelected = selectedPair === link;
                  return (
                    <div 
                      key={lIdx}
                      onClick={() => setSelectedPair(link)}
                      style={{
                        padding: '8px 10px',
                        background: isSelected ? '#171717' : '#0a0a0a',
                        border: isSelected ? '1px solid #ffffff' : '1px solid var(--border-light)',
                        cursor: 'pointer',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '2px'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
                          {link.source_id} ↔ {link.target_id}
                        </span>
                        <span style={{
                          fontSize: '11px',
                          fontWeight: '700',
                          fontFamily: 'var(--font-mono)',
                          color: link.is_match ? 'var(--status-match-text)' : 'var(--text-tertiary)'
                        }}>
                          {(link.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ fontSize: '11px', color: '#ffffff', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {link.source_name}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Selected Pair Deep Inspector */}
              {selectedPair ? (
                <div className="sq-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  
                  {/* Top Header */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-light)', paddingBottom: '10px' }}>
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                        Pairwise Link Decision Analysis
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                        Status: {selectedPair.consensus_status}
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>Ensemble Score</div>
                        <div style={{ fontSize: '16px', fontWeight: '800', fontFamily: 'var(--font-mono)', color: selectedPair.is_match ? 'var(--status-match-text)' : 'var(--text-secondary)' }}>
                          {(selectedPair.probability * 100).toFixed(2)}%
                        </div>
                      </div>
                      <span className="sq-badge" style={{
                        borderColor: selectedPair.is_match ? 'var(--status-match-border)' : 'var(--border-medium)',
                        color: selectedPair.is_match ? 'var(--status-match-text)' : 'var(--text-secondary)',
                        padding: '4px 8px'
                      }}>
                        {selectedPair.is_match ? 'MATCH (MERGED)' : 'DISTINCT'}
                      </span>
                    </div>
                  </div>

                  {/* 2-Column Record Comparison */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    <div style={{ background: '#000000', border: '1px solid var(--border-light)', padding: '10px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                        {getSourceBadge(selectedPair.source_id)}
                        <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>{selectedPair.source_id}</span>
                      </div>
                      <div style={{ fontSize: '12px', fontWeight: '600', color: '#ffffff' }}>{selectedPair.source_name}</div>
                      <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>{selectedPair.source_addr || '(No address)'}</div>
                    </div>

                    <div style={{ background: '#000000', border: '1px solid var(--border-light)', padding: '10px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                        {getSourceBadge(selectedPair.target_id)}
                        <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>{selectedPair.target_id}</span>
                      </div>
                      <div style={{ fontSize: '12px', fontWeight: '600', color: '#ffffff' }}>{selectedPair.target_name}</div>
                      <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>{selectedPair.target_addr || '(No address)'}</div>
                    </div>
                  </div>

                  {/* Dual Model Probability Cards */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    
                    {/* LightGBM Model */}
                    <div style={{ background: '#000000', border: '1px solid var(--border-light)', padding: '10px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                          1. LightGBM (Lexical/N-Gram)
                        </span>
                        <span className="sq-badge" style={{ borderColor: selectedPair.lgbm_is_match ? 'var(--status-match-border)' : 'var(--border-medium)', color: selectedPair.lgbm_is_match ? 'var(--status-match-text)' : 'var(--text-tertiary)' }}>
                          {selectedPair.lgbm_is_match ? 'MATCH' : 'NO MATCH'}
                        </span>
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: '800', fontFamily: 'var(--font-mono)' }}>
                        {((selectedPair.lgbm_prob || 0) * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                        Evaluates 11-D tree feature interactions (Jaccard, n-grams, legal suffix).
                      </div>
                    </div>

                    {/* RoBERTa Model */}
                    <div style={{ background: '#000000', border: '1px solid var(--border-light)', padding: '10px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                          2. RoBERTa (Semantic Embedding)
                        </span>
                        <span className="sq-badge" style={{ borderColor: selectedPair.roberta_is_match ? 'var(--status-match-border)' : 'var(--border-medium)', color: selectedPair.roberta_is_match ? 'var(--status-match-text)' : 'var(--text-tertiary)' }}>
                          {selectedPair.roberta_is_match ? 'MATCH' : 'NO MATCH'}
                        </span>
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: '800', fontFamily: 'var(--font-mono)' }}>
                        {((selectedPair.roberta_sim || 0) * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                        Evaluates deep neural embedding cosine similarity in 768-D dense space.
                      </div>
                    </div>
                  </div>

                  {/* 2D Metrics Breakdown */}
                  <div>
                    <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                      Decomposed Feature Metrics:
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                      
                      <div style={{ background: '#000000', border: '1px solid var(--border-light)', padding: '8px' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>TF-IDF COSINE</div>
                        <div style={{ fontSize: '13px', fontWeight: '700', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>{selectedPair.features.tfidf_sim}</div>
                      </div>

                      <div style={{ background: '#000000', border: '1px solid var(--border-light)', padding: '8px' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>NAME JACCARD</div>
                        <div style={{ fontSize: '13px', fontWeight: '700', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>{selectedPair.features.name_jaccard}</div>
                      </div>

                      <div style={{ background: '#000000', border: '1px solid var(--border-light)', padding: '8px' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>CHAR 3-GRAM</div>
                        <div style={{ fontSize: '13px', fontWeight: '700', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>{selectedPair.features.name_3gram}</div>
                      </div>

                      <div style={{ background: '#000000', border: '1px solid var(--border-light)', padding: '8px' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>ADDR JACCARD</div>
                        <div style={{ fontSize: '13px', fontWeight: '700', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>{selectedPair.features.addr_jaccard}</div>
                      </div>

                      <div style={{ background: '#000000', border: '1px solid var(--border-light)', padding: '8px' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>SUBSTRING</div>
                        <div style={{ fontSize: '13px', fontWeight: '700', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>{selectedPair.features.is_substr ? 'TRUE' : 'FALSE'}</div>
                      </div>

                      <div style={{ background: '#000000', border: '1px solid var(--border-light)', padding: '8px' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>EXACT NAME</div>
                        <div style={{ fontSize: '13px', fontWeight: '700', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>{selectedPair.features.exact_name ? 'TRUE' : 'FALSE'}</div>
                      </div>

                    </div>
                  </div>

                </div>
              ) : null}
            </div>
          )}

        </section>
      </main>
    </div>
  );
}
