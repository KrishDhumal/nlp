import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  UploadCloud, FileText, AlertTriangle, ShieldAlert, Loader2, Lightbulb, Download, 
  MessageSquare, Layers, Building2, Calendar, DollarSign, Scale, Tag, CheckCircle2, 
  ChevronRight, Filter, Bookmark, Info, ArrowUpRight 
} from 'lucide-react';
import ChatPanel from '../components/ChatPanel';
import SourceCitation from '../components/SourceCitation';
import api from '../services/api';

/** Returns Tailwind colour classes based on a 0-100 risk score. */
const getSeverityColor = (score) => {
  if (score > 80) return { text: 'text-emerald-600 dark:text-emerald-400', bg: 'bg-emerald-50 dark:bg-emerald-500/10', border: 'border-emerald-200 dark:border-emerald-500/20', bar: '#10b981' };
  if (score >= 50) return { text: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-50 dark:bg-amber-500/10', border: 'border-amber-200 dark:border-amber-500/20', bar: '#f59e0b' };
  return { text: 'text-red-600 dark:text-red-400', bg: 'bg-red-50 dark:bg-red-500/10', border: 'border-red-200 dark:border-red-500/20', bar: '#ef4444' };
};

const getRiskBadgeStyle = (level) => {
  const l = (level || '').toLowerCase();
  if (l === 'critical') return 'bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-400 border-red-200 dark:border-red-500/30';
  if (l === 'high') return 'bg-orange-100 text-orange-700 dark:bg-orange-500/20 dark:text-orange-400 border-orange-200 dark:border-orange-500/30';
  if (l === 'medium') return 'bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-400 border-amber-200 dark:border-amber-500/30';
  return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/30';
};

const getClauseCategoryColor = (category) => {
  const map = {
    'Termination': 'bg-red-50 text-red-700 dark:bg-red-500/10 dark:text-red-400 border-red-200 dark:border-red-500/20',
    'Payment': 'bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/20',
    'Liability': 'bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400 border-amber-200 dark:border-amber-500/20',
    'Indemnification': 'bg-purple-50 text-purple-700 dark:bg-purple-500/10 dark:text-purple-400 border-purple-200 dark:border-purple-500/20',
    'Confidentiality': 'bg-blue-50 text-blue-700 dark:bg-blue-500/10 dark:text-blue-400 border-blue-200 dark:border-blue-500/20',
    'Governing Law': 'bg-indigo-50 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-400 border-indigo-200 dark:border-indigo-500/20',
    'IP Rights': 'bg-cyan-50 text-cyan-700 dark:bg-cyan-500/10 dark:text-cyan-400 border-cyan-200 dark:border-cyan-500/20',
  };
  return map[category] || 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 border-slate-200 dark:border-slate-700';
};

/** The seven core clause categories the backend classifies into, plus "All". */
const CLAUSE_FILTERS = [
  'All', 'Termination', 'Payment', 'Liability',
  'Indemnification', 'Confidentiality', 'Governing Law', 'IP Rights',
];

const MAX_UPLOAD_MB = 20;

/**
 * Renders `text` with `terms` marked.
 *
 * The terms are the phrases the clause detector actually matched on, so the
 * highlight shows the reader why this passage was flagged. Longest terms are
 * matched first so "prior written notice" wins over "notice", and the regex is
 * escaped because legal phrases contain punctuation.
 */
const HighlightedText = ({ text, terms = [], className = '' }) => {
  if (!text) return null;
  const cleaned = (terms || []).filter(Boolean).sort((a, b) => b.length - a.length);
  if (cleaned.length === 0) return <span className={className}>{text}</span>;

  const escaped = cleaned.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  let parts;
  try {
    parts = text.split(new RegExp(`(${escaped.join('|')})`, 'gi'));
  } catch {
    return <span className={className}>{text}</span>;
  }

  const lookup = new Set(cleaned.map(t => t.toLowerCase()));
  return (
    <span className={className}>
      {parts.map((part, i) =>
        lookup.has((part || '').toLowerCase()) ? (
          <mark
            key={i}
            className="bg-amber-200/70 dark:bg-amber-400/30 text-slate-900 dark:text-amber-100 rounded px-0.5 font-semibold not-italic"
          >
            {part}
          </mark>
        ) : (
          <React.Fragment key={i}>{part}</React.Fragment>
        )
      )}
    </span>
  );
};

/** Consistent placeholder for a tab or grid the audit returned nothing for. */
const EmptyState = ({ icon: Icon, message }) => (
  <div className="flex flex-col items-center justify-center py-10 text-center gap-2">
    <Icon size={26} className="text-slate-300 dark:text-slate-600" />
    <p className="text-xs text-slate-500 dark:text-slate-400 max-w-xs">{message}</p>
  </div>
);

const AuditNew = () => {
  const [stage, setStage] = useState('upload'); // 'upload' | 'loading' | 'analysis'
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('summary'); // 'summary' | 'clauses' | 'entities' | 'risks' | 'suggestions'
  const [clauseFilter, setClauseFilter] = useState('All');
  const [auditResult, setAuditResult] = useState(null);
  const [error, setError] = useState(null);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [progress, setProgress] = useState({ percent: 0, label: '' });
  const fileInputRef = useRef(null);
  const [searchParams] = useSearchParams();
  const auditIdParam = searchParams.get('id');

  useEffect(() => {
    if (auditIdParam) {
      setStage('loading');
      setProgress({ percent: 100, label: 'Loading document audit...' });
      api.get(`/api/audits/${auditIdParam}`)
        .then((data) => {
          setAuditResult(data);
          setStage('analysis');
        })
        .catch((err) => {
          setError(err.message || 'Failed to load document audit');
          setStage('upload');
        });
    }
  }, [auditIdParam]);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = async (file) => {
    if (!file || isProcessing) return;

    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are currently supported.');
      return;
    }

    // Mirror the backend limit so oversized files fail instantly instead of
    // after a long upload.
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`This file is ${(file.size / (1024 * 1024)).toFixed(1)} MB, which exceeds the ${MAX_UPLOAD_MB} MB limit.`);
      return;
    }

    setError(null);
    setStage('loading');
    setIsProcessing(true);
    setProgress({ percent: 5, label: 'Uploading document...' });

    const formData = new FormData();
    formData.append('file', file);

    try {
      // The upload returns 202 as soon as the file is stored; the analysis runs
      // in the background and we poll for it.
      const accepted = await api.post('/api/audits/upload', formData, { isFormData: true });

      if (accepted.reused) {
        setProgress({ percent: 100, label: 'Already analysed — loading result' });
      }

      const result = await pollUntilReady(accepted.id);
      setAuditResult(result);
      setStage('analysis');
    } catch (err) {
      setError(err.message);
      setStage('upload');
    } finally {
      setIsProcessing(false);
    }
  };

  /**
   * Poll the status endpoint until processing finishes, then fetch the analysis.
   *
   * Summarising a long contract can take minutes, so there is a generous ceiling
   * on attempts; the interval stays short enough that the progress bar moves.
   */
  const pollUntilReady = async (auditId) => {
    const intervalMs = 1500;
    const maxAttempts = 400; // ~10 minutes

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const status = await api.get(`/api/audits/${auditId}/status`);
      setProgress({
        percent: status.progress ?? 0,
        label: status.progress_label || 'Processing...',
      });

      if (status.is_complete) {
        return api.get(`/api/audits/${auditId}`);
      }
      if (status.is_failed) {
        throw new Error(status.error || 'Processing failed for this document.');
      }
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
    throw new Error('Processing is taking longer than expected. Check the Library shortly.');
  };

  const slideVariants = {
    enter: { opacity: 0, x: 20 },
    center: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -20 },
  };

  const filteredClauses = auditResult?.clauses?.filter(c => {
    if (clauseFilter === 'All') return true;
    return c.category === clauseFilter;
  }) || [];

  return (
    <div className="p-8 max-w-[1400px] mx-auto w-full h-full flex flex-col">
      <AnimatePresence mode="wait">
        {stage === 'upload' ? (
          <motion.div
            key="upload"
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.3 }}
            className="flex-1 flex flex-col max-w-4xl mx-auto w-full pt-8"
          >
            <div className="mb-6">
              <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">Audit New Document</h1>
              <p className="text-slate-500 dark:text-slate-400 mt-2 font-medium">Upload a legal contract or agreement to extract clauses, key entities, and citation-backed insights.</p>
            </div>

            {error && (
              <div className="mb-4 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 border border-red-200 dark:border-red-900/50 flex items-center gap-3">
                <AlertTriangle size={20} />
                <span className="font-medium text-sm">{error}</span>
              </div>
            )}

            {/* Drop Zone */}
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept=".pdf" 
              onChange={handleFileChange} 
            />
            <div 
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              className="border-2 border-dashed border-slate-300 dark:border-slate-700 hover:border-primary-blue dark:hover:border-primary-blue bg-white dark:bg-slate-900 rounded-2xl p-12 flex flex-col items-center justify-center text-center cursor-pointer transition-all shadow-sm group"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="w-16 h-16 bg-blue-50 dark:bg-blue-900/30 text-primary-blue dark:text-blue-400 rounded-full flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
                <UploadCloud size={32} />
              </div>
              <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-1">Click to upload or drag & drop PDF</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">Supported formats: PDF (Contracts, NDAs, Service Agreements, MSAs)</p>
              <button 
                className="bg-primary-blue text-white px-6 py-2.5 rounded-lg font-semibold hover:bg-blue-700 transition-colors shadow-md shadow-blue-500/20"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                Select Legal File
              </button>
            </div>
          </motion.div>
        ) : stage === 'loading' ? (
          <motion.div
            key="loading"
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.3 }}
            className="flex-1 flex flex-col items-center justify-center max-w-4xl mx-auto w-full pt-8"
          >
             <Loader2 size={44} className="text-primary-blue dark:text-blue-400 animate-spin mb-4" />
             <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-2">Analyzing Legal Contract</h2>
             <p className="text-slate-500 dark:text-slate-400 text-center max-w-md mb-6">
               {progress.label || 'Extracting text, detecting sections, and summarizing...'}
             </p>

             {/* Staged progress, driven by the backend pipeline */}
             <div className="w-full max-w-md">
               <div className="h-2 w-full bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                 <motion.div
                   className="h-full bg-primary-blue rounded-full"
                   initial={{ width: 0 }}
                   animate={{ width: `${Math.max(progress.percent, 3)}%` }}
                   transition={{ duration: 0.4, ease: 'easeOut' }}
                 />
               </div>
               <div className="flex justify-between mt-2 text-[11px] font-semibold text-slate-400 dark:text-slate-500">
                 <span>{progress.percent}%</span>
                 <span>Large contracts may take a few minutes</span>
               </div>
             </div>
          </motion.div>
        ) : (
          <motion.div
            key="analysis"
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.4 }}
            className="flex-1 flex gap-6 h-full pb-4"
          >
            {/* Left Side: PDF Viewer (55%) */}
            <div className="w-[55%] bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-sm flex flex-col overflow-hidden min-h-0 transition-colors duration-200">
              <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between bg-slate-50/50 dark:bg-slate-900/50 shrink-0">
                <div className="flex items-center gap-2 truncate pr-2">
                  <FileText size={18} className="text-primary-blue dark:text-blue-400 shrink-0" />
                  <span className="font-semibold text-sm text-slate-800 dark:text-slate-200 truncate">{auditResult?.file_name}</span>
                  {auditResult?.contract_type && (
                    <span className="hidden lg:inline-flex shrink-0 items-center gap-1 text-[10px] font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded border border-slate-200 dark:border-slate-700">
                      <Tag size={10} />{auditResult.contract_type}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => setIsChatOpen(true)}
                    className="flex items-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-300 bg-indigo-50 dark:bg-indigo-500/10 px-3 py-1.5 rounded-lg border border-indigo-100 dark:border-indigo-500/20 transition-colors"
                  >
                    <MessageSquare size={13} />
                    Chat Assistant
                  </button>
                  <a
                    href={api.url(`/api/audits/file/${auditResult?.id}?token=${localStorage.getItem('token')}`)}
                    download={auditResult?.file_name}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center gap-1.5 text-xs font-semibold text-primary-blue dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 bg-blue-50 dark:bg-blue-500/10 px-3 py-1.5 rounded-lg border border-blue-100 dark:border-blue-500/20 transition-colors"
                  >
                    <Download size={13} />
                    Download
                  </a>
                  <div className="h-6 w-px bg-slate-200 dark:bg-slate-800 mx-1"></div>
                  <button 
                    className="text-xs font-semibold text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 px-2 py-1.5 transition-colors"
                    onClick={() => setStage('upload')}
                  >
                    New Audit
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-hidden bg-slate-100 dark:bg-slate-950">
                <iframe
                  src={api.url(`/api/audits/file/${auditResult?.id}?token=${localStorage.getItem('token')}`)}
                  width="100%"
                  height="100%"
                  title="PDF Preview"
                  className="w-full h-full border-0"
                  style={{ minHeight: '600px' }}
                >
                  <div className="flex flex-col items-center justify-center h-full text-slate-400 dark:text-slate-500 gap-3 p-6">
                    <FileText size={40} className="opacity-50" />
                    <p className="text-sm font-medium">Your browser cannot display the PDF inline.</p>
                    <a
                      href={api.url(`/api/audits/file/${auditResult?.id}?token=${localStorage.getItem('token')}`)}
                      className="text-primary-blue underline text-sm font-semibold"
                      target="_blank" rel="noreferrer"
                    >Open in external tab</a>
                  </div>
                </iframe>
              </div>
            </div>

            {/* Right Side: Tabbed Analysis Dashboard (45%) */}
            <div className="w-[45%] bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-sm flex flex-col overflow-hidden min-h-0 transition-colors duration-200">
              
              {/* Tab Navigation Header */}
              <div className="flex items-center border-b border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50 overflow-x-auto shrink-0 scrollbar-none">
                <button 
                  onClick={() => setActiveTab('summary')}
                  className={`flex items-center gap-1.5 px-4 py-3.5 text-xs font-bold border-b-[2px] transition-all whitespace-nowrap ${activeTab === 'summary' ? 'border-primary-blue text-primary-blue dark:text-blue-400 bg-white dark:bg-slate-900' : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'}`}
                >
                  <FileText size={14} />
                  Summary
                </button>

                <button 
                  onClick={() => setActiveTab('clauses')}
                  className={`flex items-center gap-1.5 px-4 py-3.5 text-xs font-bold border-b-[2px] transition-all whitespace-nowrap ${activeTab === 'clauses' ? 'border-primary-blue text-primary-blue dark:text-blue-400 bg-white dark:bg-slate-900' : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'}`}
                >
                  <Layers size={14} />
                  Clauses
                  {auditResult?.clauses?.length > 0 && <span className="ml-1 inline-flex items-center justify-center bg-blue-100 dark:bg-blue-500/20 text-blue-700 dark:text-blue-300 rounded-full px-1.5 py-0.2 text-[10px]">{auditResult.clauses.length}</span>}
                </button>

                <button 
                  onClick={() => setActiveTab('entities')}
                  className={`flex items-center gap-1.5 px-4 py-3.5 text-xs font-bold border-b-[2px] transition-all whitespace-nowrap ${activeTab === 'entities' ? 'border-primary-blue text-primary-blue dark:text-blue-400 bg-white dark:bg-slate-900' : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'}`}
                >
                  <Building2 size={14} />
                  Key Entities
                </button>

                <button 
                  onClick={() => setActiveTab('risks')}
                  className={`flex items-center gap-1.5 px-4 py-3.5 text-xs font-bold border-b-[2px] transition-all whitespace-nowrap ${activeTab === 'risks' ? 'border-primary-blue text-primary-blue dark:text-blue-400 bg-white dark:bg-slate-900' : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'}`}
                >
                  <ShieldAlert size={14} />
                  Risks {auditResult?.risks?.length > 0 && <span className="ml-1 inline-flex items-center justify-center bg-red-100 dark:bg-red-500/20 text-red-600 dark:text-red-400 rounded-full px-1.5 py-0.2 text-[10px]">{auditResult.risks.length}</span>}
                </button>

                <button 
                  onClick={() => setActiveTab('suggestions')}
                  className={`flex items-center gap-1.5 px-4 py-3.5 text-xs font-bold border-b-[2px] transition-all whitespace-nowrap ${activeTab === 'suggestions' ? 'border-primary-blue text-primary-blue dark:text-blue-400 bg-white dark:bg-slate-900' : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'}`}
                >
                  <Lightbulb size={14} />
                  Suggestions
                </button>
              </div>

              {/* Tab Body */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                
                {/* ── 1. SUMMARIZATION TAB ──────────────────────── */}
                {activeTab === 'summary' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                    {/* Degraded summarisation must be visible, not silent */}
                    {auditResult?.degraded && (
                      <div className="flex items-start gap-2.5 p-3 rounded-lg bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20">
                        <AlertTriangle size={15} className="text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
                        <p className="text-[11px] text-amber-800 dark:text-amber-200 leading-relaxed font-medium">
                          {auditResult.degraded_reason || 'The summarisation model was unavailable; summaries below are extracted directly from the document text.'}
                        </p>
                      </div>
                    )}
                    {/* Executive Summary Card */}
                    <div className="bg-gradient-to-br from-blue-50/50 to-indigo-50/30 dark:from-slate-800/60 dark:to-slate-800/30 border border-blue-100 dark:border-slate-800 rounded-xl p-5 shadow-xs">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-xs font-extrabold text-blue-900 dark:text-blue-300 uppercase tracking-wider flex items-center gap-1.5">
                          <FileText size={14} /> Executive Summary
                        </h4>
                        <SourceCitation citation={{ page_number: 1, section: "Overview", text: auditResult?.executive_summary }} />
                      </div>
                      <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed font-medium">
                        {auditResult?.executive_summary}
                      </p>
                    </div>

                    {/* Risk Score Gauge */}
                    {auditResult?.risk_score !== undefined && (
                      <div className="bg-slate-50 dark:bg-slate-800/40 border border-slate-200/80 dark:border-slate-800 rounded-xl p-4 flex items-center justify-between">
                        <div>
                          <h4 className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Overall Risk Score</h4>
                          <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">Calculated based on detected liability, notice & penalty clauses</p>
                        </div>
                        {(() => {
                           const score = auditResult.risk_score;
                           const colors = getSeverityColor(score);
                           const radius = 22;
                           const circumference = 2 * Math.PI * radius;
                           const strokeDashoffset = circumference - (score / 100) * circumference;
                           return (
                             <div className="flex items-center gap-3">
                               <div className="relative w-14 h-14 flex items-center justify-center">
                                 <svg className="w-full h-full transform -rotate-90">
                                   <circle cx="28" cy="28" r={radius} stroke="currentColor" strokeWidth="4" fill="transparent" className="text-slate-200 dark:text-slate-700" />
                                   <circle cx="28" cy="28" r={radius} stroke={colors.bar} strokeWidth="4" fill="transparent" strokeDasharray={circumference} strokeDashoffset={strokeDashoffset} strokeLinecap="round" className="transition-all duration-1000 ease-out" />
                                 </svg>
                                 <span className={`absolute text-sm font-extrabold ${colors.text}`}>{score}</span>
                               </div>
                               <div className={`px-2.5 py-1 rounded-md border ${colors.bg} ${colors.border} ${colors.text} font-bold text-xs`}>
                                 {score > 80 ? 'Safe' : score >= 50 ? 'Moderate Risk' : 'Critical Risk'}
                               </div>
                             </div>
                           );
                        })()}
                      </div>
                    )}

                    {/* Key Takeaways */}
                    {auditResult?.key_takeaways?.length > 0 && (
                      <div>
                        <h4 className="text-[11px] font-extrabold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-3">Key Takeaways</h4>
                        <div className="space-y-2">
                          {auditResult.key_takeaways.map((takeaway, idx) => (
                            <div key={idx} className="flex items-start gap-2.5 bg-white dark:bg-slate-800/80 p-3 rounded-lg border border-slate-200/70 dark:border-slate-800 text-xs text-slate-700 dark:text-slate-300">
                              <CheckCircle2 size={15} className="text-emerald-500 mt-0.5 shrink-0" />
                              <span className="font-medium leading-relaxed">{takeaway}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Engine provenance — which model produced this summary */}
                    {(auditResult?.summary_provider || auditResult?.model_name) && (
                      <div className="text-[10px] text-slate-400 dark:text-slate-500 border-t border-slate-100 dark:border-slate-800 pt-3 flex flex-wrap gap-x-3 gap-y-1">
                        <span>Engine: <span className="font-semibold">{auditResult.summary_provider || 'local'}</span></span>
                        {auditResult.model_name && <span>Model: {auditResult.model_name}</span>}
                        {auditResult.page_count ? <span>{auditResult.page_count} pages</span> : null}
                        {auditResult.chunk_count ? <span>{auditResult.chunk_count} chunks</span> : null}
                        {auditResult.token_count ? <span>{auditResult.token_count} tokens</span> : null}
                        {auditResult.processing_time ? <span>{auditResult.processing_time}s</span> : null}
                        {auditResult.structure_detected === false && (
                          <span className="text-amber-500">no section structure detected — page-based segmentation</span>
                        )}
                      </div>
                    )}

                    {/* Section-by-Section Breakdown */}
                    {auditResult?.section_summaries?.length > 0 && (
                      <div className="mt-6 pt-6 border-t border-slate-200/60 dark:border-slate-800">
                        <div className="flex items-center justify-between mb-4">
                          <h4 className="text-xs font-extrabold text-slate-700 dark:text-slate-300 uppercase tracking-wider flex items-center gap-2">
                            <Layers size={14} className="text-primary-blue dark:text-blue-400" />
                            Section Breakdown
                            <span className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 lowercase normal-case bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded-full">
                              {auditResult.section_summaries.length} sections
                            </span>
                          </h4>
                        </div>
                        <div className="space-y-4">
                          {auditResult.section_summaries.map((sec, idx) => {
                            const title = sec.section_title || sec.title || (sec.section_number ? `Section ${sec.section_number}` : `Section ${idx + 1}`);
                            const pageNum = sec.page_start || sec.page_number || (idx + 1);
                            const summaryText = sec.summary || '';
                            const keyPoints = sec.key_points || [];

                            return (
                              <div 
                                key={idx} 
                                className="bg-slate-50/90 dark:bg-slate-800/40 border border-slate-200/80 dark:border-slate-800 rounded-xl p-4 shadow-2xs hover:border-blue-200 dark:hover:border-blue-900/50 transition-all"
                              >
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                                  <h5 className="font-bold text-sm text-slate-800 dark:text-white flex items-center gap-2">
                                    <span className="w-5 h-5 rounded bg-blue-50 dark:bg-blue-900/30 text-primary-blue dark:text-blue-400 text-[10px] font-extrabold flex items-center justify-center shrink-0">
                                      {idx + 1}
                                    </span>
                                    <span>{title}</span>
                                  </h5>
                                  <SourceCitation 
                                    citation={{ 
                                      page_number: pageNum, 
                                      section: title, 
                                      text: summaryText || (keyPoints[0] || '') 
                                    }} 
                                  />
                                </div>

                                {summaryText && (
                                  <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed mb-3">
                                    {summaryText}
                                  </p>
                                )}

                                {keyPoints.length > 0 && (
                                  <div className="mt-2.5 pt-2.5 border-t border-slate-200/60 dark:border-slate-800/80">
                                    <div className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                                      <CheckCircle2 size={12} className="text-emerald-500" />
                                      <span>Key Provisions</span>
                                    </div>
                                    <ul className="space-y-1.5">
                                      {keyPoints.map((pt, pIdx) => (
                                        <li key={pIdx} className="text-xs text-slate-600 dark:text-slate-400 flex items-start gap-2">
                                          <span className="w-1.5 h-1.5 bg-blue-500 dark:bg-blue-400 rounded-full shrink-0 mt-1.5"></span>
                                          <span className="leading-relaxed">{pt}</span>
                                        </li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}

                {/* ── 2. CLAUSE DETECTION & HIGHLIGHTING TAB ─────── */}
                {activeTab === 'clauses' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">
                    {/* Category Filter Chips */}
                    <div className="flex flex-wrap gap-1.5 border-b border-slate-100 dark:border-slate-800 pb-3">
                      {CLAUSE_FILTERS.map(cat => (
                        <button
                          key={cat}
                          onClick={() => setClauseFilter(cat)}
                          className={`px-3 py-1 rounded-full text-xs font-semibold transition-all ${
                            clauseFilter === cat 
                              ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-xs' 
                              : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                          }`}
                        >
                          {cat}
                        </button>
                      ))}
                    </div>

                    {/* Clause Cards List */}
                    <div className="space-y-4">
                      {filteredClauses.length > 0 ? (
                        filteredClauses.map((clause) => (
                          <div 
                            key={clause.id || clause.title}
                            className="bg-white dark:bg-slate-800/90 border border-slate-200/90 dark:border-slate-800 rounded-xl p-4 shadow-xs hover:border-blue-300 dark:hover:border-blue-500/40 transition-all space-y-3"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-2">
                                <span className={`px-2.5 py-0.5 rounded-md text-[10px] font-extrabold uppercase border ${getClauseCategoryColor(clause.category)}`}>
                                  {clause.category}
                                </span>
                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getRiskBadgeStyle(clause.risk_level)}`}>
                                  {clause.risk_level} Risk
                                </span>
                                {clause.confidence > 0 && (
                                  <span className="text-[10px] font-semibold text-slate-400 dark:text-slate-500" title="Detector confidence">
                                    {Math.round(clause.confidence * 100)}%
                                  </span>
                                )}
                              </div>
                              <SourceCitation citation={{ page_number: clause.page_number, section: clause.section, text: clause.snippet }} />
                            </div>

                            <h5 className="font-extrabold text-xs text-slate-900 dark:text-white">
                              {clause.title}
                            </h5>

                            {/* Original Text Snippet */}
                            <div className="bg-slate-50 dark:bg-slate-900/60 p-3 rounded-lg border border-slate-100 dark:border-slate-800 text-[11px] font-mono text-slate-700 dark:text-slate-300 leading-relaxed italic relative pl-4 border-l-2 border-l-blue-500">
                              "<HighlightedText text={clause.snippet} terms={clause.matched_terms} />"
                            </div>

                            {/* Plain English Explanation */}
                            {clause.explanation && (
                              <div className="flex items-start gap-2 text-xs text-slate-600 dark:text-slate-400 bg-blue-50/40 dark:bg-blue-500/5 p-2.5 rounded-lg border border-blue-100/60 dark:border-blue-500/10">
                                <Info size={14} className="text-blue-500 shrink-0 mt-0.5" />
                                <span className="leading-normal font-medium">{clause.explanation}</span>
                              </div>
                            )}

                            {/* Negotiation Recommendation */}
                            {clause.recommendation && (
                              <div className="flex items-start gap-2 text-xs text-slate-600 dark:text-slate-400 bg-amber-50/50 dark:bg-amber-500/5 p-2.5 rounded-lg border border-amber-100/70 dark:border-amber-500/10">
                                <Lightbulb size={14} className="text-amber-500 shrink-0 mt-0.5" />
                                <span className="leading-normal font-medium">{clause.recommendation}</span>
                              </div>
                            )}
                          </div>
                        ))
                      ) : (
                        <EmptyState
                          icon={Layers}
                          message={
                            auditResult?.clauses?.length
                              ? 'No clauses matched the selected category.'
                              : 'No classifiable clauses were detected in this document.'
                          }
                        />
                      )}
                    </div>
                  </motion.div>
                )}

                {/* ── 3. KEY ENTITY EXTRACTION TAB ──────────────── */}
                {activeTab === 'entities' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                    {/* Grid 1: Parties & Organizations */}
                    <div className="bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
                      <h4 className="text-xs font-extrabold text-slate-900 dark:text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                        <Building2 size={16} className="text-blue-500" /> Parties & Organizations
                      </h4>
                      {auditResult?.entities?.parties?.length ? (
                      <div className="grid grid-cols-1 gap-3">
                        {auditResult.entities.parties.map((party, pIdx) => (
                          <div key={pIdx} className="bg-slate-50 dark:bg-slate-900/50 p-3.5 rounded-lg border border-slate-200/70 dark:border-slate-800">
                            <div className="flex justify-between items-start mb-1">
                              <span className="font-extrabold text-xs text-slate-900 dark:text-white">{party.name}</span>
                              <span className="text-[10px] font-semibold bg-blue-100 dark:bg-blue-500/20 text-blue-700 dark:text-blue-300 px-2 py-0.5 rounded">
                                {party.role}
                              </span>
                            </div>
                            {party.address && <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">📍 {party.address}</p>}
                            {party.signatory && <p className="text-[11px] text-slate-600 dark:text-slate-300 mt-1 font-medium">✍️ Signatory: {party.signatory}</p>}
                          </div>
                        ))}
                      </div>
                      ) : (
                        <EmptyState icon={Building2} message="No contracting parties could be identified in this document." />
                      )}
                    </div>

                    {/* Grid 2: Key Dates & Timeline */}
                    <div className="bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
                      <h4 className="text-xs font-extrabold text-slate-900 dark:text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                        <Calendar size={16} className="text-emerald-500" /> Key Dates & Milestones
                      </h4>
                      {auditResult?.entities?.dates?.length ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {auditResult.entities.dates.map((dt, dIdx) => (
                          <div key={dIdx} className="bg-slate-50 dark:bg-slate-900/50 p-3 rounded-lg border border-slate-200/70 dark:border-slate-800 space-y-1">
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase">{dt.label}</span>
                              <SourceCitation citation={{ page_number: dt.page_number, section: dt.section, text: dt.note }} />
                            </div>
                            <p className="font-bold text-xs text-slate-900 dark:text-white">{dt.value}</p>
                            {dt.note && <p className="text-[10px] text-slate-500 dark:text-slate-400">{dt.note}</p>}
                          </div>
                        ))}
                      </div>
                      ) : (
                        <EmptyState icon={Calendar} message="No key dates or timelines were stated in this document." />
                      )}
                    </div>

                    {/* Grid 3: Financial & Monetary Values */}
                    <div className="bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
                      <h4 className="text-xs font-extrabold text-slate-900 dark:text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                        <DollarSign size={16} className="text-amber-500" /> Financial & Monetary Values
                      </h4>
                      {auditResult?.entities?.financials?.length ? (
                      <div className="space-y-3">
                        {auditResult.entities.financials.map((fin, fIdx) => (
                          <div key={fIdx} className="bg-slate-50 dark:bg-slate-900/50 p-3 rounded-lg border border-slate-200/70 dark:border-slate-800 flex items-center justify-between">
                            <div>
                              <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase block">{fin.label}</span>
                              <span className="font-extrabold text-sm text-slate-900 dark:text-white">{fin.amount}</span>
                              {fin.detail && <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">{fin.detail}</p>}
                            </div>
                            <SourceCitation citation={{ page_number: fin.page_number, section: fin.section, text: fin.detail }} />
                          </div>
                        ))}
                      </div>
                      ) : (
                        <EmptyState icon={DollarSign} message="No monetary values or financial terms were found in this document." />
                      )}
                    </div>

                    {/* Grid 4: Governing Law & Jurisdiction */}
                    <div className="bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
                      <h4 className="text-xs font-extrabold text-slate-900 dark:text-white uppercase tracking-wider mb-3 flex items-center gap-2">
                        <Scale size={16} className="text-indigo-500" /> Governing Law & Venue
                      </h4>
                      <div className="bg-indigo-50/50 dark:bg-indigo-500/10 p-3.5 rounded-lg border border-indigo-100 dark:border-indigo-500/20 space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-500 dark:text-slate-400 font-medium">Governing Law:</span>
                          <span className="font-bold text-slate-900 dark:text-white">{auditResult?.entities?.jurisdiction?.governing_law || 'Not specified'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500 dark:text-slate-400 font-medium">Court Venue:</span>
                          <span className="font-bold text-slate-900 dark:text-white">{auditResult?.entities?.jurisdiction?.venue || 'Not specified'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500 dark:text-slate-400 font-medium">Dispute Resolution:</span>
                          <span className="font-bold text-slate-900 dark:text-white">{auditResult?.entities?.jurisdiction?.dispute_resolution || 'Not specified'}</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* ── 4. RISKS TAB ──────────────────────────────── */}
                {activeTab === 'risks' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                    {auditResult?.risks?.length > 0 ? (
                      auditResult.risks.map((risk, idx) => (
                        <div key={idx} className="border border-red-200 dark:border-red-900/50 bg-red-50/30 dark:bg-red-500/5 rounded-xl p-5 relative overflow-hidden">
                          <div className="absolute top-0 left-0 w-1 h-full bg-red-500"></div>
                          <div className="flex justify-between items-start mb-2">
                            <div className="flex gap-2 items-center">
                              <ShieldAlert size={18} className="text-red-500 shrink-0" />
                              <h4 className="font-bold text-slate-800 dark:text-red-100 text-xs">{risk.title}</h4>
                              {risk.severity && (
                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border shrink-0 ${getRiskBadgeStyle(risk.severity)}`}>
                                  {risk.severity}
                                </span>
                              )}
                            </div>
                            <SourceCitation citation={{ page_number: risk.page_number, section: risk.section || "Risk Area", text: risk.snippet || risk.description }} />
                          </div>
                          <p className="text-[12px] text-slate-600 dark:text-slate-400 leading-relaxed pl-6">
                            {risk.description}
                          </p>
                          {risk.snippet && (
                            <div className="mt-2 ml-6 bg-white/70 dark:bg-slate-900/50 border border-red-100 dark:border-red-900/40 rounded-lg p-2.5 text-[11px] font-mono italic text-slate-700 dark:text-slate-300 leading-relaxed">
                              "{risk.snippet}"
                            </div>
                          )}
                          {risk.recommendation && (
                            <div className="mt-2 ml-6 flex items-start gap-2 text-[11px] text-slate-600 dark:text-slate-400">
                              <Lightbulb size={13} className="text-amber-500 shrink-0 mt-0.5" />
                              <span className="font-medium">{risk.recommendation}</span>
                            </div>
                          )}
                        </div>
                      ))
                    ) : (
                      <EmptyState icon={ShieldAlert} message="No material risks were identified in this document." />
                    )}
                  </motion.div>
                )}

                {/* ── 5. SUGGESTIONS TAB ────────────────────────── */}
                {activeTab === 'suggestions' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                    {auditResult?.suggestions?.length > 0 ? (
                      auditResult.suggestions.map((suggestion, idx) => (
                        <div key={idx} className="border border-blue-200 dark:border-blue-900/50 bg-blue-50/30 dark:bg-blue-500/5 rounded-xl p-4 relative overflow-hidden flex gap-3">
                          <div className="absolute top-0 left-0 w-1 h-full bg-primary-blue"></div>
                          <Lightbulb size={18} className="text-primary-blue mt-0.5 shrink-0" />
                          <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed font-medium">
                            {typeof suggestion === 'string' ? suggestion : suggestion.text}
                          </p>
                        </div>
                      ))
                    ) : (
                      <EmptyState icon={Lightbulb} message="No remediation suggestions were generated for this document." />
                    )}
                  </motion.div>
                )}

              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Chat Drawer */}
      <ChatPanel
        auditId={auditResult?.id}
        fileName={auditResult?.file_name}
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
      />
    </div>
  );
};

export default AuditNew;
