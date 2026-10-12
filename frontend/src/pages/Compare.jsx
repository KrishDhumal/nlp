import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  GitCompareArrows, Loader2, AlertTriangle, ArrowRight, FileText, 
  Filter, CheckCircle2, Zap, ArrowLeftRight, Check, Sparkles, Info
} from 'lucide-react';
import api from '../services/api';
import SourceCitation from '../components/SourceCitation';

/**
 * Splits a block of text into logical paragraphs/sentences for granular diffing.
 * Splits on: blank lines, sentences ending with period/question/exclamation,
 * or lines starting with a heading-like pattern (number, bullet, ALL-CAPS word).
 */
const splitIntoParagraphs = (text) => {
  if (!text) return [];
  // First split on double newlines (paragraph breaks)
  const blocks = text.split(/\n{2,}/);
  const sentences = [];
  for (const block of blocks) {
    // Then split on sentence boundaries (. ! ? followed by space + capital)
    const parts = block.split(/(?<=[.!?])\s+(?=[A-Z0-9"("])/);
    for (const p of parts) {
      const trimmed = p.trim();
      if (trimmed) sentences.push(trimmed);
    }
  }
  return sentences.length > 0 ? sentences : [text];
};

/**
 * Computes word-level diff between two strings.
 * Returns { oldParts, newParts } with tokens marked as 'same', 'del', or 'add'.
 */
const computeWordDiff = (oldStr, newStr) => {
  if (!oldStr && !newStr) return { oldParts: [], newParts: [] };
  if (!oldStr) return { oldParts: [], newParts: [{ type: 'add', text: newStr }] };
  if (!newStr) return { oldParts: [{ type: 'del', text: oldStr }], newParts: [] };
  if (oldStr.trim() === newStr.trim()) {
    return {
      oldParts: [{ type: 'same', text: oldStr }],
      newParts: [{ type: 'same', text: newStr }]
    };
  }

  // Tokenize into words, whitespace, and punctuation
  const tokenize = (s) => s.match(/[\w'-]+|[^\w\s]+|\s+/g) || [s];
  const aWords = tokenize(oldStr);
  const bWords = tokenize(newStr);

  const m = aWords.length;
  const n = bWords.length;

  // Safeguard for very large sections
  if (m * n > 40000) {
    return {
      oldParts: [{ type: 'del', text: oldStr }],
      newParts: [{ type: 'add', text: newStr }]
    };
  }

  // Compute Longest Common Subsequence (LCS)
  const dp = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (aWords[i] === bWords[j]) {
        dp[i + 1][j + 1] = dp[i][j] + 1;
      } else {
        dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }

  // Backtrack to assemble diff tokens
  let i = m, j = n;
  const oldResult = [];
  const newResult = [];

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && aWords[i - 1] === bWords[j - 1]) {
      oldResult.unshift({ type: 'same', text: aWords[i - 1] });
      newResult.unshift({ type: 'same', text: bWords[j - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      newResult.unshift({ type: 'add', text: bWords[j - 1] });
      j--;
    } else if (i > 0 && (j === 0 || dp[i][j - 1] < dp[i - 1][j])) {
      oldResult.unshift({ type: 'del', text: aWords[i - 1] });
      i--;
    }
  }

  // Group adjacent tokens of same type
  const collapse = (tokens) => {
    const collapsed = [];
    for (const t of tokens) {
      if (collapsed.length > 0 && collapsed[collapsed.length - 1].type === t.type) {
        collapsed[collapsed.length - 1].text += t.text;
      } else {
        collapsed.push({ ...t });
      }
    }
    return collapsed;
  };

  return {
    oldParts: collapse(oldResult),
    newParts: collapse(newResult)
  };
};

/**
 * Renders a single word-diff row for one sentence pair.
 */
const SentenceDiffRow = ({ oldSentence, newSentence, isOnlyRow }) => {
  const { oldParts, newParts } = computeWordDiff(oldSentence || '', newSentence || '');
  const hasChanges = oldParts.some(p => p.type === 'del') || newParts.some(p => p.type === 'add');
  // Skip rendering rows where both sides are identical (no changes)
  if (!hasChanges && !isOnlyRow) return null;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-2">
      {/* Doc A sentence */}
      <div className="bg-rose-50/40 dark:bg-rose-950/20 p-3 rounded-lg border border-rose-200/70 dark:border-rose-900/40">
        <div className="text-xs text-slate-800 dark:text-slate-200 leading-relaxed font-mono">
          {oldSentence === undefined ? (
            <span className="text-slate-400 italic">Not present</span>
          ) : oldParts.map((part, pIdx) => (
            part.type === 'del' ? (
              <span
                key={pIdx}
                className="bg-rose-200/90 dark:bg-rose-900/70 text-rose-950 dark:text-rose-100 px-1 py-0.5 rounded font-bold line-through decoration-rose-600 mx-0.5"
                title="Removed or modified in Version B"
              >
                {part.text}
              </span>
            ) : (
              <span key={pIdx}>{part.text}</span>
            )
          ))}
        </div>
      </div>
      {/* Doc B sentence */}
      <div className="bg-emerald-50/40 dark:bg-emerald-950/20 p-3 rounded-lg border border-emerald-200/70 dark:border-emerald-900/40">
        <div className="text-xs text-slate-800 dark:text-slate-200 leading-relaxed font-mono">
          {newSentence === undefined ? (
            <span className="text-slate-400 italic">Not present</span>
          ) : newParts.map((part, pIdx) => (
            part.type === 'add' ? (
              <span
                key={pIdx}
                className="bg-emerald-200/90 dark:bg-emerald-900/70 text-emerald-950 dark:text-emerald-100 px-1 py-0.5 rounded font-bold mx-0.5"
                title="Added or modified in Version B"
              >
                {part.text}
              </span>
            ) : (
              <span key={pIdx}>{part.text}</span>
            )
          ))}
        </div>
      </div>
    </div>
  );
};

/**
 * Side-by-side diff renderer with visual word highlights.
 * Bug 3 fix: splits text by paragraph/sentence so only changed lines show highlights,
 * instead of marking the entire section block as removed/added.
 */
const DiffViewer = ({ textA, textB, status }) => {
  if (status === 'matched') {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-50 dark:bg-slate-950/60 p-4 rounded-xl border border-slate-200/80 dark:border-slate-800 space-y-2">
          <span className="text-[10px] font-extrabold text-blue-600 dark:text-blue-400 uppercase tracking-wider block border-b border-slate-200/60 dark:border-slate-800 pb-1.5">
            Doc A (Original)
          </span>
          <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed font-mono">
            {textA || <span className="text-slate-400 italic">Empty section</span>}
          </p>
        </div>
        <div className="bg-slate-50 dark:bg-slate-950/60 p-4 rounded-xl border border-slate-200/80 dark:border-slate-800 space-y-2">
          <span className="text-[10px] font-extrabold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider block border-b border-slate-200/60 dark:border-slate-800 pb-1.5 flex items-center gap-1">
            <Check size={12} /> Doc B (Identical Match)
          </span>
          <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed font-mono">
            {textB || <span className="text-slate-400 italic">Empty section</span>}
          </p>
        </div>
      </div>
    );
  }

  if (status === 'missing_in_a') {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-50 dark:bg-slate-950/60 p-4 rounded-xl border border-slate-200/80 dark:border-slate-800 space-y-2 opacity-60">
          <span className="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider block border-b border-slate-200/60 dark:border-slate-800 pb-1.5">
            Doc A (Not Present)
          </span>
          <p className="text-xs text-slate-400 italic">Clause does not exist in Base Document A.</p>
        </div>
        <div className="bg-emerald-50/50 dark:bg-emerald-950/20 p-4 rounded-xl border border-emerald-200/80 dark:border-emerald-800/60 space-y-2">
          <span className="text-[10px] font-extrabold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider block border-b border-emerald-200/60 dark:border-emerald-800 pb-1.5">
            Doc B (New Clause Added)
          </span>
          <p className="text-xs text-emerald-800 dark:text-emerald-200 leading-relaxed font-mono font-medium">
            {textB}
          </p>
        </div>
      </div>
    );
  }

  if (status === 'missing_in_b') {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-rose-50/50 dark:bg-rose-950/20 p-4 rounded-xl border border-rose-200/80 dark:border-rose-800/60 space-y-2">
          <span className="text-[10px] font-extrabold text-rose-600 dark:text-rose-400 uppercase tracking-wider block border-b border-rose-200/60 dark:border-rose-800 pb-1.5">
            Doc A (Original - Deleted in B)
          </span>
          <p className="text-xs text-rose-800 dark:text-rose-200 leading-relaxed font-mono font-medium">
            {textA}
          </p>
        </div>
        <div className="bg-slate-50 dark:bg-slate-950/60 p-4 rounded-xl border border-slate-200/80 dark:border-slate-800 space-y-2 opacity-60">
          <span className="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider block border-b border-slate-200/60 dark:border-slate-800 pb-1.5">
            Doc B (Removed)
          </span>
          <p className="text-xs text-slate-400 italic">Clause was completely omitted in Revised Document B.</p>
        </div>
      </div>
    );
  }

  // Modified: Split into sentences, then highlight only changed ones side-by-side.
  const aSentences = splitIntoParagraphs(textA);
  const bSentences = splitIntoParagraphs(textB);
  const rowCount = Math.max(aSentences.length, bSentences.length);

  return (
    <div className="space-y-1">
      {/* Column headers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-1">
        <div className="flex items-center justify-between px-3 py-1 bg-rose-100/60 dark:bg-rose-950/40 rounded-lg border border-rose-200/60 dark:border-rose-900/40">
          <span className="text-[10px] font-extrabold text-rose-600 dark:text-rose-400 uppercase tracking-wider">Doc A (Original Text)</span>
          <span className="text-[10px] bg-rose-200/80 dark:bg-rose-900/80 text-rose-700 dark:text-rose-300 font-bold px-2 py-0.5 rounded">Removed / Modified</span>
        </div>
        <div className="flex items-center justify-between px-3 py-1 bg-emerald-100/60 dark:bg-emerald-950/40 rounded-lg border border-emerald-200/60 dark:border-emerald-900/40">
          <span className="text-[10px] font-extrabold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">Doc B (Revised Text)</span>
          <span className="text-[10px] bg-emerald-200/80 dark:bg-emerald-900/80 text-emerald-700 dark:text-emerald-300 font-bold px-2 py-0.5 rounded">New / Added</span>
        </div>
      </div>
      {Array.from({ length: rowCount }, (_, i) => (
        <SentenceDiffRow
          key={i}
          oldSentence={aSentences[i]}
          newSentence={bSentences[i]}
          isOnlyRow={rowCount === 1}
        />
      ))}
    </div>
  );
};

const Compare = () => {
  const [documents, setDocuments] = useState([]);
  const [docA, setDocA] = useState('');
  const [docB, setDocB] = useState('');
  const [filterStatus, setFilterStatus] = useState('all'); // 'all' | 'modified' | 'missing' | 'matched'
  const [loading, setLoading] = useState(false);
  const [docsLoading, setDocsLoading] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDocs = async () => {
      try {
        const docs = await api.get('/api/library');
        setDocuments(docs || []);
        if (docs && docs.length >= 2) {
          setDocA(docs[0].id);
          setDocB(docs[1].id);
        } else if (docs && docs.length === 1) {
          // Only one document — pre-fill slot A but leave B empty with a hint
          setDocA(docs[0].id);
          setDocB('');
        }
      } catch (err) { 
        console.error('Failed to load library documents for comparison:', err); 
      } finally { 
        setDocsLoading(false); 
      }
    };
    fetchDocs();
  }, []);

  const handleCompare = async () => {
    if (!docA || !docB) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await api.post('/api/compare', { audit_id_a: docA, audit_id_b: docB });
      const enriched = enrichCompareResult(data);
      setResult(enriched);
      // Auto-switch to 'modified' filter so changes are visible immediately
      const hasChanges = enriched?.clauses?.some(c => c.status !== 'matched');
      setFilterStatus(hasChanges ? 'modified' : 'all');
    } catch (err) { 
      setError(err.message || 'Comparison failed between selected documents.');
    } finally { 
      setLoading(false); 
    }
  };

  const enrichCompareResult = (data) => {
    if (!data) return null;
    const docAName = documents.find(d => d.id === docA)?.name || data.doc_a_name || 'Document A';
    const docBName = documents.find(d => d.id === docB)?.name || data.doc_b_name || (docA === docB ? `${docAName} (Copy)` : 'Document B');

    // Extract clauses from data.clauses or data.changes
    let rawClauses = data.clauses || [];
    if (!rawClauses.length && data.changes?.length) {
      const statusMap = {
        MODIFIED: 'modified',
        ADDED: 'missing_in_a',
        REMOVED: 'missing_in_b',
        UNCHANGED: 'matched'
      };
      rawClauses = data.changes.map(c => ({
        topic: c.section || c.clause_type || 'Clause',
        status: statusMap[c.change_type] || 'modified',
        change_type: c.change_type,
        doc_a_text: c.old_text || '',
        doc_b_text: c.new_text || '',
        doc_a_citation: c.old_text ? { page_number: c.page_old || 1, section: c.section, text: c.old_text.slice(0, 150) } : null,
        doc_b_citation: c.new_text ? { page_number: c.page_new || 1, section: c.section, text: c.new_text.slice(0, 150) } : null,
        risk_note: c.explanation || '',
        severity: c.severity || 'LOW',
        similarity: c.similarity ?? 1.0,
        term_changes: c.term_changes || []
      }));
    }

    return {
      doc_a_name: data.doc_a_name || docAName,
      doc_b_name: data.doc_b_name || docBName,
      difference_summary: data.difference_summary || data.summary || "Comparison completed successfully.",
      risk_shift_score: data.risk_shift_score ?? 0,
      clauses: rawClauses,
      missing_in_a: data.missing_in_a || [],
      missing_in_b: data.missing_in_b || []
    };
  };

  const getScoreColor = (score) => {
    if (score > 5) return 'text-emerald-600 dark:text-emerald-400';
    if (score < -5) return 'text-red-600 dark:text-red-400';
    return 'text-slate-700 dark:text-slate-300';
  };

  const getStatusBadge = (status) => {
    const map = {
      'matched': { bg: 'bg-emerald-50 dark:bg-emerald-500/10 border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400', label: 'Identical' },
      'modified': { bg: 'bg-amber-50 dark:bg-amber-500/10 border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400', label: 'Clause Modified' },
      'missing_in_a': { bg: 'bg-blue-50 dark:bg-blue-500/10 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-400', label: 'Added in Doc B' },
      'missing_in_b': { bg: 'bg-rose-50 dark:bg-rose-500/10 border-rose-200 dark:border-rose-800 text-rose-700 dark:text-rose-400', label: 'Removed in Doc B' },
    };
    const s = map[status] || map['matched'];
    return <span className={`px-2.5 py-0.5 rounded-md text-[11px] font-bold border ${s.bg}`}>{s.label}</span>;
  };

  const filteredClauses = result?.clauses?.filter(c => {
    if (filterStatus === 'all') return true;
    if (filterStatus === 'modified') return c.status === 'modified';
    if (filterStatus === 'missing') return c.status === 'missing_in_a' || c.status === 'missing_in_b';
    if (filterStatus === 'matched') return c.status === 'matched';
    return true;
  }) || [];

  return (
    <div className="p-8 max-w-[1400px] mx-auto w-full space-y-6">
      <div className="mb-6">
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">Contract Version Comparison</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-2 font-medium">
          Compare two document versions side-by-side with word-level change highlighting, term modifications, and legal risk shifts.
        </p>
      </div>

      {/* Document Selectors */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
        <div className="grid grid-cols-1 md:grid-cols-[1fr,auto,1fr] gap-6 items-end">
          <div>
            <label className="block text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider mb-2">
              Base Document (Version A)
            </label>
            <select
              value={docA}
              onChange={(e) => setDocA(e.target.value)}
              className="w-full px-4 py-2.5 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-xs font-medium dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-blue/20"
              disabled={docsLoading}
            >
              <option value="">Select base document...</option>
              {documents.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>

          <div className="flex items-center justify-center pb-1">
            <div className="w-10 h-10 rounded-full bg-blue-50 dark:bg-blue-500/10 flex items-center justify-center border border-blue-100 dark:border-blue-500/20">
              <ArrowLeftRight size={20} className="text-primary-blue dark:text-blue-400" />
            </div>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider mb-2">
              Revised Document (Version B)
            </label>
            <select
              value={docB}
              onChange={(e) => setDocB(e.target.value)}
              className="w-full px-4 py-2.5 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-xs font-medium dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-blue/20"
              disabled={docsLoading}
            >
              <option value="">Select revised document...</option>
              {documents.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>
        </div>

        {/* Same-document warning */}
        {docA && docB && docA === docB && (
          <div className="mt-4 p-3 rounded-lg bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20 text-amber-800 dark:text-amber-300 text-xs font-medium flex items-start gap-2">
            <AlertTriangle size={16} className="shrink-0 mt-0.5 text-amber-500" />
            <span>
              <strong>Same document selected on both sides.</strong> No differences will be found.
              To highlight clause changes, <a href="/audit" className="underline font-bold">upload the modified version</a> of this document, then select the two versions here.
            </span>
          </div>
        )}

        {/* Not enough docs hint */}
        {!docsLoading && documents.length < 2 && !(docA && docB && docA === docB) && (
          <div className="mt-4 p-3 rounded-lg bg-blue-50 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 text-blue-700 dark:text-blue-300 text-xs font-medium flex items-start gap-2">
            <Info size={16} className="shrink-0 mt-0.5 text-blue-500" />
            <span>
              You need <strong>at least 2 documents</strong> in your library to compare.{' '}
              <a href="/audit" className="underline font-bold">Upload a document</a> to get started.
            </span>
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-xs font-medium flex items-center gap-2">
            <AlertTriangle size={16} />
            {error}
          </div>
        )}

        <div className="mt-6 flex justify-center">
          <button
            onClick={handleCompare}
            disabled={loading || !docA || !docB || docA === docB}
            className="bg-primary-blue text-white px-8 py-3 rounded-xl font-semibold hover:bg-blue-700 transition-all disabled:opacity-50 flex items-center gap-2 shadow-lg shadow-blue-500/20 cursor-pointer"
          >
            {loading ? <><Loader2 size={18} className="animate-spin" /> Analyzing Differences...</> : <><GitCompareArrows size={18} /> Compare &amp; Highlight Changes</>}
          </button>
        </div>
      </div>

      {/* Results View */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-6"
          >
            {/* Top Stat Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-2 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-xs">
                <h3 className="text-xs font-extrabold text-slate-400 dark:text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                  <Sparkles size={14} className="text-primary-blue" />
                  Comparison Executive Summary
                </h3>
                <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed font-medium">
                  {result.difference_summary}
                </p>
                <div className="mt-4 flex items-center flex-wrap gap-3 text-xs font-semibold text-slate-500">
                  <span className="flex items-center gap-1.5 text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-500/10 px-2.5 py-1 rounded">
                    <FileText size={13} /> {result.doc_a_name}
                  </span>
                  <span>vs</span>
                  <span className="flex items-center gap-1.5 text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-500/10 px-2.5 py-1 rounded">
                    <FileText size={13} /> {result.doc_b_name}
                  </span>
                </div>
              </div>

              <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-xs flex flex-col items-center justify-center text-center">
                <h3 className="text-xs font-extrabold text-slate-400 dark:text-slate-500 uppercase tracking-widest mb-2">Net Risk Shift</h3>
                <div className={`text-4xl font-extrabold ${getScoreColor(result.risk_shift_score)} flex items-center`}>
                  {result.risk_shift_score > 0 ? `+${result.risk_shift_score}` : result.risk_shift_score}
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 font-medium">
                  {result.risk_shift_score < -10 ? 'Version B improves client protections' :
                   result.risk_shift_score > 10 ? 'Version B increases legal risk exposure' :
                   'Risk profile remains balanced or identical'}
                </p>
              </div>
            </div>

            {/* Filter Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-white dark:bg-slate-900 px-6 py-3.5 rounded-xl border border-slate-200 dark:border-slate-800">
              <span className="text-xs font-extrabold text-slate-700 dark:text-slate-300 flex items-center gap-2">
                <Filter size={14} className="text-primary-blue" />
                Clauses &amp; Sections ({filteredClauses.length})
              </span>
              <div className="flex flex-wrap gap-1.5">
                {[
                  { id: 'all', label: 'All', count: result.clauses.length },
                  { id: 'modified', label: 'Modified', count: result.clauses.filter(c => c.status === 'modified').length },
                  { id: 'missing', label: 'Added/Removed', count: result.clauses.filter(c => c.status === 'missing_in_a' || c.status === 'missing_in_b').length },
                  { id: 'matched', label: 'Identical', count: result.clauses.filter(c => c.status === 'matched').length },
                ].map(f => (
                  <button
                    key={f.id}
                    onClick={() => setFilterStatus(f.id)}
                    className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                      filterStatus === f.id
                        ? 'bg-primary-blue text-white shadow-xs'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                    }`}
                  >
                    {f.label}
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${filterStatus === f.id ? 'bg-white/20 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400'}`}>
                      {f.count}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Side-by-Side Split View with Highlights */}
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-xs overflow-hidden divide-y divide-slate-100 dark:divide-slate-800">
              {filteredClauses.map((clause, idx) => (
                <div key={idx} className="p-6 space-y-3.5">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div className="flex items-center gap-2.5">
                      <span className="w-5 h-5 rounded bg-blue-50 dark:bg-blue-900/30 text-primary-blue dark:text-blue-400 text-[10px] font-extrabold flex items-center justify-center shrink-0">
                        {idx + 1}
                      </span>
                      <h4 className="font-extrabold text-sm text-slate-900 dark:text-white">
                        {clause.topic}
                      </h4>
                      {getStatusBadge(clause.status)}
                    </div>

                    {/* Citations */}
                    <div className="flex items-center gap-2 self-start sm:self-auto">
                      {clause.doc_a_citation && (
                        <SourceCitation citation={clause.doc_a_citation} />
                      )}
                      {clause.doc_b_citation && clause.doc_b_citation.page_number !== clause.doc_a_citation?.page_number && (
                        <SourceCitation citation={clause.doc_b_citation} />
                      )}
                    </div>
                  </div>

                  {/* Commercial Term Changes Highlight Badge */}
                  {clause.term_changes && clause.term_changes.length > 0 && (
                    <div className="flex flex-wrap items-center gap-2 bg-blue-50/70 dark:bg-blue-950/30 p-2.5 rounded-lg border border-blue-200/70 dark:border-blue-900/40 text-xs">
                      <Zap size={14} className="text-primary-blue dark:text-blue-400 shrink-0" />
                      <span className="font-bold text-slate-700 dark:text-slate-300">Detected Term Changes:</span>
                      {clause.term_changes.map((tc, tcIdx) => (
                        <span key={tcIdx} className="inline-flex items-center gap-1.5 bg-white dark:bg-slate-800 px-2.5 py-0.5 rounded text-xs font-semibold text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700">
                          <span className="text-slate-500 font-normal">{tc.term}:</span>
                          {tc.old ? <span className="text-rose-600 dark:text-rose-400 line-through">{tc.old}</span> : null}
                          {tc.old && tc.new ? <ArrowRight size={11} className="text-slate-400" /> : null}
                          {tc.new ? <span className="text-emerald-600 dark:text-emerald-400 font-bold">{tc.new}</span> : null}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Visual Word-Level Diff Highlighter */}
                  <DiffViewer
                    textA={clause.doc_a_text}
                    textB={clause.doc_b_text}
                    status={clause.status}
                  />

                  {/* Risk Implication Note */}
                  {clause.risk_note && (
                    <div className="flex items-start gap-2 bg-amber-50/70 dark:bg-amber-500/10 p-3 rounded-lg border border-amber-200/70 dark:border-amber-500/20 text-xs text-amber-900 dark:text-amber-300 font-medium">
                      <AlertTriangle size={15} className="text-amber-500 shrink-0 mt-0.5" />
                      <span>{clause.risk_note}</span>
                    </div>
                  )}
                </div>
              ))}

              {filteredClauses.length === 0 && (
                <div className="p-12 text-center text-slate-500 dark:text-slate-400">
                  <CheckCircle2 size={36} className="mx-auto text-emerald-500 mb-2" />
                  <p className="text-sm font-semibold">No clauses match the selected filter.</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Compare;
