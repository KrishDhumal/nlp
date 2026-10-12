import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  FileText, Search, Filter, MessageSquare, Trash2, Loader2, 
  ExternalLink, CheckCircle, AlertTriangle, AlertCircle, 
  Calendar, Clock, Plus, Tag, ArrowUpRight, FolderOpen 
} from 'lucide-react';
import ChatPanel from '../components/ChatPanel';
import api from '../services/api';
import { formatAuditCardDate } from '../utils/dateUtils';

const Library = () => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterSeverity, setFilterSeverity] = useState('All');
  const [chatDoc, setChatDoc] = useState(null); // { id, name }
  const [deletingId, setDeletingId] = useState(null);
  const navigate = useNavigate();

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const result = await api.get('/api/library');
      setDocuments(result || []);
    } catch (error) {
      console.error('Failed to fetch library documents:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleDelete = async (e, docId) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this document analysis from your library?')) return;
    setDeletingId(docId);
    try {
      await api.delete(`/api/library/${docId}`);
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
    } catch (error) {
      alert(error.message || 'Failed to delete document.');
    } finally {
      setDeletingId(null);
    }
  };

  const getSeverityBadge = (score, level) => {
    const s = score !== undefined && score !== null ? score : 100;
    if (s > 80 || level === 'Safe' || level === 'Low') {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 text-xs font-semibold rounded-md border border-emerald-200/80 dark:border-emerald-500/20">
          <CheckCircle size={13} />
          <span>Safe ({s}/100)</span>
        </span>
      );
    }
    if (s >= 50 || level === 'Moderate' || level === 'Medium') {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 text-xs font-semibold rounded-md border border-amber-200/80 dark:border-amber-500/20">
          <AlertTriangle size={13} />
          <span>Moderate ({s}/100)</span>
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-red-50 dark:bg-red-500/10 text-red-700 dark:text-red-400 text-xs font-semibold rounded-md border border-red-200/80 dark:border-red-500/20">
        <AlertCircle size={13} />
        <span>Critical ({s}/100)</span>
      </span>
    );
  };

  // Filter documents by search and risk severity
  const filteredDocs = documents.filter((doc) => {
    const query = searchQuery.toLowerCase().trim();
    const matchesSearch = !query || 
      doc.name?.toLowerCase().includes(query) ||
      doc.contract_type?.toLowerCase().includes(query) ||
      doc.type?.toLowerCase().includes(query) ||
      doc.summary?.toLowerCase().includes(query);

    if (!matchesSearch) return false;

    if (filterSeverity === 'All') return true;
    const score = doc.risk_score ?? 100;
    if (filterSeverity === 'Safe') return score > 80;
    if (filterSeverity === 'Moderate') return score >= 50 && score <= 80;
    if (filterSeverity === 'Critical') return score < 50;
    return true;
  });

  return (
    <div className="p-8 max-w-[1400px] mx-auto w-full flex-1">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
              Document Analysis Library
            </h1>
            <span className="bg-blue-100 dark:bg-blue-900/40 text-primary-blue dark:text-blue-400 text-xs font-bold px-2.5 py-1 rounded-full">
              {documents.length} Analyzed
            </span>
          </div>
          <p className="text-slate-500 dark:text-slate-400 mt-1.5 font-medium text-sm">
            All contract audits, extractions, and summaries processed for your account.
          </p>
        </div>

        <button
          onClick={() => navigate('/audit')}
          className="bg-primary-blue hover:bg-blue-700 text-white px-5 py-2.5 rounded-xl font-semibold shadow-sm hover:shadow transition-all flex items-center justify-center gap-2 cursor-pointer self-start sm:self-auto"
        >
          <Plus size={18} />
          New Document Analysis
        </button>
      </div>

      {/* Main Container */}
      <div className="bg-white dark:bg-slate-900 rounded-xl shadow-sm border border-slate-200/80 dark:border-slate-800 p-6 transition-colors duration-200">
        {/* Search & Filter Toolbar */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={17} />
            <input
              type="text"
              placeholder="Search by contract name, type, or keywords..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 rounded-lg text-sm text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-blue/20 focus:border-primary-blue transition-all"
            />
          </div>

          {/* Filter Pills */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 md:pb-0">
            {['All', 'Safe', 'Moderate', 'Critical'].map((level) => (
              <button
                key={level}
                onClick={() => setFilterSeverity(level)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  filterSeverity === level
                    ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-xs'
                    : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                }`}
              >
                {level === 'All' ? 'All Analyses' : `${level} Risk`}
              </button>
            ))}
          </div>
        </div>

        {/* Documents Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-400 dark:text-slate-500 text-xs font-bold uppercase tracking-wider">
                <th className="pb-3.5 px-4">Document / Analysis</th>
                <th className="pb-3.5 px-4">Contract Type</th>
                <th className="pb-3.5 px-4">Compliance Health</th>
                <th className="pb-3.5 px-4">Date & Timing</th>
                <th className="pb-3.5 px-4">Size</th>
                <th className="pb-3.5 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredDocs.map((doc) => {
                const dateInfo = formatAuditCardDate(doc.created_at || doc.date);

                return (
                  <tr
                    key={doc.id}
                    onClick={() => navigate(`/audit?id=${doc.id}`)}
                    className="border-b border-slate-100 dark:border-slate-800/70 hover:bg-slate-50/80 dark:hover:bg-slate-800/50 transition-colors group cursor-pointer"
                  >
                    {/* Document Title & Preview */}
                    <td className="py-4 px-4">
                      <div className="flex items-start gap-3">
                        <div className="w-9 h-9 rounded-lg bg-blue-50 dark:bg-blue-900/30 text-primary-blue dark:text-blue-400 flex items-center justify-center shrink-0 mt-0.5 group-hover:scale-105 transition-transform">
                          <FileText size={18} />
                        </div>
                        <div className="min-w-0 max-w-xs sm:max-w-md">
                          <span className="font-bold text-sm text-slate-800 dark:text-white group-hover:text-primary-blue dark:group-hover:text-blue-400 transition-colors block truncate">
                            {doc.name}
                          </span>
                          {doc.summary ? (
                            <p className="text-xs text-slate-500 dark:text-slate-400 line-clamp-1 mt-0.5">
                              {doc.summary}
                            </p>
                          ) : (
                            <span className="text-[11px] text-slate-400">Analysis completed</span>
                          )}
                        </div>
                      </div>
                    </td>

                    {/* Contract Type Badge */}
                    <td className="py-4 px-4 text-xs">
                      <span className="inline-flex items-center gap-1 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-2.5 py-1 rounded-md font-semibold border border-slate-200/60 dark:border-slate-700">
                        <Tag size={11} className="text-slate-400" />
                        {doc.contract_type || doc.type || 'Legal Agreement'}
                      </span>
                    </td>

                    {/* Risk Badge */}
                    <td className="py-4 px-4">
                      {getSeverityBadge(doc.risk_score, doc.risk_level)}
                    </td>

                    {/* Date & Proper Timing */}
                    <td className="py-4 px-4 text-xs">
                      <div className="flex flex-col gap-0.5">
                        <div className="flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300">
                          <Calendar size={12} className="text-primary-blue dark:text-blue-400" />
                          <span>{dateInfo.formattedDate}</span>
                        </div>
                        <div className="flex items-center gap-1.5 text-slate-400 dark:text-slate-500 text-[11px]">
                          <Clock size={11} />
                          <span>{dateInfo.formattedTime || dateInfo.timeAgo}</span>
                          {dateInfo.formattedTime && <span>• {dateInfo.timeAgo}</span>}
                        </div>
                      </div>
                    </td>

                    {/* Size */}
                    <td className="py-4 px-4 text-xs font-medium text-slate-500 dark:text-slate-400">
                      {doc.size}
                    </td>

                    {/* Actions */}
                    <td className="py-4 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                      <div className="inline-flex items-center gap-1.5">
                        {/* View Audit Button */}
                        <button
                          onClick={() => navigate(`/audit?id=${doc.id}`)}
                          className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold text-primary-blue dark:text-blue-400 bg-blue-50 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 hover:bg-blue-100 dark:hover:bg-blue-500/20 transition-colors cursor-pointer"
                          title="Open document analysis"
                        >
                          <ExternalLink size={12} />
                          <span>View Analysis</span>
                        </button>

                        {/* Chat Button */}
                        <button
                          onClick={() => setChatDoc({ id: doc.id, name: doc.name })}
                          className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 hover:bg-indigo-100 dark:hover:bg-indigo-500/20 transition-colors cursor-pointer"
                          title="Chat with document AI"
                        >
                          <MessageSquare size={12} />
                          <span>Chat</span>
                        </button>

                        {/* Delete Button */}
                        <button
                          onClick={(e) => handleDelete(e, doc.id)}
                          disabled={deletingId === doc.id}
                          className="inline-flex items-center justify-center p-1.5 rounded-lg text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-500/10 border border-red-100 dark:border-red-500/20 hover:bg-red-100 dark:hover:bg-red-500/20 transition-colors cursor-pointer disabled:opacity-50"
                          title="Delete analysis"
                        >
                          {deletingId === doc.id ? (
                            <Loader2 size={13} className="animate-spin" />
                          ) : (
                            <Trash2 size={13} />
                          )}
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Empty States */}
        {!loading && filteredDocs.length === 0 && documents.length > 0 && (
          <div className="text-center py-12 text-slate-500 dark:text-slate-400">
            <Search size={40} className="mx-auto text-slate-300 dark:text-slate-600 mb-3" />
            <p className="text-base font-semibold text-slate-700 dark:text-slate-300">No matching document analyses</p>
            <p className="text-xs text-slate-400 mt-1">Try refining your search keyword or clearing the risk filter.</p>
            <button
              onClick={() => { setSearchQuery(''); setFilterSeverity('All'); }}
              className="mt-3 text-xs font-semibold text-primary-blue hover:underline"
            >
              Reset filters
            </button>
          </div>
        )}

        {!loading && documents.length === 0 && (
          <div className="text-center py-16 text-slate-500 dark:text-slate-400">
            <FolderOpen size={48} className="mx-auto text-slate-300 dark:text-slate-600 mb-3" />
            <p className="text-lg font-bold text-slate-800 dark:text-slate-200">No document analyses yet</p>
            <p className="text-sm text-slate-500 dark:text-slate-400 max-w-sm mx-auto mt-1 mb-6">
              Upload contracts, agreements, or legal files to review full clause breakdowns and risk summaries.
            </p>
            <button
              onClick={() => navigate('/audit')}
              className="bg-primary-blue hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg text-sm font-semibold transition-colors"
            >
              Upload Your First Document
            </button>
          </div>
        )}

        {loading && (
          <div className="text-center py-16 text-slate-500 dark:text-slate-400 flex flex-col items-center gap-2">
            <Loader2 size={28} className="animate-spin text-primary-blue" />
            <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">Loading document analyses...</p>
          </div>
        )}
      </div>

      {/* Persistent Chat Drawer */}
      <ChatPanel
        auditId={chatDoc?.id}
        fileName={chatDoc?.name}
        isOpen={!!chatDoc}
        onClose={() => setChatDoc(null)}
      />
    </div>
  );
};

export default Library;
