import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Clock, Calendar, CheckCircle, AlertCircle, AlertTriangle, FileText, ArrowRight } from 'lucide-react';
import api from '../services/api';
import { formatAuditCardDate } from '../utils/dateUtils';

const RecentAudits = () => {
  const [audits, setAudits] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchAudits = async () => {
      try {
        const result = await api.get('/api/audits/recent');
        const formatted = (result || []).slice(0, 3).map(a => {
           let level = 'Low';
           const hasHigh = a.risks?.some(r => r.severity === 'High');
           const hasMed = a.risks?.some(r => r.severity === 'Medium' || r.severity === 'Moderate');
           if (hasHigh) level = 'Critical';
           else if (hasMed) level = 'Moderate';
           
           const dateInfo = formatAuditCardDate(a.created_at);

           return {
              id: a._id,
              title: a.file_name,
              ...dateInfo,
              score: a.risk_score !== undefined ? a.risk_score : 100,
              description: a.summary || 'No summary provided',
              tags: a.risks?.map(r => `#${r.title?.split(' ')[0] || 'Risk'}`).slice(0, 3) || []
           };
        });
        setAudits(formatted);
      } catch (error) {
        console.error('Failed to fetch recent audits:', error);
      }
    };
    fetchAudits();
  }, []);

  const getSeverityBadge = (score) => {
    if (score > 80) {
      return <span className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-semibold rounded-md border border-emerald-100 dark:border-emerald-500/20"><CheckCircle size={14}/> Safe</span>;
    }
    if (score >= 50) {
      return <span className="flex items-center gap-1.5 px-2.5 py-1 bg-amber-50 dark:bg-amber-500/10 text-amber-600 dark:text-amber-400 text-xs font-semibold rounded-md border border-amber-100 dark:border-amber-500/20"><AlertTriangle size={14}/> Moderate</span>;
    }
    return <span className="flex items-center gap-1.5 px-2.5 py-1 bg-red-50 dark:bg-red-500/10 text-red-600 dark:text-red-400 text-xs font-semibold rounded-md border border-red-100 dark:border-red-500/20"><AlertCircle size={14}/> Critical</span>;
  };

  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl shadow-sm border border-slate-200/60 dark:border-slate-800 overflow-hidden flex flex-col h-full transition-colors duration-200">
      <div className="px-6 py-5 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
        <h2 className="text-lg font-bold text-slate-800 dark:text-white tracking-tight">Recent Audits</h2>
        <button 
          onClick={() => navigate('/library')} 
          className="text-sm font-semibold text-primary-blue dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 inline-flex items-center gap-1 transition-colors cursor-pointer bg-transparent border-none p-0"
        >
          View All <ArrowRight size={14} />
        </button>
      </div>
      
      <div className="p-6 flex-1 flex flex-col gap-4">
        {audits.map((audit) => (
          <div 
            key={audit.id} 
            onClick={() => navigate(`/audit?id=${audit.id}`)}
            className="p-5 border border-slate-100 dark:border-slate-800 rounded-lg hover:border-blue-200 dark:hover:border-blue-800 hover:shadow-md transition-all bg-slate-50/50 dark:bg-slate-800/50 hover:bg-white dark:hover:bg-slate-800 group cursor-pointer"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 mb-3">
              <div className="flex items-center gap-2.5 min-w-0 pr-2">
                <div className="w-8 h-8 rounded-lg bg-blue-50 dark:bg-blue-900/30 text-primary-blue dark:text-blue-400 flex items-center justify-center shrink-0">
                  <FileText size={16} />
                </div>
                <h3 className="font-semibold text-slate-800 dark:text-white text-base group-hover:text-primary-blue dark:group-hover:text-blue-400 transition-colors truncate">
                  {audit.title}
                </h3>
              </div>

              {/* Exact Date, Exact Time & Relative Timing Badges */}
              <div className="flex items-center flex-wrap gap-2 text-xs font-medium shrink-0">
                <span 
                  className="inline-flex items-center gap-1.5 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-2.5 py-1 rounded-md border border-slate-200/60 dark:border-slate-700" 
                  title={`Analyzed on ${audit.fullDateTime}`}
                >
                  <Calendar size={13} className="text-primary-blue dark:text-blue-400" />
                  {audit.formattedDate}
                </span>

                <span 
                  className="inline-flex items-center gap-1.5 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-2.5 py-1 rounded-md border border-slate-200/60 dark:border-slate-700" 
                  title={`Time: ${audit.formattedTime}`}
                >
                  <Clock size={13} className="text-slate-400 dark:text-slate-500" />
                  {audit.formattedTime}
                </span>

                <span className="text-slate-500 dark:text-slate-400 text-xs font-medium ml-0.5">
                  • {audit.timeAgo}
                </span>
              </div>
            </div>
            
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed mb-4 line-clamp-2">
              {audit.description}
            </p>
            
            <div className="flex items-center justify-between border-t border-slate-100/60 dark:border-slate-700/60 pt-4 mt-auto">
              <div className="flex gap-2 flex-wrap">
                {audit.tags.map(tag => (
                  <span key={tag} className="text-xs font-medium bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 text-slate-500 dark:text-slate-300 px-2 py-1 rounded-md">
                    {tag}
                  </span>
                ))}
              </div>
              {getSeverityBadge(audit.score)}
            </div>
          </div>
        ))}
        {audits.length === 0 && (
          <div className="text-center py-8 text-slate-500">
            <p>No recent audits found.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default RecentAudits;
