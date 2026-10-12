import React, { useState, useEffect, useRef } from 'react';
import { Search, Bell, Mail, X, AlertTriangle, CheckCircle, ShieldAlert } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';
import { formatTimeAgo } from '../utils/dateUtils';


const Header = () => {
  const { user } = useAuth();
  const name = user?.name || 'Patrick';
  const role = user?.role || 'Researcher';
  const avatarUrl = `https://ui-avatars.com/api/?name=${encodeURIComponent(name)}&background=eff6ff&color=2563eb`;

  const [showNotifs, setShowNotifs] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [notifLoading, setNotifLoading] = useState(false);
  const notifRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handler = (e) => {
      if (notifRef.current && !notifRef.current.contains(e.target)) {
        setShowNotifs(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const fetchNotifications = async () => {
    setNotifLoading(true);
    try {
      const data = await api.get('/api/notifications');
      setNotifications(data);
    } catch (err) {
      console.error('Failed to fetch notifications:', err);
    } finally {
      setNotifLoading(false);
    }
  };

  const handleBellClick = () => {
    const next = !showNotifs;
    setShowNotifs(next);
    if (next) fetchNotifications();
  };

  const getTimeAgo = (isoString) => {
    return formatTimeAgo(isoString);
  };

  const getNotifIcon = (type) => {
    if (type === 'critical') return <ShieldAlert size={16} className="text-red-500" />;
    if (type === 'warning') return <AlertTriangle size={16} className="text-amber-500" />;
    return <CheckCircle size={16} className="text-emerald-500" />;
  };

  return (
    <header className="h-20 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between px-8 sticky top-0 z-10 w-full transition-colors duration-200">
      <div className="flex-1 max-w-xl">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
          <input
            type="text"
            placeholder="Search documents, entities, or labels..."
            className="w-full pl-10 pr-4 py-2 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-sm dark:text-white dark:placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-blue/20 focus:border-primary-blue transition-all"
          />
        </div>
      </div>
      
      <div className="flex items-center gap-6 ml-4">
        <div className="flex items-center gap-3 text-slate-500 dark:text-slate-400">
          <button 
            className="hover:text-slate-800 dark:hover:text-slate-200 transition-colors relative group"
            title="Messages — Coming soon"
          >
            <Mail size={20} />
            <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full border border-white dark:border-slate-900"></span>
            {/* Tooltip */}
            <span className="absolute -bottom-8 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-[10px] font-medium px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
              Coming soon
            </span>
          </button>
          
          {/* Notification Bell with Dropdown */}
          <div className="relative" ref={notifRef}>
            <button 
              onClick={handleBellClick}
              className={`hover:text-slate-800 dark:hover:text-slate-200 transition-colors relative ${showNotifs ? 'text-primary-blue' : ''}`}
            >
              <Bell size={20} />
              {notifications.some(n => n.type === 'critical') && (
                <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full border border-white dark:border-slate-900 animate-pulse"></span>
              )}
              {!notifications.some(n => n.type === 'critical') && (
                <span className="absolute top-0 right-0 w-2 h-2 bg-primary-blue rounded-full border border-white dark:border-slate-900"></span>
              )}
            </button>

            {/* Dropdown Panel */}
            {showNotifs && (
              <div className="absolute right-0 top-10 w-80 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-2xl shadow-slate-200/50 dark:shadow-slate-900/50 z-50 overflow-hidden animate-in fade-in slide-in-from-top-2">
                <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-800/50">
                  <h4 className="font-bold text-sm text-slate-800 dark:text-white">Notifications</h4>
                  <button onClick={() => setShowNotifs(false)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200">
                    <X size={14} />
                  </button>
                </div>
                <div className="max-h-72 overflow-y-auto">
                  {notifLoading ? (
                    <div className="p-6 text-center text-sm text-slate-400">Loading...</div>
                  ) : notifications.length === 0 ? (
                    <div className="p-6 text-center text-sm text-slate-400">No notifications yet. Audit a document to get started.</div>
                  ) : (
                    notifications.map((n) => (
                      <div key={n.id} className="flex items-start gap-3 px-4 py-3 border-b border-slate-50 dark:border-slate-800/50 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors last:border-0">
                        <div className="mt-0.5 shrink-0">{getNotifIcon(n.type)}</div>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium text-slate-700 dark:text-slate-200 leading-snug">{n.message}</p>
                          <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1 font-medium">{getTimeAgo(n.created_at)}</p>
                        </div>
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded shrink-0 ${
                          n.type === 'critical' ? 'bg-red-50 dark:bg-red-500/10 text-red-500' :
                          n.type === 'warning' ? 'bg-amber-50 dark:bg-amber-500/10 text-amber-500' :
                          'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-500'
                        }`}>
                          {n.risk_score}%
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="h-8 w-px bg-slate-200 dark:bg-slate-700"></div>
        
        <div className="flex items-center gap-3 cursor-pointer group">
          <img
            src={avatarUrl}
            alt="User profile"
            className="w-9 h-9 rounded-full object-cover border-2 border-slate-100 dark:border-slate-800 group-hover:border-primary-blue/30 transition-colors"
          />
          <div className="hidden md:block">
            <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">{name}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 font-medium">{role}</p>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
