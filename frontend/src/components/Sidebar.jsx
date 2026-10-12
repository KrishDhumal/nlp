import React from 'react';
import { LayoutDashboard, FileSearch, Library, Settings, LogOut, GitCompareArrows, Shield } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Sidebar = () => {
  const location = useLocation();
  const { logout } = useAuth();
  
  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'New Summary', path: '/audit', icon: FileSearch },
    { name: 'Compare', path: '/compare', icon: GitCompareArrows },
    { name: 'Library', path: '/library', icon: Library },
    { name: 'Settings', path: '/settings', icon: Settings },
  ];

  return (
    <aside className="w-64 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col justify-between h-screen fixed top-0 left-0 z-20 transition-colors duration-200">
      <div className="p-6">
        <Link to="/dashboard" className="flex items-center gap-2 mb-8 group cursor-pointer w-fit">
          <div className="w-8 h-8 bg-primary-blue rounded flex items-center justify-center group-hover:bg-blue-700 transition-colors">
            <span className="text-white font-bold text-lg">D</span>
          </div>
          <span className="text-xl font-bold text-slate-900 dark:text-white tracking-tight group-hover:text-primary-blue dark:group-hover:text-blue-400 transition-colors">DocSummarizer</span>
        </Link>
        
        <nav className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = location.pathname === item.path || (item.path !== '/dashboard' && location.pathname.startsWith(item.path));
            
            return (
              <Link
                key={item.name}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg font-medium transition-colors ${
                  active
                    ? 'bg-blue-50 dark:bg-blue-900/30 text-primary-blue dark:text-blue-400'
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                <Icon size={20} className={active ? 'text-primary-blue dark:text-blue-400' : 'text-slate-400 dark:text-slate-500'} />
                {item.name}
              </Link>
            );
          })}
        </nav>
      </div>
      
      <div className="p-4 border-t border-slate-100 dark:border-slate-800 flex flex-col gap-3">
        <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-xl p-4 text-white shadow-md">
          <h4 className="font-semibold text-sm mb-1">Premium Plan</h4>
          <p className="text-xs text-blue-100 mb-3 opacity-90 leading-tight">Upgrade to unlock unlimited multi-page summaries.</p>
          <button className="w-full bg-white text-blue-700 text-xs font-semibold py-2 rounded-lg hover:bg-slate-50 transition-colors shadow-sm">
            Upgrade Now
          </button>
        </div>
        
        <button 
          onClick={logout}
          className="flex items-center justify-center gap-2 w-full py-2.5 text-sm font-semibold text-slate-600 dark:text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-500/10 rounded-lg transition-colors border border-transparent hover:border-red-100 dark:hover:border-red-500/20"
        >
          <LogOut size={16} />
          Log Out
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
