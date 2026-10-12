import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Bot, Sparkles, X } from 'lucide-react';
import ChatPanel from './ChatPanel';

/**
 * FloatingChatWidget — Floating bottom-right chatbot widget with a friendly popup prompt.
 */
const FloatingChatWidget = ({ activeAuditId, activeFileName }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [showTooltip, setShowTooltip] = useState(true);

  return (
    <>
      {/* Floating Action Trigger Button */}
      <div className="fixed bottom-6 right-6 z-30 flex flex-col items-end gap-2">
        {/* Friendly greeting tooltip bubble */}
        <AnimatePresence>
          {showTooltip && !isOpen && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 5, scale: 0.9 }}
              className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-3 rounded-2xl shadow-xl max-w-xs flex items-center gap-3 relative border-l-4 border-l-blue-600"
            >
              <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-500/20 flex items-center justify-center shrink-0">
                <Bot size={18} className="text-blue-600 dark:text-blue-400" />
              </div>
              <div className="text-xs text-slate-700 dark:text-slate-200">
                <p className="font-bold text-slate-900 dark:text-white">👋 Hey there!</p>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">Got questions about your contract? Ask away.</p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowTooltip(false);
                }}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 p-1"
              >
                <X size={12} />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Floating Bubble Button */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => {
            setIsOpen(true);
            setShowTooltip(false);
          }}
          className="w-14 h-14 rounded-full bg-gradient-to-r from-blue-600 via-indigo-600 to-indigo-700 text-white shadow-xl shadow-blue-500/30 flex items-center justify-center relative group border-2 border-white/20 cursor-pointer"
        >
          <Bot size={26} />
          
          {/* Active online pulse ring */}
          <span className="absolute top-0 right-0 w-3.5 h-3.5 bg-emerald-400 rounded-full border-2 border-slate-900 animate-pulse"></span>
          
          <Sparkles size={14} className="absolute -top-1 -left-1 text-amber-300 animate-bounce" />
        </motion.button>
      </div>

      {/* RAG Chatbot Drawer */}
      <ChatPanel
        auditId={activeAuditId}
        fileName={activeFileName}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
      />
    </>
  );
};

export default FloatingChatWidget;
