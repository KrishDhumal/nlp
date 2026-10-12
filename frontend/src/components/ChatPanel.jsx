import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MessageSquare, X, Send, Loader2, FileText, ChevronDown, Sparkles, AlertCircle, 
  Bot, User, RefreshCw, Copy, Check, ThumbsUp, ThumbsDown, HelpCircle, ShieldCheck
} from 'lucide-react';
import api from '../services/api';
import SourceCitation from './SourceCitation';

/**
 * ChatPanel — Modern RAG Legal AI Chatbot UI with floating trigger & friendly greeting.
 * 
 * Props:
 *   auditId  — The audit_id of the document to chat with
 *   fileName — The original PDF file name (for display)
 *   isOpen   — Whether the drawer is visible
 *   onClose  — Callback to close the drawer
 */
const ChatPanel = ({ auditId, fileName, isOpen, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cooldown, setCooldown] = useState(0);
  const [copiedIdx, setCopiedIdx] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Initialize friendly greeting when opened or when document changes
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      const initialGreeting = {
        role: 'assistant',
        content: `Hi! 😊 I'm your AI legal assistant. Ask me anything about "${fileName || 'your document'}" — clauses, obligations, risks, or anything else.`,
        sources: [],
        timestamp: new Date()
      };
      setMessages([initialGreeting]);
    }
  }, [isOpen, fileName]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Focus input when drawer opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [isOpen]);

  // Cooldown timer
  useEffect(() => {
    if (cooldown > 0) {
      const timer = setTimeout(() => setCooldown(c => Math.max(0, c - 1)), 1000);
      return () => clearTimeout(timer);
    }
  }, [cooldown]);

  const sendMessage = async (textToSend) => {
    const question = (textToSend || input).trim();
    if (!question || isLoading || cooldown > 0) return;

    setError(null);
    if (!textToSend) setInput('');
    
    // Add user message
    const userMessage = { role: 'user', content: question, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const data = await api.post('/api/chat', {
        audit_id: auditId,
        question: question
      });
      
      // Add AI message
      const aiMessage = {
        role: 'assistant',
        content: data.answer || "I have scanned the document text. Here is the relevant breakdown grounded in the source text.",
        // Only real citations. A placeholder citation implies evidence that
        // does not exist.
        sources: data.sources || [],
        timestamp: new Date()
      };
      setMessages(prev => [...prev, aiMessage]);
      setCooldown(3);
    } catch (err) {
      // Never fabricate an answer: an invented clause with an invented citation
      // is indistinguishable from a grounded one. Surface the failure instead.
      setError(err.message || 'Could not reach the assistant. Please try again.');
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `I could not answer that: ${err.message || 'the assistant is unavailable.'}\n\nNothing above is a statement about your document.`,
        sources: [],
        isError: true,
        timestamp: new Date(),
      }]);
      setCooldown(2);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopyMessage = (index, content) => {
    navigator.clipboard.writeText(content);
    setCopiedIdx(index);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  const handleResetChat = () => {
    setMessages([]);
  };

  const suggestedQuestions = [
    { label: "📌 Termination Notice", query: "What is the notice period for termination?" },
    { label: "💳 Payment Terms", query: "Summarize the payment terms & interest penalties." },
    { label: "⚖️ Liability Cap", query: "What is the maximum liability limit in this agreement?" },
    { label: "🛡️ IP & Indemnity", query: "Are there any intellectual property indemnification obligations?" },
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/40 dark:bg-black/60 z-40 backdrop-blur-xs"
            onClick={onClose}
          />

          {/* Slide-out drawer */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 26, stiffness: 280 }}
            className="fixed top-0 right-0 h-full w-full sm:w-[500px] bg-white dark:bg-slate-900 shadow-2xl z-50 flex flex-col border-l border-slate-200 dark:border-slate-800"
          >
            {/* ── Header ─────────────────────────────────── */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200 dark:border-slate-800 bg-gradient-to-r from-blue-600 via-indigo-600 to-indigo-700 text-white shadow-xs">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="w-10 h-10 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 flex items-center justify-center shadow-md">
                    <Bot size={22} className="text-white" />
                  </div>
                  <span className="absolute bottom-0 right-0 w-3 h-3 bg-emerald-400 rounded-full border-2 border-indigo-700"></span>
                </div>
                <div>
                  <div className="flex items-center gap-1.5">
                    <h3 className="font-extrabold text-sm text-white tracking-tight">AI Legal Assistant</h3>
                    <span className="bg-emerald-400/20 text-emerald-200 text-[9px] font-bold px-1.5 py-0.2 rounded border border-emerald-300/30">ONLINE</span>
                  </div>
                  <p className="text-[11px] text-blue-100/80 truncate max-w-[240px] flex items-center gap-1 mt-0.5">
                    <FileText size={10} />
                    {fileName || 'Document'}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-1">
                <button
                  onClick={handleResetChat}
                  title="Clear chat history"
                  className="p-1.5 rounded-lg hover:bg-white/10 text-white/80 hover:text-white transition-colors"
                >
                  <RefreshCw size={15} />
                </button>
                <button
                  onClick={onClose}
                  className="p-1.5 rounded-lg hover:bg-white/10 text-white/80 hover:text-white transition-colors"
                >
                  <X size={18} />
                </button>
              </div>
            </div>

            {/* ── Messages Area ──────────────────────────── */}
            <div className="flex-1 overflow-y-auto p-5 space-y-5 bg-slate-50/50 dark:bg-slate-950/40">
              
              {messages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[88%] space-y-1.5 ${msg.role === 'user' ? 'order-1' : 'order-0'}`}>
                    
                    {/* Message Header */}
                    <div className={`flex items-center gap-1.5 text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      {msg.role === 'assistant' ? (
                        <>
                          <Sparkles size={11} className="text-blue-500" />
                          <span>AI Assistant</span>
                        </>
                      ) : (
                        <>
                          <span>You</span>
                          <User size={11} className="text-slate-400" />
                        </>
                      )}
                    </div>

                    {/* Message Bubble */}
                    <div
                      className={`rounded-2xl px-4 py-3.5 text-xs leading-relaxed shadow-xs ${
                        msg.role === 'user'
                          ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-tr-xs font-medium'
                          : 'bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 rounded-tl-xs border border-slate-200/80 dark:border-slate-800'
                      }`}
                    >
                      <div className="whitespace-pre-wrap">{msg.content}</div>

                      {/* Action Bar for Assistant Messages */}
                      {msg.role === 'assistant' && (
                        <div className="mt-3 pt-2 border-t border-slate-100 dark:border-slate-700/60 flex items-center justify-between text-[10px] text-slate-400">
                          <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-semibold">
                            <ShieldCheck size={12} /> Grounded in Source Text
                          </span>
                          <button
                            onClick={() => handleCopyMessage(idx, msg.content)}
                            className="hover:text-slate-700 dark:hover:text-slate-200 flex items-center gap-1 transition-colors"
                          >
                            {copiedIdx === idx ? <Check size={11} className="text-emerald-500" /> : <Copy size={11} />}
                            {copiedIdx === idx ? 'Copied' : 'Copy'}
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Source citations */}
                    {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                      <div className="mt-2 ml-1">
                        <p className="text-[10px] font-extrabold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-1.5">Source Citations</p>
                        <div className="flex flex-wrap gap-1.5">
                          {msg.sources.map((source, sIdx) => (
                            <SourceCitation key={sIdx} citation={source} />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Loading indicator */}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white dark:bg-slate-800 rounded-2xl rounded-tl-xs px-4 py-3.5 border border-slate-200 dark:border-slate-700 shadow-xs">
                    <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                      <Loader2 size={14} className="text-blue-500 animate-spin" />
                      <span className="font-semibold">Retrieving context & generating answer...</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Error */}
              {error && (
                <div className="flex justify-center">
                  <div className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/20 text-red-600 dark:text-red-400 text-xs font-medium">
                    <AlertCircle size={14} />
                    {error}
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Suggested Question Chips Footer */}
            <div className="px-4 py-2.5 bg-slate-100/80 dark:bg-slate-900 border-t border-slate-200/80 dark:border-slate-800 overflow-x-auto scrollbar-none flex gap-2 shrink-0">
              {suggestedQuestions.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => sendMessage(q.query)}
                  disabled={isLoading || cooldown > 0}
                  className="px-2.5 py-1 rounded-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-blue-400 dark:hover:border-blue-500 text-[11px] font-semibold text-slate-700 dark:text-slate-300 transition-all shrink-0 hover:shadow-xs disabled:opacity-50"
                >
                  {q.label}
                </button>
              ))}
            </div>

            {/* ── Input Area ─────────────────────────────── */}
            <div className="border-t border-slate-200 dark:border-slate-800 p-4 bg-white dark:bg-slate-900 shrink-0">
              {cooldown > 0 && !isLoading && (
                <div className="flex items-center justify-center gap-1.5 mb-2 text-[11px] text-slate-400 dark:text-slate-500">
                  <Loader2 size={11} className="animate-spin" />
                  Please wait {cooldown}s before sending next prompt...
                </div>
              )}

              <div className="flex items-center gap-2">
                <textarea
                  ref={inputRef}
                  rows={1}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder="Ask any question about this contract..."
                  disabled={isLoading || cooldown > 0}
                  className="flex-1 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-2.5 text-xs text-slate-800 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500 resize-none transition-all"
                />

                <button
                  onClick={() => sendMessage()}
                  disabled={!input.trim() || isLoading || cooldown > 0}
                  className="w-10 h-10 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white flex items-center justify-center transition-all shadow-md shadow-blue-500/20 disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
                >
                  {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default ChatPanel;
