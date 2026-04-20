"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { ChatMsg, HistoryEntry } from "../lib/types";
import { sendChatMessage } from "../lib/api";

const QUICK_ACTIONS = [
  "Come accedo a ClassyFarm?",
  "Come mi registro a ClassyFarm",
  "Dove trovo le checklist?",
];

const WELCOME_MSG: ChatMsg = {
  sender: "assistant",
  text: "Ciao! Sono PALES-AI, l'Assistente virtuale di ClassyFarm 👋 Come posso aiutarti oggi?",
};

function Bubble({ msg, animate }: { msg: ChatMsg; animate?: boolean }) {
  const isUser = msg.sender === "user";
  const isSystem = msg.sender === "system";

  return (
    <div
      className={[
        "flex items-end gap-2 mb-3",
        animate ? "animate-msg-in" : "",
        isUser ? "flex-row-reverse" : "flex-row",
      ].join(" ")}
    >
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-cf-blue flex items-center justify-center text-white text-xs flex-shrink-0 mb-0.5">
          CF
        </div>
      )}
      <div
        className={[
          "max-w-[78%] px-3 py-2 rounded-2xl text-sm leading-relaxed shadow-sm",
          "break-words overflow-hidden",
          isUser
            ? "bg-cf-blue text-white rounded-br-sm"
            : isSystem
            ? "bg-amber-100 text-amber-900 border border-amber-200 rounded-bl-sm"
            : "bg-cf-blue-light text-blue-900 rounded-bl-sm",
        ].join(" ")}
        style={{ wordBreak: "break-word", overflowWrap: "anywhere" }}
      >
        {isUser ? (
          <span>{msg.text}</span>
        ) : (
          <ReactMarkdown
            components={{
              p: ({ children }) => (
                <p className="mb-1 last:mb-0">{children}</p>
              ),
              strong: ({ children }) => (
                <strong className="font-semibold">{children}</strong>
              ),
              ol: ({ children }) => (
                <ol className="list-decimal list-outside ml-4 my-1 space-y-0.5">
                  {children}
                </ol>
              ),
              ul: ({ children }) => (
                <ul className="list-disc list-outside ml-4 my-1 space-y-0.5">
                  {children}
                </ul>
              ),
              li: ({ children }) => (
                <li className="leading-snug">{children}</li>
              ),
              // ✅ Link cliccabile, si apre in nuova tab
              a: ({ href, children }) => (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline text-cf-blue hover:text-cf-blue-dark transition-colors"
                >
                  {children}
                </a>
              ),
            }}
          >
            {msg.text}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <div className="flex items-end gap-2 mb-3">
      <div className="w-7 h-7 rounded-full bg-cf-blue flex items-center justify-center text-white text-xs flex-shrink-0">
        CF
      </div>
      <div className="bg-cf-blue-light px-4 py-3 rounded-2xl rounded-bl-sm flex gap-1 items-center shadow-sm">
        <span className="w-2 h-2 rounded-full bg-cf-blue animate-typing-dot typing-dot-1 inline-block" />
        <span className="w-2 h-2 rounded-full bg-cf-blue animate-typing-dot typing-dot-2 inline-block" />
        <span className="w-2 h-2 rounded-full bg-cf-blue animate-typing-dot typing-dot-3 inline-block" />
      </div>
    </div>
  );
}

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMsg[]>([WELCOME_MSG]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [newMsgIndexes, setNewMsgIndexes] = useState<Set<number>>(new Set([0]));
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isTyping, isOpen]);

  const addMessage = (msg: ChatMsg) => {
    setMessages((prev) => {
      const newIndex = prev.length;
      setNewMsgIndexes((s) => new Set(s).add(newIndex));
      return [...prev, msg];
    });
  };

  const resetChat = () => {
    setMessages([WELCOME_MSG]);
    setNewMsgIndexes(new Set([0]));
    setInput("");
    setIsTyping(false);
    setSessionId(undefined);
  };

  const buildChatHistory = (currentMessages: ChatMsg[]): HistoryEntry[] => {
    return currentMessages
      .slice(1)
      .filter((m) => m.sender !== "system")
      .map((m) => ({
        role: m.sender as "user" | "assistant",
        content: m.text,
      }))
      .slice(-6);
  };

  const sendMessage = async (text?: string) => {
    const userText = (text ?? input).trim();
    if (!userText) return;

    setInput("");
    const historySnapshot = buildChatHistory(messages);
    addMessage({ sender: "user", text: userText });
    setIsTyping(true);

    try {
      const data = await sendChatMessage({
        message: userText,
        chat_history: historySnapshot,
        session_id: sessionId,
      });

      // Salva session_id dal server
      if (data.session_id) {
        setSessionId(data.session_id);
      }

      const answerText =
        data.answer ?? data.response ?? "Nessuna risposta ricevuta.";
      addMessage({ sender: "assistant", text: answerText });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Errore di connessione al server. Riprova.";
      addMessage({
        sender: "system",
        text: errorMsg,
      });
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {isOpen && (
        <div className="fixed bottom-24 right-5 z-50 w-[380px] h-[600px] flex flex-col bg-white rounded-2xl shadow-2xl animate-chat-open overflow-hidden border border-slate-200">

          {/* HEADER */}
          <header className="bg-cf-blue px-4 py-3 flex items-center gap-3 flex-shrink-0">
            <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
              CF
            </div>
          <div className="flex-1 min-w-0">
              <p className="text-white font-semibold text-sm leading-tight truncate">
                PALES-AI
              </p>
              <p className="text-blue-200 text-xs leading-tight">
                Assistente virtuale ClassyFarm · Online
              </p>
            </div>

            {/* Reset */}
            <button
              onClick={resetChat}
              title="Nuova conversazione"
              className="text-white/70 hover:text-white transition-colors flex-shrink-0"
              aria-label="Nuova conversazione"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>

            {/* Chiudi */}
            <button
              onClick={() => setIsOpen(false)}
              className="text-white/70 hover:text-white transition-colors flex-shrink-0"
              aria-label="Chiudi chat"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </header>

          {/* MESSAGGI */}
          <div className="flex-1 overflow-y-auto px-4 py-4 bg-slate-50">
            {messages.map((msg, i) => (
              <Bubble key={i} msg={msg} animate={newMsgIndexes.has(i)} />
            ))}
            {isTyping && <TypingDots />}
            {messages.length === 1 && !isTyping && (
              <div className="flex flex-col gap-2 mt-3 ml-9">
                {QUICK_ACTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => sendMessage(q)}
                    className="text-left text-xs px-3 py-2 rounded-full border border-cf-blue-mid bg-cf-blue-light text-cf-blue hover:bg-cf-blue hover:text-white transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* INPUT */}
          <div className="flex-shrink-0 px-3 py-3 bg-white border-t border-slate-200 flex gap-2 items-center">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Scrivi un messaggio..."
              disabled={isTyping}
              className="flex-1 bg-slate-50 border border-slate-300 rounded-full px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cf-blue focus:border-cf-blue disabled:opacity-50"
            />
            <button
              onClick={() => sendMessage()}
              disabled={isTyping || !input.trim()}
              aria-label="Invia messaggio"
              className="w-9 h-9 rounded-full bg-cf-blue flex items-center justify-center hover:bg-cf-blue-dark transition-colors disabled:bg-slate-300 disabled:cursor-not-allowed flex-shrink-0"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* LAUNCHER */}
      <button
        onClick={() => setIsOpen((v) => !v)}
        aria-label={isOpen ? "Chiudi assistente" : "Apri assistente ClassyFarm"}
        className="fixed bottom-5 right-5 z-50 w-14 h-14 rounded-full bg-cf-blue hover:bg-cf-blue-dark shadow-lg hover:shadow-xl flex items-center justify-center transition-all duration-200 focus:outline-none focus:ring-4 focus:ring-cf-blue/40"
      >
        {isOpen ? (
          <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M21 12c0 4.418-4.03 8-9 8a9.77 9.77 0 01-4.38-1.02L3 21l1.56-4.38A8.14 8.14 0 013 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        )}
      </button>
    </>
  );
}