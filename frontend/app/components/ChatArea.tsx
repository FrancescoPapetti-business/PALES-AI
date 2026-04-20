import { ChatMsg } from "../lib/types";
import MessageBubble from "./MessageBubble";

type Props = {
  messages: ChatMsg[];
  isTyping: boolean;
};

export default function ChatArea({ messages, isTyping }: Props) {
  return (
    <section className="flex-1 overflow-y-auto bg-white border border-slate-200 rounded-lg p-4 shadow-sm flex flex-col space-y-3">
      {/* Stato iniziale (nessun messaggio) */}
      {messages.length === 0 && (
        <div className="text-sm text-slate-500 bg-slate-50 border border-dashed border-slate-200 rounded-lg p-4">
          <p className="font-semibold text-slate-700 mb-1">
            Benvenuto in PALES-AI.
          </p>
          <p>
            Seleziona il tuo ruolo (se pertinente) e fai una domanda, ad esempio:
          </p>
          <ul className="mt-2 list-disc list-inside">
            <li>“Come faccio ad accedere a ClassyFarm?”</li>
            <li>“Come funziona la delega per l’operatore?”</li>
            <li>“Dove trovo le checklist di biosicurezza?”</li>
          </ul>
        </div>
      )}

      {/* Lista messaggi */}
      {messages.map((m, i) => (
        <MessageBubble key={i} msg={m} />
      ))}

      {/* Indicatore "sta scrivendo" */}
      {isTyping && (
        <div className="flex items-center gap-2 self-start mt-1">
          <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
          <div className="px-3 py-2 bg-slate-100 rounded-2xl text-xs text-slate-600">
            PALES-AI sta scrivendo…
          </div>
        </div>
      )}
    </section>
  );
}
