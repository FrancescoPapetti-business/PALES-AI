export default function Sidebar() {
  return (
    <aside className="w-1/3 max-w-md bg-white border-r border-slate-200 p-6 flex flex-col shadow-sm">
      <header className="mb-6">
        <div className="inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-800">
          <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
          PALES-AI · Online
        </div>
        <h1 className="mt-4 text-2xl font-bold text-blue-900">
          PALES-AI
        </h1>
        <p className="text-sm text-slate-600">
          Assistente istituzionale ClassyFarm
        </p>
      </header>

      {/* KPI BOX */}
      <section className="mb-4">
        <h2 className="text-sm font-semibold text-slate-700 mb-2">
          Monitoraggio KPI
        </h2>
        <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg h-48 overflow-y-auto text-sm text-slate-600">
          <p className="text-xs text-slate-500 mb-1">
            (Integrazione KPI in arrivo)
          </p>
          <ul className="mt-2 space-y-1 text-xs">
            <li>• Accuracy risposte</li>
            <li>• Hallucination rate</li>
            <li>• Tempo medio di risposta</li>
            <li>• Interazioni concluse con successo</li>
          </ul>
        </div>
      </section>

      {/* INFO TRASPARENZA */}
      <section className="mt-4 text-xs text-slate-600 space-y-1">
        <p>• Le risposte si basano su documenti ufficiali ClassyFarm.</p>
        <p>
          • Architettura <span className="font-semibold">RAG</span> con FAISS.
        </p>
        <p>• Layer di sicurezza e moderazione attivo.</p>
        <p>• Nessun dato personale viene trattato.</p>
      </section>
    </aside>
  );
}
