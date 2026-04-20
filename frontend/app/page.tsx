export default function Home() {
  return (
    <main className="min-h-screen bg-slate-50 flex flex-col items-center justify-center px-8">
      <div className="max-w-lg text-center">
        <div className="inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-800 mb-6">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          PALES-AI · Online
        </div>
        <h1 className="text-4xl font-bold text-blue-900 mb-4">
          Assistente ClassyFarm
        </h1>
        <p className="text-slate-600 text-base leading-relaxed">
          Benvenuto nel portale di supporto ClassyFarm. Usa il pulsante in basso
          a destra per avviare una conversazione con l&apos;assistente virtuale.
        </p>
      </div>
    </main>
  );
}