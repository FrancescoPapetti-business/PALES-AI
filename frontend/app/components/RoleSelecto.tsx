type RoleSelectorProps = {
  role: string | null;
  setRole: (r: string | null) => void;
};

export default function RoleSelector({ role, setRole }: RoleSelectorProps) {
  const roles = [
    { id: "operatore", label: "Operatore" },
    { id: "delegato", label: "Delegato" },
    { id: "veterinario_aziendale", label: "Vet. aziendale" },
    { id: "veterinario_ufficiale", label: "Vet. ufficiale" },
  ];

  return (
    <section className="mb-6">
      <p className="text-sm font-semibold text-slate-700 mb-2">
        Seleziona il tuo ruolo:
      </p>

      <div className="flex flex-wrap gap-2">
        {roles.map((r) => (
          <button
            key={r.id}
            onClick={() => setRole(r.id)}
            className={[
              "px-4 py-2 rounded-full border text-xs md:text-sm transition-all",
              "focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-1",
              role === r.id
                ? "bg-blue-600 text-white border-blue-700 shadow-sm"
                : "bg-white text-slate-700 border-slate-300 hover:bg-slate-50",
            ].join(" ")}
          >
            {r.label}
          </button>
        ))}
      </div>

      <p className="mt-2 text-xs text-slate-500">
        Se non selezioni alcun ruolo, PALES-AI risponde in modalità generica.
      </p>
    </section>
  );
}
