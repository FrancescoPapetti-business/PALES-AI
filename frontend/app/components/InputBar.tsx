type Props = {
  input: string;
  setInput: (v: string) => void;
  sendMessage: () => void;
  isTyping: boolean;
};

export default function InputBar({
  input,
  setInput,
  sendMessage,
  isTyping,
}: Props) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <section className="mt-4 flex gap-2">
      <input
        className="flex-grow border border-slate-300 rounded-lg px-3 py-2 text-sm shadow-sm
                   focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 bg-white"
        placeholder="Scrivi la tua domanda su ClassyFarm..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <button
        onClick={sendMessage}
        className="px-5 py-2 rounded-lg bg-blue-600 text-white text-sm font-semibold
                   shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400
                   focus:ring-offset-1 disabled:bg-slate-300 disabled:cursor-not-allowed"
        disabled={isTyping}
      >
        Invia
      </button>
    </section>
  );
}
