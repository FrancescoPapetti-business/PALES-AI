import { ChatMsg } from "../lib/types";

type Props = {
  msg: ChatMsg;
  isTyping?: boolean;
};

export default function MessageBubble({ msg }: Props) {
  const isUser = msg.sender === "user";
  const isAssistant = msg.sender === "assistant";
  const isSystem = msg.sender === "system";

  return (
    <div
      className={[
        "max-w-[80%] px-3 py-2 rounded-2xl text-sm shadow-sm",
        isUser && "self-end bg-blue-600 text-white rounded-br-sm",
        isAssistant && "self-start bg-slate-100 text-slate-900 rounded-bl-sm",
        isSystem &&
          "self-center bg-amber-100 text-amber-900 border border-amber-200",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {!isSystem && (
        <p className="text-[0.7rem] uppercase tracking-wide mb-1 opacity-70">
          {isUser ? "TU" : "PALES-AI"}
        </p>
      )}
      <p>{msg.text}</p>
    </div>
  );
}
