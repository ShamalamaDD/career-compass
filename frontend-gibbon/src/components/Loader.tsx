export default function Loader({ label = 'Thinking…' }: { label?: string }) {
  return (
    <div className="ai-card flex items-center gap-3">
      <div className="h-3 w-3 rounded-full bg-cyan-300 animate-ping" />
      <div className="h-3 w-3 rounded-full bg-emerald-300 animate-ping [animation-delay:150ms]" />
      <div className="h-3 w-3 rounded-full bg-cyan-300 animate-ping [animation-delay:300ms]" />
      <span className="ml-2 text-sm text-white/70">{label}</span>
    </div>
  );
}
