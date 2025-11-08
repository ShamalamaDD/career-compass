export default function Brand() {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/10 p-5 bg-white/5">
      <div className="absolute -inset-16 bg-ai-orbit opacity-40" />
      <div className="relative">
        <h2 className="text-xl font-semibold">Welcome</h2>
        <p className="text-sm text-white/60 mt-1">
          Your personal AI career guide. Upload your CVs and discover the best-fit roles.
        </p>
      </div>
    </div>
  );
}
