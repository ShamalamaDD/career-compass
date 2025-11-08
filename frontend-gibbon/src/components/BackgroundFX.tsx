export default function BackgroundFX() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10">
      {/* animated grid */}
      <div className="absolute inset-0 bg-ai-grid opacity-20" />
      {/* soft radial glows */}
      <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-[80vw] h-[80vw] max-w-[950px] bg-ai-radial blur-3xl opacity-30" />
      <div className="absolute -bottom-40 right-10 w-[50vw] h-[50vw] max-w-[700px] bg-ai-radial-2 blur-3xl opacity-25" />
    </div>
  );
}
