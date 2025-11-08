import { UploadCloud } from 'lucide-react';
import { useRef } from 'react';

export default function Dropzone({ onFiles }: { onFiles: (files: File[]) => void }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <div
      onDragOver={e => e.preventDefault()}
      onDrop={e => {
        e.preventDefault();
        onFiles(Array.from(e.dataTransfer.files));
      }}
      className="ai-card border-dashed border-2 border-white/15 hover:border-white/30 transition-colors"
    >
      <div className="text-center">
        <UploadCloud className="h-8 w-8 mx-auto text-cyan-300" />
        <p className="mt-2">Drag & drop your CV PDFs</p>
        <button
          onClick={() => input.current?.click()}
          className="mt-3 px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500/20 to-emerald-500/20 border border-white/10"
        >
          Browse files
        </button>
      </div>
      <input
        ref={input}
        type="file"
        accept="application/pdf"
        multiple
        hidden
        onChange={e => e.target.files && onFiles(Array.from(e.target.files))}
      />
    </div>
  );
}
