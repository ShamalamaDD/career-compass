import { X } from 'lucide-react';
const TEAM = ['Mitheel Ramdaw', 'Ryan Chitate', 'Mikhaar Ramdaw', 'Lionel Muke'];

export default function TeamSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="ml-auto h-full w-full max-w-md glass p-6 overflow-auto">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Cheat Code — Team</h3>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/5">
            <X />
          </button>
        </div>
        <div className="mt-4 space-y-3">
          {TEAM.map(n => (
            <div key={n} className="flex items-center gap-3 p-3 rounded-xl bg-white/5 border border-white/10">
              <div className="h-10 w-10 rounded-full bg-gradient-to-tr from-cyan-500/30 to-emerald-500/30 flex items-center justify-center font-semibold">
                {n.split(' ').map(p => p[0]).join('')}
              </div>
              <div className="font-medium">{n}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
