import { X, UsersRound } from "lucide-react";

type Props = {
  open: boolean;
  onClose: () => void;
};

const TEAM = [
  { name: "Mitheel Ramdaw" },
  { name: "Ryan Chitate" },
  { name: "Mikhaar Ramdaw" },
  { name: "Lionel Muke" },
];

export default function SideNav({ open, onClose }: Props) {
  return (
    <>
      {/* dark overlay */}
      <div
        className={`fixed inset-0 z-40 bg-black/40 backdrop-blur-sm transition-opacity ${
          open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        }`}
        onClick={onClose}
      />

      {/* panel */}
      <aside
        className={`fixed z-50 left-0 top-0 h-full w-[280px] bg-base-900/80 border-r border-white/10 backdrop-blur-xl ai-card/flat
        transition-transform duration-300 ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-2">
            <UsersRound className="h-5 w-5 text-cyan-300" />
            <span className="font-medium">Team Members</span>
          </div>
          <button
            aria-label="Close"
            onClick={onClose}
            className="rounded-xl p-1 hover:bg-white/5 border border-white/10"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-4 pb-6 space-y-2">
          {TEAM.map((m) => (
            <div
              key={m.name}
              className="flex items-center gap-3 rounded-xl px-3 py-2 bg-white/5 border border-white/10 hover:bg-white/7 transition group"
            >
              <div className="shimmer h-9 w-9 rounded-full grid place-items-center text-sm font-medium">
                {m.name
                  .split(" ")
                  .map((s) => s[0])
                  .join("")
                  .slice(0, 3)
                  .toUpperCase()}
              </div>
              <div className="flex-1">
                <p className="text-sm">{m.name}</p>
                <p className="text-xs text-white/50">Core Contributor</p>
              </div>
            </div>
          ))}

          <p className="text-[11px] text-white/40 pt-4 leading-5">
            CareerCompass is powered by AI. Upload CVs and let the model score jobs by semantic similarity. ✨
          </p>
        </div>
      </aside>
    </>
  );
}
