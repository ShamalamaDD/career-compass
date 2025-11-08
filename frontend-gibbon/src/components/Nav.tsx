import { Menu } from "lucide-react";

type Props = {
  onMenu: () => void;
};

export default function Nav({ onMenu }: Props) {
  return (
    <header className="sticky top-0 z-30 border-b border-white/10 backdrop-blur-xl bg-base-900/70">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            aria-label="Open menu"
            onClick={onMenu}
            className="rounded-xl p-2 hover:bg-white/5 border border-white/10"
          >
            <Menu className="h-4 w-4" />
          </button>
          <div className="font-semibold gradient-text">CareerCompass AI</div>
        </div>

        {/* Right side kept clean (GitHub removed) */}
        <div className="text-xs text-white/50 hidden sm:block">
          ✍️ CV Ranking System
        </div>
      </div>
    </header>
  );
}
