import { Github, Moon, Sun, UsersRound } from 'lucide-react';
import { useEffect, useState } from 'react';

export default function Nav({ onTeam }: { onTeam: () => void }) {
  const [dark, setDark] = useState(true);
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
  }, [dark]);

  return (
    <nav className="sticky top-0 z-40 backdrop-blur bg-base-900/70 border-b border-white/10">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <span className="gradient-text font-medium">CV Ranking System</span>
        <div className="flex items-center gap-3">
          <button onClick={onTeam} className="glass px-3 py-2 rounded-xl text-sm flex items-center gap-2">
            <UsersRound className="h-4 w-4" /> Team
          </button>
          <a className="glass p-2 rounded-xl" href="https://github.com/" target="_blank" rel="noreferrer">
            <Github className="h-5 w-5" />
          </a>
          <button className="glass p-2 rounded-xl" onClick={() => setDark(d => !d)}>
            {dark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </button>
        </div>
      </div>
    </nav>
  );
}
