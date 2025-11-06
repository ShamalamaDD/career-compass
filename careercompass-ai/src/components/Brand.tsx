import { Compass } from 'lucide-react';
export default function Brand() {
  return (
    <div className="flex items-center gap-3">
      <Compass className="h-8 w-8 text-cyan-300" />
      <div>
        <div className="text-xl font-semibold">
          CareerCompass <span className="text-cyan-300">AI</span>
        </div>
        <div className="text-xs text-white/50 -mt-1">CV → Job Matching</div>
      </div>
    </div>
  );
}
