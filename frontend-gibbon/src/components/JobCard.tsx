import { Building2 } from 'lucide-react';
import type { RankedJob } from '../store';

export default function JobCard({ job }: { job: RankedJob }) {
  return (
    <div className="ai-card hover:shadow-neon transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <Building2 className="h-6 w-6 text-emerald-300" />
          <div>
            <div className="text-lg font-semibold">{job.title}</div>
            <div className="text-white/60 text-sm">{job.company}</div>
          </div>
        </div>
        <div className="text-2xl font-bold gradient-text">{job.match}%</div>
      </div>
      <p className="mt-3 text-white/80 text-sm">{job.summary}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {job.tags.map(t => (
          <span key={t} className="px-2 py-1 rounded-full text-xs bg-white/5 border border-white/10">
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
