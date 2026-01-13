import type { RankedCV } from "../store";

export default function JobCard({ job }: { job: RankedCV }) {
  return (
    <div className="ai-card hover:shadow-neon transition-shadow">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-lg font-semibold">{job.filename}</div>
          {job.highlights?.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {job.highlights.map((t) => (
                <span
                  key={t}
                  className="px-2 py-1 rounded-full text-xs bg-white/5 border border-white/10"
                >
                  {t}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="text-2xl font-bold gradient-text">
          {job.score.toFixed(2)}%
        </div>
      </div>
    </div>
  );
}
  