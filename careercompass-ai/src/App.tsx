import { useState } from 'react';
import Brand from './components/Brand';
import Nav from './components/Nav';
import Dropzone from './components/Dropzone';
import TeamSheet from './components/TeamSheet';
import JobCard from './components/JobCard';
import Loader from './components/Loader';
import { useApp, type CVFile, type RankedJob } from './store';
import { uploadCVs, rankJobs } from './lib/api';

export default function App() {
  const { cvs, jobs, loading, addCVs, setJobs, setLoading, clear } = useApp();
  const [teamOpen, setTeamOpen] = useState(false);

  async function handleFiles(files: File[]) {
    setLoading(true);
    const uploaded = await uploadCVs(files);
    addCVs(uploaded);
    setLoading(false);
  }

  async function handleRank() {
    setLoading(true);
    const result = await rankJobs();
    setJobs(result);
    setLoading(false);
  }

  return (
    <div>
      <Nav onTeam={() => setTeamOpen(true)} />
      <TeamSheet open={teamOpen} onClose={() => setTeamOpen(false)} />

      <main className="max-w-6xl mx-auto px-4 py-8 grid grid-cols-1 md:grid-cols-[260px,1fr] gap-8">
        <aside className="ai-card h-fit sticky top-20">
          <Brand />
          <p className="mt-4 text-sm text-white/70">
            Welcome to CareerCompass, your personal career guide. Upload one or more CV PDFs and let the AI
            rank job opportunities by similarity.
          </p>
          <button
            onClick={() => setTeamOpen(true)}
            className="mt-4 w-full px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-sm"
          >
            Show Team Members
          </button>
        </aside>

        <section>
          <header className="ai-card">
            <h1 className="text-2xl font-semibold">✍️ CV Ranking System</h1>
            <p className="text-white/60 text-sm mt-1">
              Upload CVs then click <span className="gradient-text font-medium">Rank Jobs</span>.
            </p>
          </header>

          <div className="mt-6 space-y-6">
            <Dropzone onFiles={handleFiles} />

            {cvs.length > 0 && (
              <div className="ai-card">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium">Uploaded CVs</h3>
                  <button onClick={clear} className="text-xs px-2 py-1 rounded-lg bg-white/5 border border-white/10">
                    Clear
                  </button>
                </div>
                <ul className="mt-3 grid sm:grid-cols-2 gap-2 text-sm">
                  {cvs.map((cv: CVFile) => (
                    <li key={cv.id} className="p-2 rounded-lg bg-white/5 border border-white/10">
                      {cv.name} <span className="text-white/50">({Math.round(cv.size / 1024)} KB)</span>
                    </li>
                  ))}
                </ul>
                <button
                  onClick={handleRank}
                  className="mt-4 w-full py-2 rounded-xl bg-gradient-to-r from-cyan-500/20 to-emerald-500/20 border border-white/10"
                >
                  Rank Jobs
                </button>
              </div>
            )}

            {loading && <Loader label="Scoring job matches…" />}

            {jobs.length > 0 && (
              <div className="grid md:grid-cols-2 gap-4">
                {jobs.map((j: RankedJob) => (
                  <JobCard key={j.id} job={j} />
                ))}
              </div>
            )}

            <footer className="text-xs text-white/40 text-center py-6">
              Made for CheatCode — Team: Mitheel Ramdaw · Ryan Chitate · Mikhaar Ramdaw · Lionel Muke
            </footer>
          </div>
        </section>
      </main>
    </div>
  );
}
