import { useState } from "react";
import Brand from "./components/Brand";
import Nav from "./components/Nav";
import Dropzone from "./components/Dropzone";
import JobCard from "./components/JobCard";
import Loader from "./components/Loader";
import SideNav from "./components/SideNav";
import BackgroundFX from "./components/BackgroundFX";
import { useApp, type CVFile, type RankedJob } from "./store";
import { uploadCVs, rankJobs } from "./lib/api";

export default function App() {
  const { cvs, jobs, loading, addCVs, setJobs, setLoading, clear } = useApp();
  const [open, setOpen] = useState(false);

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
    <>
      <BackgroundFX />
      <Nav onMenu={() => setOpen(true)} />
      <SideNav open={open} onClose={() => setOpen(false)} />

      <main className="max-w-6xl mx-auto px-4 py-8 grid grid-cols-1 md:grid-cols-[280px,1fr] gap-8">
        {/* Left column – welcome/CTA card */}
        <aside className="space-y-6">
          <Brand />

          <div className="ai-card">
            <h3 className="font-medium">How it works</h3>
            <ul className="mt-3 space-y-2 text-sm text-white/70">
              <li>1. Upload one or more CVs (PDF).</li>
              <li>2. Click <span className="gradient-text">Rank Jobs</span>.</li>
              <li>3. Explore AI-scored opportunities.</li>
            </ul>
          </div>
        </aside>

        {/* Right column – core flow */}
        <section className="space-y-6">
          <div className="ai-card neon">
            <h1 className="text-2xl font-semibold">Upload CVs</h1>
            <p className="text-sm text-white/60 mt-1">
              Drag and drop your CV PDFs below.
            </p>

            <div className="mt-4">
              <Dropzone onFiles={handleFiles} />
            </div>

            {cvs.length > 0 && (
              <div className="mt-4 rounded-xl bg-white/5 border border-white/10 p-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-sm">Selected CVs</h4>
                  <button
                    onClick={clear}
                    className="text-xs px-2 py-1 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10"
                  >
                    Clear
                  </button>
                </div>
                <ul className="mt-2 grid sm:grid-cols-2 gap-2 text-sm">
                  {cvs.map((cv: CVFile) => (
                    <li key={cv.id} className="rounded-lg bg-base-900/60 border border-white/10 px-3 py-2">
                      {cv.name} <span className="text-white/50">({Math.round(cv.size / 1024)} KB)</span>
                    </li>
                  ))}
                </ul>

                <button
                  onClick={handleRank}
                  className="mt-4 w-full py-2 rounded-xl bg-gradient-to-r from-cyan-500/20 to-emerald-500/20 border border-white/10 hover:from-cyan-500/30 hover:to-emerald-500/30"
                >
                  Rank Jobs
                </button>
              </div>
            )}
          </div>

          {loading && <Loader label="Scoring job matches…" />}

          {jobs.length > 0 && (
            <div className="ai-card">
              <h3 className="font-medium">Top Matches</h3>
              <div className="mt-3 grid md:grid-cols-2 gap-4">
                {jobs.map((j: RankedJob) => (
                  <JobCard key={j.id} job={j} />
                ))}
              </div>
            </div>
          )}

          <footer className="text-xs text-white/40 text-center py-6">
            Made for CheatCode — Team: Mitheel Ramdaw · Ryan Chitate · Mikhaar Ramdaw · Lionel Muke
          </footer>
        </section>
      </main>
    </>
  );
}
