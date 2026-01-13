import type { CVFile, RankedCV } from "../store";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

type MatchResult = {
  filename: string;
  score: number;
  highlights: string[];
};

type MatchResponse = {
  job_description_preview: string;
  count: number;
  results: MatchResult[];
};

export async function healthCheck(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error("Backend health check failed");
  return res.json();
}

/**
 * Calls FastAPI POST /match
 * - multipart/form-data
 * - job_description (string)
 * - cvs (one or more PDFs)
 * - top_k (optional)
 */
export async function rankCVsAgainstJob(params: {
  jobDescription: string;
  files: File[];
  topK?: number;
}): Promise<RankedCV[]> {
  const { jobDescription, files, topK } = params;

  const form = new FormData();
  form.append("job_description", jobDescription);

  if (typeof topK === "number") {
    form.append("top_k", String(topK));
  }

  // IMPORTANT: field name must be "cvs" (plural)
  for (const f of files) {
    form.append("cvs", f);
  }

  const res = await fetch(`${API_BASE_URL}/match`, {
    method: "POST",
    body: form,
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    // FastAPI errors often return { detail: "..." } or { detail: [...] }
    const msg =
      typeof data?.detail === "string"
        ? data.detail
        : "Match request failed";
    throw new Error(msg);
  }

  const parsed = data as MatchResponse;

  // Map backend response -> frontend type
  return (parsed.results || []).map((r, i) => ({
    id: `${Date.now()}-${i}`,
    filename: r.filename,
    score: r.score,
    highlights: r.highlights || [],
  }));
}

// Keep your upload metadata helper (frontend-only)
export async function uploadCVs(files: File[]): Promise<CVFile[]> {
  return files.map((f, i) => ({
    id: `${Date.now()}-${i}`,
    name: f.name,
    size: f.size,
    file: f,
  }));
}
