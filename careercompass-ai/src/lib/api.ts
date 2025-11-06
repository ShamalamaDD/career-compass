import type { CVFile, RankedJob } from '../store';
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

export async function uploadCVs(files: File[]): Promise<CVFile[]> {
  await sleep(300);
  return files.map((f, i) => ({ id: `${Date.now()}-${i}`, name: f.name, size: f.size }));
}

export async function rankJobs(): Promise<RankedJob[]> {
  await sleep(900);
  const base = [
    { id: '1', title: 'Machine Learning Engineer', company: 'NeuroNet Labs', summary: 'Build LLM pipelines & retrieval.', tags: ['Python','LLMs','Vector DB'] },
    { id: '2', title: 'Full-Stack Developer', company: 'OrbitWorks',        summary: 'Ship React/Node apps end-to-end.', tags: ['React','Node','PostgreSQL'] },
    { id: '3', title: 'Data Scientist',            company: 'QuantaIQ',          summary: 'Own experimentation & inference.', tags: ['Pandas','A/B','Stats'] },
  ];
  return base.map(j => ({ ...j, match: Math.min(99, 70 + Math.round(Math.random() * 29)) }));
}
