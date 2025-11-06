import { create } from 'zustand';

export type CVFile = { id: string; name: string; size: number };
export type RankedJob = {
  id: string;
  title: string;
  company: string;
  summary: string;
  tags: string[];
  match: number;
};

type State = {
  cvs: CVFile[];
  jobs: RankedJob[];
  loading: boolean;
  addCVs: (x: CVFile[]) => void;
  setJobs: (x: RankedJob[]) => void;
  setLoading: (x: boolean) => void;
  clear: () => void;
};

export const useApp = create<State>((set) => ({
  cvs: [],
  jobs: [],
  loading: false,
  addCVs: (x) => set((s) => ({ cvs: [...s.cvs, ...x] })),
  setJobs: (x) => set({ jobs: x }),
  setLoading: (x) => set({ loading: x }),
  clear: () => set({ cvs: [], jobs: [] }),
}));
