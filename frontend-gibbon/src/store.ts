import { create } from "zustand";

export type CVFile = { id: string; name: string; size: number; file: File };

export type RankedCV = {
  id: string;
  filename: string;
  score: number;
  highlights: string[];
};

type State = {
  cvs: CVFile[];
  results: RankedCV[];
  loading: boolean;
  addCVs: (x: CVFile[]) => void;
  setResults: (x: RankedCV[]) => void;
  setLoading: (x: boolean) => void;
  clear: () => void;
};

export const useApp = create<State>((set) => ({
  cvs: [],
  results: [],
  loading: false,
  addCVs: (x) => set((s) => ({ cvs: [...s.cvs, ...x] })),
  setResults: (x) => set({ results: x }),
  setLoading: (x) => set({ loading: x }),
  clear: () => set({ cvs: [], results: [], loading: false }),
}));
