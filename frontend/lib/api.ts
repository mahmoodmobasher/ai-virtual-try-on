import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";
export const BACKEND_BASE = API_BASE.replace(/\/api$/, "");

export type Garment = {
  id: number;
  sku: string;
  name: string;
  category: string;
  hero_image_path?: string | null;
  price?: number | null;
  colors: string[];
  sizes: string[];
};

export type TryOnJob = {
  id: number;
  garment_id: number | null;
  garment_image_path?: string | null;
  status: "pending" | "processing" | "completed" | "failed";
  result_image_path?: string | null;
  failure_reason?: string | null;
  meta?: Record<string, unknown> | null;
};

export async function fetchGarments(): Promise<Garment[]> {
  const { data } = await axios.get<Garment[]>(`${API_BASE}/garments/`);
  return data;
}

export async function createTryOn(form: FormData): Promise<TryOnJob> {
  console.log("[API] Sending POST request to", `${API_BASE}/tryon/`);
  const { data } = await axios.post<TryOnJob>(`${API_BASE}/tryon/`, form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  console.log("[API] Received response:", data);
  return data;
}

export async function getTryOnJob(jobId: number): Promise<TryOnJob> {
  const { data } = await axios.get<TryOnJob>(`${API_BASE}/tryon/${jobId}`);
  return data;
}

export type VideoClipResponse = {
  result_video_path: string;
  frames: number;
  fps: number;
  duration_seconds: number;
};

export async function createVideoClip(form: FormData): Promise<VideoClipResponse> {
  console.log("[API] Sending POST request to", `${API_BASE}/video/clip`);
  const { data } = await axios.post<VideoClipResponse>(`${API_BASE}/video/clip`, form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  console.log("[API] Received video clip response:", data);
  return data;
}

export type VideoJob = {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  frames_done: number;
  frames_total: number;
  result_video_path?: string | null;
  failure_reason?: string | null;
  created_at: number;
};

export async function createVideoFaceSwap(form: FormData): Promise<VideoJob> {
  console.log("[API] Sending POST request to", `${API_BASE}/video/faceswap`);
  const { data } = await axios.post<VideoJob>(`${API_BASE}/video/faceswap`, form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
}

export async function getVideoFaceSwapJob(jobId: string): Promise<VideoJob> {
  const { data } = await axios.get<VideoJob>(`${API_BASE}/video/faceswap/${jobId}`);
  return data;
}
