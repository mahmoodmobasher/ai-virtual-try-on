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
  // Diffusion step progress; null/undefined means no step-wise progress is
  // available yet (e.g. before the first step, or a hosted engine).
  progress_current?: number | null;
  progress_total?: number | null;
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

export type VideoJob = {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed" | "cancelled";
  frames_done: number;
  frames_total: number;
  result_video_path?: string | null;
  failure_reason?: string | null;
  created_at: number;
};

export type DetectedFace = {
  face_index: number;
  bbox: number[];
  det_score: number;
  thumbnail_url: string;
};

export type FaceScanResult = {
  upload_id: string;
  faces: DetectedFace[];
};

export async function scanVideoFaces(form: FormData): Promise<FaceScanResult> {
  const { data } = await axios.post<FaceScanResult>(`${API_BASE}/video/faceswap/scan`, form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
}

export async function createVideoFaceSwap(form: FormData): Promise<VideoJob> {
  const { data } = await axios.post<VideoJob>(`${API_BASE}/video/faceswap`, form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
}

export async function getVideoFaceSwapJob(jobId: string): Promise<VideoJob> {
  const { data } = await axios.get<VideoJob>(`${API_BASE}/video/faceswap/${jobId}`);
  return data;
}

export async function cancelVideoFaceSwapJob(jobId: string): Promise<VideoJob> {
  const { data } = await axios.post<VideoJob>(`${API_BASE}/video/faceswap/${jobId}/cancel`);
  return data;
}
