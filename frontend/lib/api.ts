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
  // Faces swapped so far out of the total; null/undefined means no progress
  // is available yet (e.g. before the first face is done).
  progress_current?: number | null;
  progress_total?: number | null;
};

export async function fetchGarments(): Promise<Garment[]> {
  const { data } = await axios.get<Garment[]>(`${API_BASE}/garments/`);
  return data;
}

// The backend's own explanation (FastAPI's `detail`) when it sent one, rather
// than axios's generic "Request failed with status code 400".
export function apiErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : fallback;
}

export type ImageFaceScanResult = FaceScanResult & {
  image_width: number;
  image_height: number;
  max_swap_faces: number;
};

export async function scanImageFaces(form: FormData): Promise<ImageFaceScanResult> {
  const { data } = await axios.post<ImageFaceScanResult>(`${API_BASE}/tryon/faceswap/scan`, form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
}

export async function createImageFaceSwap(form: FormData): Promise<TryOnJob> {
  const { data } = await axios.post<TryOnJob>(`${API_BASE}/tryon/faceswap`, form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
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
  max_swap_faces: number;
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
