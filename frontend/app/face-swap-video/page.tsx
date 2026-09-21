"use client";

import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import Link from "next/link";

import { BACKEND_BASE, VideoJob, createVideoFaceSwap, getVideoFaceSwapJob } from "@/lib/api";

export default function FaceSwapVideoPage() {
  const [targetVideo, setTargetVideo] = useState<File | null>(null);
  const [targetVideoPreview, setTargetVideoPreview] = useState<string | null>(null);
  const [sourceFace, setSourceFace] = useState<File | null>(null);
  const [sourceFacePreview, setSourceFacePreview] = useState<string | null>(null);
  const [frameStride, setFrameStride] = useState<number>(1);
  const [maxFrames, setMaxFrames] = useState<number | "">("");
  const [job, setJob] = useState<VideoJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (targetVideoPreview) URL.revokeObjectURL(targetVideoPreview);
      if (sourceFacePreview) URL.revokeObjectURL(sourceFacePreview);
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [targetVideoPreview, sourceFacePreview]);

  const handleTargetVideoChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      setError("Please upload a video file.");
      return;
    }
    setError(null);
    setTargetVideo(file);
    setTargetVideoPreview(URL.createObjectURL(file));
  };

  const handleSourceFaceChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please upload a JPEG or PNG image.");
      return;
    }
    setError(null);
    setSourceFace(file);
    setSourceFacePreview(URL.createObjectURL(file));
  };

  const pollJob = (jobId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const updated = await getVideoFaceSwapJob(jobId);
        setJob(updated);
        if (updated.status === "completed" || updated.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch (pollError) {
        console.error("[FaceSwapVideo] poll error", pollError);
      }
    }, 2000);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!targetVideo || !sourceFace) {
      setError("Upload both a target video and a source face photo.");
      return;
    }

    const formData = new FormData();
    formData.append("target_video", targetVideo);
    formData.append("source_face_image", sourceFace);
    formData.append("frame_stride", frameStride.toString());
    if (maxFrames !== "" && Number.isFinite(Number(maxFrames))) {
      formData.append("max_frames", Number(maxFrames).toString());
    }

    try {
      setError(null);
      setJob(null);
      const created = await createVideoFaceSwap(formData);
      setJob(created);
      pollJob(created.job_id);
    } catch (submitError) {
      console.error("[FaceSwapVideo] submit error", submitError);
      setError(submitError instanceof Error ? submitError.message : "Failed to start the face swap job.");
    }
  };

  const isRunning = job?.status === "pending" || job?.status === "processing";
  const resultUrl = job?.result_video_path ? `${BACKEND_BASE}${job.result_video_path}` : null;
  const progressPct = job && job.frames_total > 0 ? Math.round((job.frames_done / job.frames_total) * 100) : 0;

  return (
    <main className="min-h-screen bg-black text-white">
      <section className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-6 py-12">
        <div className="flex flex-col gap-4">
          <Link href="/" className="text-sm text-white/60 hover:text-white/90">
            ← Back to Studio
          </Link>
          <h1 className="font-display text-4xl tracking-tight">Face Swap Video (Experimental)</h1>
          <p className="max-w-2xl text-white/70">
            Upload a video and a single source face photo — every frame of the video gets that face swapped in,
            while the original body, pose and audio are preserved. This is a separate feature from image face
            swap and from garment try-on video; it runs entirely on this app&apos;s own local face-swap pipeline.
          </p>
        </div>

        <div className="grid gap-12 lg:grid-cols-2">
          <form onSubmit={handleSubmit} className="flex flex-col gap-6 rounded-3xl border border-white/10 bg-white/5 p-6">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-semibold uppercase tracking-wide text-white/70">Target video</label>
              <input
                type="file"
                accept="video/*"
                onChange={handleTargetVideoChange}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-3 text-sm"
              />
              <p className="text-xs text-white/40">The video whose faces will be replaced.</p>
              {targetVideoPreview ? (
                <video src={targetVideoPreview} className="mt-2 w-full rounded-2xl border border-white/10" controls muted />
              ) : null}
            </div>

            <div className="flex flex-col gap-2">
              <label className="text-sm font-semibold uppercase tracking-wide text-white/70">Source face photo</label>
              <input
                type="file"
                accept="image/*"
                onChange={handleSourceFaceChange}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-3 text-sm"
              />
              <p className="text-xs text-white/40">A single clear, front-facing photo of the face to swap in.</p>
              {sourceFacePreview ? (
                <img src={sourceFacePreview} alt="Source face preview" className="mt-2 w-40 rounded-2xl border border-white/10" />
              ) : null}
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <label className="flex flex-col gap-1 text-sm">
                <span className="font-semibold text-white/70">Frame stride</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={frameStride}
                  onChange={(event) => setFrameStride(Number(event.target.value))}
                  className="rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm"
                />
                <span className="text-xs text-white/40">Higher = faster, less per-frame detail.</span>
              </label>
              <label className="flex flex-col gap-1 text-sm">
                <span className="font-semibold text-white/70">Max frames (optional)</span>
                <input
                  type="number"
                  min={1}
                  value={maxFrames === "" ? "" : maxFrames}
                  onChange={(event) => setMaxFrames(event.target.value === "" ? "" : Number(event.target.value))}
                  className="rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm"
                  placeholder="All frames"
                />
              </label>
            </div>

            {error ? <p className="text-sm text-red-400">{error}</p> : null}

            <button
              type="submit"
              disabled={isRunning}
              className="rounded-full bg-white px-6 py-3 text-sm font-semibold text-black transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isRunning ? "Swapping faces…" : "Generate face-swap video"}
            </button>
          </form>

          <aside className="flex flex-col gap-6 rounded-3xl border border-white/10 bg-white/5 p-6">
            <div>
              <h2 className="text-lg font-semibold tracking-wide text-white/80">Output</h2>
              <p className="text-sm text-white/50">The face-swapped video will appear here once rendering completes.</p>
            </div>

            {job ? (
              <div className="rounded-2xl border border-white/10 bg-black/40 p-4 text-sm text-white/70">
                <div className="flex items-center justify-between">
                  <span>Status</span>
                  <span className="capitalize">{job.status}</span>
                </div>
                {job.frames_total > 0 ? (
                  <div className="mt-3">
                    <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                      <div className="h-full bg-white transition-all" style={{ width: `${progressPct}%` }} />
                    </div>
                    <p className="mt-1 text-xs text-white/50">
                      {job.frames_done} / {job.frames_total} frames ({progressPct}%)
                    </p>
                  </div>
                ) : null}
                {job.status === "failed" && job.failure_reason ? (
                  <p className="mt-3 text-sm text-red-400">{job.failure_reason}</p>
                ) : null}
              </div>
            ) : null}

            {resultUrl ? (
              <video key={resultUrl} src={resultUrl} className="w-full rounded-2xl border border-white/10" controls autoPlay loop />
            ) : (
              <div className="flex h-64 w-full items-center justify-center rounded-2xl border border-dashed border-white/20 text-sm text-white/40">
                No video yet
              </div>
            )}

            {resultUrl ? (
              <a
                href={resultUrl}
                download
                className="inline-flex items-center justify-center rounded-full border border-white/20 px-5 py-2 text-sm font-semibold text-white transition hover:bg-white/10"
              >
                Download MP4
              </a>
            ) : null}
          </aside>
        </div>
      </section>
    </main>
  );
}
