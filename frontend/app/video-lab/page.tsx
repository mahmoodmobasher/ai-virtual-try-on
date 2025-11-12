"use client";

import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import Link from "next/link";

import { BACKEND_BASE, VideoClipResponse, createVideoClip } from "@/lib/api";

export default function VideoLabPage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [frames, setFrames] = useState<number>(14);
  const [fps, setFps] = useState<number>(6);
  const [guidance, setGuidance] = useState<number>(1.2);
  const [motionBucket, setMotionBucket] = useState<number>(127);
  const [noiseStrength, setNoiseStrength] = useState<number>(0.02);
  const [prompt, setPrompt] = useState<string>("");
  const [seed, setSeed] = useState<number | "">("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VideoClipResponse | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (imagePreview) {
        URL.revokeObjectURL(imagePreview);
      }
    };
  }, [imagePreview]);

  const handleImageChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      setImageFile(null);
      setImagePreview(null);
      return;
    }
    if (!file.type.startsWith("image/")) {
      setError("Please upload a valid image file.");
      return;
    }
    setError(null);
    setImageFile(file);
    const preview = URL.createObjectURL(file);
    setImagePreview(preview);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!imageFile) {
      setError("Upload a source image first.");
      return;
    }

    const formData = new FormData();
    formData.append("user_image", imageFile);
    formData.append("frames", frames.toString());
    formData.append("fps", fps.toString());
    formData.append("guidance_scale", guidance.toString());
    formData.append("motion_bucket_id", motionBucket.toString());
    formData.append("noise_aug_strength", noiseStrength.toString());
    if (prompt.trim()) {
      formData.append("prompt", prompt);
    }
    if (seed !== "" && Number.isFinite(Number(seed))) {
      formData.append("seed", Number(seed).toString());
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);
      setVideoUrl(null);
      const clip = await createVideoClip(formData);
      setResult(clip);
      setVideoUrl(`${BACKEND_BASE}${clip.result_video_path}`);
    } catch (clipError) {
      console.error("[VideoLab] generate clip error", clipError);
      setError(clipError instanceof Error ? clipError.message : "Failed to generate clip. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-black text-white">
      <section className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-6 py-12">
        <div className="flex flex-col gap-4">
          <Link href="/" className="text-sm text-white/60 hover:text-white/90">
            ← Back to Studio
          </Link>
          <h1 className="font-display text-4xl tracking-tight">Video Lab (Experimental)</h1>
          <p className="max-w-2xl text-white/70">
            Upload a single model photo and we&apos;ll turn it into a short motion clip using Stable Video Diffusion. This
            lab runs on heavyweight models and works best with a GPU. Expect generation to take 30–60 seconds.
          </p>
        </div>

        <div className="grid gap-12 lg:grid-cols-2">
          <form onSubmit={handleSubmit} className="flex flex-col gap-6 rounded-3xl border border-white/10 bg-white/5 p-6">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-semibold uppercase tracking-wide text-white/70">Source Image</label>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-3 text-sm"
              />
              <p className="text-xs text-white/40">Square images (≈576×576) produce the cleanest motion.</p>
              {imagePreview ? (
                <img src={imagePreview} alt="Preview" className="mt-2 w-full rounded-2xl border border-white/10" />
              ) : null}
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <label className="flex flex-col gap-1 text-sm">
                <span className="font-semibold text-white/70">Frames</span>
                <input
                  type="number"
                  min={8}
                  max={24}
                  value={frames}
                  onChange={(event) => setFrames(Number(event.target.value))}
                  className="rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                <span className="font-semibold text-white/70">Frames Per Second</span>
                <input
                  type="number"
                  min={2}
                  max={24}
                  value={fps}
                  onChange={(event) => setFps(Number(event.target.value))}
                  className="rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                <span className="font-semibold text-white/70">Guidance Scale</span>
                <input
                  type="number"
                  step={0.1}
                  min={0}
                  max={10}
                  value={guidance}
                  onChange={(event) => setGuidance(Number(event.target.value))}
                  className="rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                <span className="font-semibold text-white/70">Motion Bucket</span>
                <input
                  type="number"
                  min={0}
                  max={255}
                  value={motionBucket}
                  onChange={(event) => setMotionBucket(Number(event.target.value))}
                  className="rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                <span className="font-semibold text-white/70">Noise Augmentation Strength</span>
                <input
                  type="number"
                  step={0.01}
                  min={0}
                  max={1}
                  value={noiseStrength}
                  onChange={(event) => setNoiseStrength(Number(event.target.value))}
                  className="rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                <span className="font-semibold text-white/70">Seed (optional)</span>
                <input
                  type="number"
                  value={seed === "" ? "" : seed}
                  onChange={(event) =>
                    setSeed(event.target.value === "" ? "" : Number(event.target.value))
                  }
                  className="rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm"
                />
              </label>
            </div>

            <label className="flex flex-col gap-2 text-sm">
              <span className="font-semibold text-white/70">Creative Prompt (optional)</span>
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="Soft studio lighting, gentle camera arc, focus on fabric detail"
                className="min-h-[96px] rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm"
              />
            </label>

            {error ? <p className="text-sm text-red-400">{error}</p> : null}

            <button
              type="submit"
              disabled={loading}
              className="rounded-full bg-white px-6 py-3 text-sm font-semibold text-black transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Generating clip..." : "Generate clip"}
            </button>
          </form>

          <aside className="flex flex-col gap-6 rounded-3xl border border-white/10 bg-white/5 p-6">
            <div>
              <h2 className="text-lg font-semibold tracking-wide text-white/80">Output</h2>
              <p className="text-sm text-white/50">Video results will appear here once rendering completes.</p>
            </div>

            {loading ? <p className="text-sm text-white/60">Running Stable Video Diffusion… this can take a minute.</p> : null}

            {videoUrl ? (
              <video
                key={videoUrl}
                src={videoUrl}
                className="w-full rounded-2xl border border-white/10"
                controls
                autoPlay
                loop
                muted
              />
            ) : (
              <div className="flex h-64 w-full items-center justify-center rounded-2xl border border-dashed border-white/20 text-sm text-white/40">
                No clip yet
              </div>
            )}

            {result ? (
              <div className="rounded-2xl border border-white/10 bg-black/40 p-4 text-sm text-white/70">
                <div className="flex items-center justify-between"><span>Frames</span><span>{result.frames}</span></div>
                <div className="flex items-center justify-between"><span>Frame rate</span><span>{result.fps} fps</span></div>
                <div className="flex items-center justify-between">
                  <span>Duration</span>
                  <span>{result.duration_seconds.toFixed(1)} s</span>
                </div>
                <a
                  href={videoUrl ?? "#"}
                  download
                  className="mt-4 inline-flex items-center justify-center rounded-full border border-white/20 px-5 py-2 font-semibold text-white transition hover:bg-white/10"
                >
                  Download MP4
                </a>
              </div>
            ) : null}
          </aside>
        </div>
      </section>
    </main>
  );
}
