"use client";

import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";

import { Navigation } from "@/components/navigation";
import {
  BACKEND_BASE,
  FaceScanResult,
  VideoJob,
  cancelVideoFaceSwapJob,
  createVideoFaceSwap,
  getVideoFaceSwapJob,
  scanVideoFaces
} from "@/lib/api";

// Fallback until the scan response reports the backend's cap.
const DEFAULT_MAX_FACES = 3;

type Replacement = { file: File; preview: string };

export default function FaceSwapVideoPage() {
  const [targetVideo, setTargetVideo] = useState<File | null>(null);
  const [targetVideoPreview, setTargetVideoPreview] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scan, setScan] = useState<FaceScanResult | null>(null);
  const maxFaces = scan?.max_swap_faces ?? DEFAULT_MAX_FACES;
  const [replacements, setReplacements] = useState<Record<number, Replacement | null>>({});
  const [frameStride, setFrameStride] = useState<number>(1);
  const [maxFrames, setMaxFrames] = useState<number | "">("");
  const [noseRing, setNoseRing] = useState(false);
  const [job, setJob] = useState<VideoJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isStopping, setIsStopping] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleTargetVideoChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      setError("Please upload a video file.");
      return;
    }

    if (targetVideoPreview) URL.revokeObjectURL(targetVideoPreview);
    Object.values(replacements).forEach((entry) => entry && URL.revokeObjectURL(entry.preview));

    setError(null);
    setTargetVideo(file);
    setTargetVideoPreview(URL.createObjectURL(file));
    setScan(null);
    setReplacements({});
    setJob(null);

    const formData = new FormData();
    formData.append("video", file);

    try {
      setIsScanning(true);
      const result = await scanVideoFaces(formData);
      setScan(result);
      if (result.faces.length === 0) {
        setError("No faces were found in that video. Try a clip where faces are clearly visible.");
      }
    } catch (scanError) {
      console.error("[FaceSwapVideo] scan error", scanError);
      setError(scanError instanceof Error ? scanError.message : "Failed to scan the video for faces.");
    } finally {
      setIsScanning(false);
    }
  };

  const toggleFace = (faceIndex: number) => {
    setError(null);
    setReplacements((current) => {
      const next = { ...current };
      if (faceIndex in next) {
        const entry = next[faceIndex];
        if (entry) URL.revokeObjectURL(entry.preview);
        delete next[faceIndex];
        return next;
      }
      if (Object.keys(next).length >= maxFaces) {
        setError(`You can swap up to ${maxFaces} faces at a time.`);
        return current;
      }
      next[faceIndex] = null;
      return next;
    });
  };

  const handleReplacementChange = (faceIndex: number) => (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please upload a JPEG or PNG image.");
      return;
    }
    setError(null);
    setReplacements((current) => {
      const previous = current[faceIndex];
      if (previous) URL.revokeObjectURL(previous.preview);
      return { ...current, [faceIndex]: { file, preview: URL.createObjectURL(file) } };
    });
  };

  const pollJob = (jobId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const updated = await getVideoFaceSwapJob(jobId);
        setJob(updated);
        if (updated.status === "completed" || updated.status === "failed" || updated.status === "cancelled") {
          if (pollRef.current) clearInterval(pollRef.current);
          setIsStopping(false);
        }
      } catch (pollError) {
        console.error("[FaceSwapVideo] poll error", pollError);
      }
    }, 2000);
  };

  const selectedIndices = Object.keys(replacements).map(Number);
  const readyIndices = selectedIndices.filter((index) => replacements[index]);
  const canSubmit = scan !== null && selectedIndices.length > 0 && readyIndices.length === selectedIndices.length;

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!scan || !canSubmit) {
      setError("Choose at least one face from the video and upload a replacement photo for it.");
      return;
    }

    const formData = new FormData();
    formData.append("upload_id", scan.upload_id);
    selectedIndices.forEach((faceIndex) => {
      const entry = replacements[faceIndex];
      if (!entry) return;
      formData.append("source_face_images", entry.file);
      formData.append("target_face_indices", faceIndex.toString());
    });
    formData.append("frame_stride", frameStride.toString());
    formData.append("nose_ring", noseRing.toString());
    if (maxFrames !== "" && Number.isFinite(Number(maxFrames))) {
      formData.append("max_frames", Number(maxFrames).toString());
    }

    try {
      setError(null);
      setJob(null);
      setIsStopping(false);
      const created = await createVideoFaceSwap(formData);
      setJob(created);
      pollJob(created.job_id);
    } catch (submitError) {
      console.error("[FaceSwapVideo] submit error", submitError);
      setError(submitError instanceof Error ? submitError.message : "Failed to start the face swap job.");
    }
  };

  const handleStop = async () => {
    if (!job) return;
    try {
      setIsStopping(true);
      await cancelVideoFaceSwapJob(job.job_id);
    } catch (stopError) {
      console.error("[FaceSwapVideo] stop error", stopError);
      setIsStopping(false);
      setError(stopError instanceof Error ? stopError.message : "Failed to stop the job.");
    }
  };

  const isRunning = job?.status === "pending" || job?.status === "processing";
  const resultUrl = job?.result_video_path ? `${BACKEND_BASE}${job.result_video_path}` : null;
  const progressPct = job && job.frames_total > 0 ? Math.round((job.frames_done / job.frames_total) * 100) : 0;

  return (
    <div className="relative overflow-hidden">
      <div className="absolute inset-0 -z-20 dot-grid opacity-30" />
      <div className="absolute -top-48 -left-32 h-96 w-96 rounded-full bg-brand-500 blur-3xl opacity-20 -z-10" />
      <div className="absolute top-96 -right-24 h-96 w-96 rounded-full bg-fuchsia-500 blur-3xl opacity-20 -z-10" />
      <Navigation />
      <section className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-6 py-16">
        <div className="flex flex-col gap-2">
          <h1 className="font-display text-4xl text-white">Face Swap Video (Experimental)</h1>
          <p className="max-w-2xl text-base text-white/60">
            Upload a video and the app scans the whole clip for faces, including people who only appear partway
            through. Pick up to {maxFaces} of the people it finds, give each one a replacement photo, and every
            frame gets re-faced — body, pose and audio preserved. Each chosen person is tracked by face identity
            across frames, so the swap stays on the right person. Runs entirely on this machine.
          </p>
        </div>

        <div className="grid gap-12 lg:grid-cols-2">
          <form onSubmit={handleSubmit} className="flex flex-col gap-6 rounded-3xl border border-white/10 bg-white/5 p-6">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-semibold uppercase tracking-wide text-white/70">1 · Target video</label>
              <input
                type="file"
                accept="video/*"
                onChange={handleTargetVideoChange}
                disabled={isScanning || isRunning}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-3 text-sm disabled:opacity-60"
              />
              <p className="text-xs text-white/40">The video whose faces will be replaced.</p>
              {targetVideoPreview ? (
                <video src={targetVideoPreview} className="mt-2 w-full rounded-2xl border border-white/10" controls muted />
              ) : null}
            </div>

            <div className="flex flex-col gap-3">
              <label className="text-sm font-semibold uppercase tracking-wide text-white/70">
                2 · Choose faces to swap
              </label>

              {isScanning ? (
                <p className="text-sm text-white/50">Scanning the whole video for faces…</p>
              ) : !targetVideo ? (
                <p className="text-xs text-white/40">Upload a video first — its faces will show up here.</p>
              ) : scan && scan.faces.length > 0 ? (
                <>
                  <p className="text-xs text-white/40">
                    Found {scan.faces.length} face{scan.faces.length === 1 ? "" : "s"}. Click the ones you want to
                    replace (up to {maxFaces}), then add a photo for each.
                  </p>
                  <div className="grid grid-cols-3 gap-3 sm:grid-cols-4">
                    {scan.faces.map((face) => {
                      const isSelected = face.face_index in replacements;
                      return (
                        <button
                          key={face.face_index}
                          type="button"
                          onClick={() => toggleFace(face.face_index)}
                          disabled={isRunning}
                          className={`overflow-hidden rounded-2xl border-2 transition ${
                            isSelected
                              ? "border-white ring-2 ring-white/40"
                              : "border-white/10 opacity-70 hover:opacity-100"
                          } disabled:cursor-not-allowed`}
                        >
                          <img
                            src={`${BACKEND_BASE}${face.thumbnail_url}`}
                            alt={`Face ${face.face_index + 1} from the video`}
                            className="aspect-square w-full object-cover"
                          />
                        </button>
                      );
                    })}
                  </div>

                  {selectedIndices.length > 0 ? (
                    <div className="mt-2 flex flex-col gap-4">
                      {selectedIndices
                        .sort((a, b) => a - b)
                        .map((faceIndex) => {
                          const face = scan.faces.find((item) => item.face_index === faceIndex);
                          const entry = replacements[faceIndex];
                          return (
                            <div
                              key={faceIndex}
                              className="flex items-center gap-4 rounded-2xl border border-white/10 bg-black/30 p-3"
                            >
                              {face ? (
                                <img
                                  src={`${BACKEND_BASE}${face.thumbnail_url}`}
                                  alt={`Face ${faceIndex + 1} from the video`}
                                  className="h-16 w-16 shrink-0 rounded-xl object-cover"
                                />
                              ) : null}
                              <span className="text-white/40">→</span>
                              {entry ? (
                                <img
                                  src={entry.preview}
                                  alt={`Replacement face for face ${faceIndex + 1}`}
                                  className="h-16 w-16 shrink-0 rounded-xl object-cover"
                                />
                              ) : (
                                <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-xl border border-dashed border-white/20 text-[10px] text-white/40">
                                  No photo
                                </div>
                              )}
                              <input
                                type="file"
                                accept="image/*"
                                onChange={handleReplacementChange(faceIndex)}
                                disabled={isRunning}
                                className="min-w-0 flex-1 rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-xs disabled:opacity-60"
                              />
                            </div>
                          );
                        })}
                    </div>
                  ) : null}
                </>
              ) : scan ? (
                <p className="text-sm text-white/50">No faces found in that video. Try a different clip.</p>
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

            <label className="flex items-start gap-3 rounded-2xl border border-white/10 bg-black/30 p-4 text-sm">
              <input
                type="checkbox"
                checked={noseRing}
                onChange={(event) => setNoseRing(event.target.checked)}
                disabled={isRunning}
                className="mt-0.5 h-4 w-4 accent-amber-400"
              />
              <span className="flex flex-col gap-1">
                <span className="font-semibold text-white/70">Add nose ring</span>
                <span className="text-xs text-white/40">
                  Adds a gold hoop to the left nostril of each swapped face. It follows the face as it moves and
                  fades out when the head turns that side away from the camera.
                </span>
              </span>
            </label>

            {error ? <p className="text-sm text-red-400">{error}</p> : null}

            <div className="flex gap-3">
              <button
                type="submit"
                disabled={isRunning || isScanning || !canSubmit}
                className="flex-1 rounded-full bg-white px-6 py-3 text-sm font-semibold text-black transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isRunning ? "Swapping faces…" : "Generate face-swap video"}
              </button>
              {isRunning ? (
                <button
                  type="button"
                  onClick={handleStop}
                  disabled={isStopping}
                  className="rounded-full border border-red-400/40 px-6 py-3 text-sm font-semibold text-red-300 transition hover:bg-red-400/10 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isStopping ? "Stopping…" : "Stop"}
                </button>
              ) : null}
            </div>
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
                {job.status === "cancelled" ? (
                  <p className="mt-3 text-sm text-white/50">
                    Stopped early — the video below only includes the frames processed before you stopped it.
                  </p>
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
    </div>
  );
}
