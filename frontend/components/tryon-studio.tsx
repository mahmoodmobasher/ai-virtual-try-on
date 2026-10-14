"use client";

import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

import {
  apiErrorMessage,
  BACKEND_BASE,
  createImageFaceSwap,
  getTryOnJob,
  ImageFaceScanResult,
  scanImageFaces,
  TryOnJob
} from "@/lib/api";

// Used until a scan reports the backend's actual limit.
const DEFAULT_MAX_FACES = 4;

type Replacement = { file: File; preview: string };

type TryOnState = {
  job?: TryOnJob;
  status: "idle" | "uploading" | "processing" | "done" | "error";
  previewUrl?: string;
  error?: string;
};

export function TryOnStudio() {
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [photoName, setPhotoName] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scan, setScan] = useState<ImageFaceScanResult | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  // Chosen faces by scan index; null until a replacement photo is added.
  const [replacements, setReplacements] = useState<Record<number, Replacement | null>>({});
  const [state, setState] = useState<TryOnState>({ status: "idle" });
  const isMounted = useRef(true);
  const photoObjectUrlRef = useRef<string | null>(null);
  const replacementsRef = useRef(replacements);
  replacementsRef.current = replacements;
  // Guards against a slow scan of a previous photo landing after a newer upload.
  const scanRequestRef = useRef(0);
  const uploadInputId = "faceswap-upload-photo";

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      if (photoObjectUrlRef.current) {
        URL.revokeObjectURL(photoObjectUrlRef.current);
      }
      Object.values(replacementsRef.current).forEach((entry) => entry && URL.revokeObjectURL(entry.preview));
    };
  }, []);

  const maxFaces = scan?.max_swap_faces ?? DEFAULT_MAX_FACES;
  const selectedIndices = Object.keys(replacements)
    .map(Number)
    .sort((a, b) => a - b);
  const canGenerate = scan !== null && selectedIndices.length > 0 && selectedIndices.every((index) => replacements[index]);

  const mutation = useMutation({
    mutationFn: async () => {
      if (!scan || !canGenerate) {
        throw new Error("Choose at least one face and add a replacement photo for it.");
      }
      const formData = new FormData();
      formData.append("upload_id", scan.upload_id);
      selectedIndices.forEach((faceIndex) => {
        const entry = replacements[faceIndex];
        if (!entry) return;
        formData.append("source_face_images", entry.file);
        formData.append("target_face_indices", faceIndex.toString());
      });
      return createImageFaceSwap(formData);
    },
    onMutate: () => {
      if (!isMounted.current) return;
      setState({ status: "uploading" });
    },
    onError: (error: Error) => {
      if (!isMounted.current) return;
      setState((prev) => ({ ...prev, status: "error", error: apiErrorMessage(error, "Failed to start the face swap.") }));
    },
    onSuccess: async (job) => {
      if (job.status === "completed") {
        const url = job.result_image_path ? `${BACKEND_BASE}${job.result_image_path}` : undefined;
        setState({ status: "done", job, previewUrl: url });
      } else if (job.status === "failed") {
        setState({ status: "error", error: job.failure_reason ?? "Processing failed", job });
      } else {
        setState((prev) => ({ ...prev, status: "processing", job }));
        if (isMounted.current) {
          pollJob(job.id);
        }
      }
    }
  });

  const pollJob = async (jobId: number) => {
    try {
      const job = await getTryOnJob(jobId);
      if (!isMounted.current) return;
      if (job.status === "completed") {
        const url = job.result_image_path ? `${BACKEND_BASE}${job.result_image_path}` : undefined;
        setState({ status: "done", job, previewUrl: url });
      } else if (job.status === "failed") {
        setState({ status: "error", error: job.failure_reason ?? "Processing failed", job });
      } else {
        setState((prev) => ({ ...prev, status: "processing", job }));
        setTimeout(() => {
          if (isMounted.current) pollJob(jobId);
        }, 1500);
      }
    } catch (error) {
      setTimeout(() => {
        if (isMounted.current) pollJob(jobId);
      }, 1500);
    }
  };

  const clearSelection = () => {
    Object.values(replacements).forEach((entry) => entry && URL.revokeObjectURL(entry.preview));
    setReplacements({});
    setScan(null);
    setScanError(null);
  };

  const clearPhoto = () => {
    scanRequestRef.current += 1;
    if (photoObjectUrlRef.current) {
      URL.revokeObjectURL(photoObjectUrlRef.current);
      photoObjectUrlRef.current = null;
    }
    setPhotoPreview(null);
    setPhotoName(null);
    setIsScanning(false);
    clearSelection();
  };

  const handleUpload = async (fileList: FileList | null) => {
    const file = fileList?.[0];
    if (!file) return;

    clearPhoto();
    const objectUrl = URL.createObjectURL(file);
    photoObjectUrlRef.current = objectUrl;
    setPhotoPreview(objectUrl);
    setPhotoName(file.name);
    setState({ status: "idle" });

    const requestId = scanRequestRef.current;
    const formData = new FormData();
    formData.append("image", file);
    try {
      setIsScanning(true);
      const result = await scanImageFaces(formData);
      if (!isMounted.current || scanRequestRef.current !== requestId) return;
      setScan(result);
      if (result.faces.length === 0) {
        setScanError("No faces were found in that photo. Try one where the faces are clearly visible.");
      } else if (result.faces.length === 1) {
        // Only one person to pick, so pick them.
        setReplacements({ [result.faces[0].face_index]: null });
      }
    } catch (error) {
      if (!isMounted.current || scanRequestRef.current !== requestId) return;
      setScanError(apiErrorMessage(error, "Failed to scan the photo for faces."));
    } finally {
      if (isMounted.current && scanRequestRef.current === requestId) setIsScanning(false);
    }
  };

  const toggleFace = (faceIndex: number) => {
    setScanError(null);
    if (faceIndex in replacements) {
      const entry = replacements[faceIndex];
      if (entry) URL.revokeObjectURL(entry.preview);
      const next = { ...replacements };
      delete next[faceIndex];
      setReplacements(next);
      return;
    }
    if (selectedIndices.length >= maxFaces) {
      setScanError(`You can swap up to ${maxFaces} faces at a time.`);
      return;
    }
    setReplacements({ ...replacements, [faceIndex]: null });
  };

  const handleReplacementUpload = (faceIndex: number, fileList: FileList | null) => {
    const file = fileList?.[0];
    if (!file) return;
    const previous = replacements[faceIndex];
    if (previous) URL.revokeObjectURL(previous.preview);
    setReplacements({ ...replacements, [faceIndex]: { file, preview: URL.createObjectURL(file) } });
  };

  const isRunning = state.status === "uploading" || state.status === "processing";
  const progressCurrent = state.job?.progress_current ?? 0;
  const progressTotal = state.job?.progress_total ?? 0;
  const progressPct = progressTotal > 0 ? Math.round((progressCurrent / progressTotal) * 100) : 0;

  return (
    <section id="studio" className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-6">
      <header className="flex flex-col gap-2">
        <h2 className="font-display text-4xl text-white">Try-On Studio</h2>
        <p className="max-w-3xl text-base text-white/60">
          Upload a photo and the app finds every face in it. Pick up to {maxFaces} of them, give each a replacement
          face, and generate — outfit, body and pose stay exactly as they are.
        </p>
      </header>

      <div className="flex flex-col gap-6 xl:flex-row">
        <aside className="flex w-full flex-col gap-4 xl:max-w-sm">
          <div className="space-y-4 rounded-3xl border border-white/10 bg-white/5 p-5">
            <div className="space-y-2 rounded-2xl border border-dashed border-white/10 bg-black/30 p-4">
              <p className="text-xs uppercase tracking-[0.3em] text-white/40">1 · Upload photo</p>
              {photoPreview ? (
                <div className="relative overflow-hidden rounded-3xl bg-[#0c0c18]">
                  <img src={photoPreview} alt="Uploaded photo" className="block h-auto w-full" />
                  {scan?.faces.map((face) => {
                    const [x1, y1, x2, y2] = face.bbox;
                    const isSelected = face.face_index in replacements;
                    return (
                      <button
                        key={face.face_index}
                        type="button"
                        onClick={() => toggleFace(face.face_index)}
                        disabled={isRunning}
                        aria-label={`Face ${face.face_index + 1}`}
                        aria-pressed={isSelected}
                        className={clsx(
                          "absolute rounded-lg border-2 transition disabled:cursor-not-allowed",
                          isSelected ? "border-brand-500 bg-brand-500/10" : "border-white/70 hover:border-white"
                        )}
                        style={{
                          left: `${(x1 / scan.image_width) * 100}%`,
                          top: `${(y1 / scan.image_height) * 100}%`,
                          width: `${((x2 - x1) / scan.image_width) * 100}%`,
                          height: `${((y2 - y1) / scan.image_height) * 100}%`
                        }}
                      >
                        <span
                          className={clsx(
                            "absolute -left-0.5 -top-0.5 rounded-br-md rounded-tl-md px-1.5 text-[10px] font-semibold",
                            isSelected ? "bg-brand-500 text-white" : "bg-white/80 text-black"
                          )}
                        >
                          {face.face_index + 1}
                        </span>
                      </button>
                    );
                  })}
                  <button
                    onClick={() => {
                      clearPhoto();
                      setState({ status: "idle" });
                    }}
                    disabled={isRunning}
                    className="absolute right-2 top-2 rounded-full bg-black/70 px-3 py-1 text-xs text-white/80 hover:bg-black disabled:opacity-60"
                  >
                    ✕
                  </button>
                  {photoName ? (
                    <span className="absolute bottom-3 right-3 max-w-[70%] truncate rounded-full bg-black/70 px-3 py-1 text-xs text-white/80">
                      {photoName}
                    </span>
                  ) : null}
                </div>
              ) : (
                <label
                  htmlFor={uploadInputId}
                  className="flex h-80 w-full cursor-pointer items-center justify-center rounded-3xl border border-dashed border-white/20 bg-[#0c0c18] text-white/60 transition hover:border-white/40 hover:bg-white/5"
                >
                  <span className="text-center text-sm uppercase tracking-[0.3em]">Upload photo to re-face</span>
                </label>
              )}
              <input
                id={uploadInputId}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(event) => {
                  handleUpload(event.target.files);
                  event.target.value = "";
                }}
              />
            </div>

            <div className="space-y-3 rounded-2xl border border-dashed border-white/10 bg-black/30 p-4">
              <p className="text-xs uppercase tracking-[0.3em] text-white/40">2 · Choose faces to swap</p>
              {isScanning ? (
                <p className="text-sm text-white/50">Scanning the photo for faces…</p>
              ) : !photoPreview ? (
                <p className="text-xs text-white/40">Upload a photo first — its faces will show up here.</p>
              ) : scan && scan.faces.length > 0 ? (
                <>
                  <p className="text-xs text-white/40">
                    Found {scan.faces.length} face{scan.faces.length === 1 ? "" : "s"}. Click the ones you want to
                    replace (up to {maxFaces}), then add a photo for each.
                  </p>
                  <div className="grid grid-cols-4 gap-2">
                    {scan.faces.map((face) => {
                      const isSelected = face.face_index in replacements;
                      return (
                        <button
                          key={face.face_index}
                          type="button"
                          onClick={() => toggleFace(face.face_index)}
                          disabled={isRunning}
                          aria-pressed={isSelected}
                          className={clsx(
                            "relative overflow-hidden rounded-2xl border-2 transition disabled:cursor-not-allowed",
                            isSelected ? "border-brand-500" : "border-white/10 opacity-70 hover:opacity-100"
                          )}
                        >
                          <img
                            src={`${BACKEND_BASE}${face.thumbnail_url}`}
                            alt={`Face ${face.face_index + 1} from the photo`}
                            className="aspect-square w-full object-cover"
                          />
                          <span className="absolute left-1 top-1 rounded-md bg-black/70 px-1.5 text-[10px] font-semibold text-white">
                            {face.face_index + 1}
                          </span>
                        </button>
                      );
                    })}
                  </div>

                  {selectedIndices.map((faceIndex) => {
                    const face = scan.faces.find((item) => item.face_index === faceIndex);
                    const entry = replacements[faceIndex];
                    const inputId = `faceswap-replacement-${faceIndex}`;
                    return (
                      <div
                        key={faceIndex}
                        className="flex items-center gap-3 rounded-2xl border border-white/10 bg-black/30 p-3"
                      >
                        {face ? (
                          <img
                            src={`${BACKEND_BASE}${face.thumbnail_url}`}
                            alt={`Face ${faceIndex + 1} from the photo`}
                            className="h-14 w-14 shrink-0 rounded-xl object-cover"
                          />
                        ) : null}
                        <span className="text-white/40">→</span>
                        {entry ? (
                          <img
                            src={entry.preview}
                            alt={`Replacement for face ${faceIndex + 1}`}
                            className="h-14 w-14 shrink-0 rounded-xl object-cover"
                          />
                        ) : (
                          <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl border border-dashed border-white/20 text-[10px] text-white/40">
                            No photo
                          </div>
                        )}
                        <label
                          htmlFor={inputId}
                          className={clsx(
                            "ml-auto cursor-pointer rounded-full border border-white/20 px-3 py-2 text-xs text-white/80 transition hover:bg-white/10",
                            isRunning && "pointer-events-none opacity-60"
                          )}
                        >
                          {entry ? "Change" : "Add face"}
                        </label>
                        <input
                          id={inputId}
                          type="file"
                          accept="image/*"
                          className="hidden"
                          disabled={isRunning}
                          onChange={(event) => {
                            handleReplacementUpload(faceIndex, event.target.files);
                            event.target.value = "";
                          }}
                        />
                      </div>
                    );
                  })}
                </>
              ) : null}
              {scanError ? <p className="text-sm text-red-400">{scanError}</p> : null}
            </div>

            <div className="rounded-2xl bg-lime-200/20 p-4 text-xs text-lime-200">
              <p className="font-medium text-lime-100">Note</p>
              <p className="mt-1 text-lime-100/80">
                Use clear, front-facing replacement photos. If a replacement photo shows several people, its largest
                face is used.
              </p>
            </div>

            <button
              onClick={() => mutation.mutate()}
              disabled={isRunning || isScanning || !canGenerate}
              className="w-full rounded-full bg-brand-500 py-3 text-sm font-semibold text-white shadow-neon transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:bg-brand-500/40"
            >
              {isRunning ? "Generating…" : "Generate Face Swap"}
            </button>
          </div>
        </aside>

        <div className="flex flex-1 flex-col gap-4">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
            <div className="flex items-center justify-between text-xs text-white/50">
              <span className="rounded-full bg-white/10 px-3 py-1 text-white/70">Result</span>
              <span>Status: {state.status.toUpperCase()}</span>
            </div>
            {isRunning ? (
              <div className="mt-3">
                <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                  <div
                    className={
                      progressTotal
                        ? "h-full bg-brand-500 transition-all"
                        : "h-full w-1/3 animate-pulse bg-brand-500/60"
                    }
                    style={progressTotal ? { width: `${progressPct}%` } : undefined}
                  />
                </div>
                <p className="mt-1 text-xs text-white/50">
                  {progressTotal ? `Face ${progressCurrent} / ${progressTotal} (${progressPct}%)` : "Generating…"}
                </p>
              </div>
            ) : null}
            <div className="relative mt-4 flex h-[600px] w-full items-center justify-center overflow-hidden rounded-3xl bg-black/30">
              <AnimatePresence>
                {state.previewUrl ? (
                  <motion.div
                    key={state.previewUrl}
                    initial={{ opacity: 0.4, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.98 }}
                    className="h-full w-full"
                  >
                    <img src={state.previewUrl} alt="Face swap result" className="h-full w-full object-contain" />
                  </motion.div>
                ) : (
                  <div className="flex flex-col items-center gap-2 text-sm text-white/50">
                    <span>Upload a photo, choose faces, then generate to see the result here.</span>
                  </div>
                )}
              </AnimatePresence>
            </div>
            {state.previewUrl ? (
              <a
                href={state.previewUrl}
                download
                target="_blank"
                rel="noreferrer"
                className="mt-4 inline-flex items-center justify-center rounded-full border border-white/20 px-5 py-2 text-sm font-semibold text-white transition hover:bg-white/10"
              >
                Open full-resolution PNG
              </a>
            ) : null}
            {state.error ? <p className="mt-3 text-sm text-red-400">Error: {state.error}</p> : null}
          </div>
        </div>
      </div>
    </section>
  );
}
