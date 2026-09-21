"use client";

import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";

import { BACKEND_BASE, createTryOn, getTryOnJob, TryOnJob } from "@/lib/api";

type TryOnMode = "garment_swap" | "face_swap";
type GarmentCategory = "auto" | "tops" | "bottoms" | "one-pieces";

const MODE_COPY: Record<
  TryOnMode,
  { label: string; description: string; personLabel: string; garmentLabel: string; note: string; missing: string }
> = {
  garment_swap: {
    label: "Clothes swap",
    description: "The garment is dressed onto your photo while your face, body and pose stay the same.",
    personLabel: "Upload your photo",
    garmentLabel: "Upload garment photo",
    note: "Use a clear, well-lit photo of yourself (half or full body). The garment can be a flat-lay, product shot or worn by another model.",
    missing: "Upload a photo of yourself and a photo of the garment."
  },
  face_swap: {
    label: "Face swap",
    description: "Your face is placed on the reference model while their outfit and pose are preserved.",
    personLabel: "Upload source face",
    garmentLabel: "Upload reference model wearing outfit",
    note: "Use two clear, front-facing photos. The result keeps the reference model's outfit and body while replacing their face with the source face.",
    missing: "Upload a source face photo and a reference photo of someone wearing the outfit."
  }
};

const CATEGORIES: { value: GarmentCategory; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "tops", label: "Top" },
  { value: "bottoms", label: "Bottom" },
  { value: "one-pieces", label: "Full outfit" }
];

type TryOnState = {
  job?: TryOnJob;
  status: "idle" | "uploading" | "processing" | "done" | "error";
  previewUrl?: string;
  error?: string;
};

export function TryOnStudio() {
  const [userImage, setUserImage] = useState<File | null>(null);
  const [portraitPreview, setPortraitPreview] = useState<string | null>(null);
  const [modelName, setModelName] = useState<string | null>(null);
  const [state, setState] = useState<TryOnState>({ status: "idle" });
  const [tryOnMode, setTryOnMode] = useState<TryOnMode>("garment_swap");
  const [category, setCategory] = useState<GarmentCategory>("auto");
  const copy = MODE_COPY[tryOnMode];
  const [activeCollection, setActiveCollection] = useState("My gallery (all)");
  const [activeEngine, setActiveEngine] = useState<"AI Engine 1" | "AI Engine 2">("AI Engine 1");
  const [swapSource, setSwapSource] = useState<"Text" | "Photo">("Photo");
  const [galleryTab, setGalleryTab] = useState<"My gallery" | "Upload" | "Info">("My gallery");
  const isMounted = useRef(true);
  const portraitObjectUrlRef = useRef<string | null>(null);
  const uploadInputId = "tryon-upload-model";
  const [garmentFile, setGarmentFile] = useState<File | null>(null);
  const [garmentPreview, setGarmentPreview] = useState<string | null>(null);
  const garmentObjectUrlRef = useRef<string | null>(null);
  const garmentInputId = "garment-upload";

  useEffect(() => {
    return () => {
      isMounted.current = false;
      if (portraitObjectUrlRef.current) {
        URL.revokeObjectURL(portraitObjectUrlRef.current);
      }
      if (garmentObjectUrlRef.current) {
        URL.revokeObjectURL(garmentObjectUrlRef.current);
      }
    };
  }, []);

  // Catalog feature removed - users upload their own garments

  const mutation = useMutation({
    mutationFn: async () => {
      console.log("[mutation] Starting try-on generation...");
      console.log("[mutation] userImage:", userImage?.name, "garmentFile:", garmentFile?.name);
      if (!garmentFile || !userImage) {
        throw new Error(copy.missing);
      }
      const formData = new FormData();
      console.log("[mutation] Appending garment_image:", garmentFile.name);
      formData.append("garment_image", garmentFile);
      formData.append("user_image", userImage);
      formData.append("mode", tryOnMode);
      if (tryOnMode === "garment_swap") {
        formData.append("category", category);
      }
      console.log("[mutation] Try-on mode:", tryOnMode, "category:", category);
      console.log("[mutation] Calling createTryOn API...");
      return createTryOn(formData);
    },
    onMutate: () => {
      if (!isMounted.current) return;
      console.log("[mutation] onMutate - setting status to uploading");
      setState((prev) => ({ ...prev, status: "uploading" }));
    },
    onError: (error: Error) => {
      if (!isMounted.current) return;
      console.error("[mutation] onError:", error);
      setState((prev) => ({ ...prev, status: "error", error: error.message }));
    },
    onSuccess: async (job) => {
      console.log("[mutation] onSuccess - received job:", job);
      console.log("[mutation] isMounted.current:", isMounted.current);
      
      // Check if job is already completed (synchronous processing)
      if (job.status === "completed") {
        const url = job.result_image_path ? `${BACKEND_BASE}${job.result_image_path}` : undefined;
        console.log("[mutation] Job already completed! Setting preview URL to:", url);
        setState({ status: "done", job, previewUrl: url });
      } else if (job.status === "failed") {
        console.error("[mutation] Job already failed:", job.failure_reason);
        setState({ status: "error", error: job.failure_reason ?? "Processing failed", job });
      } else {
        // Job still processing, start polling
        console.log("[mutation] Job processing, starting poll...");
        setState((prev) => ({ ...prev, status: "processing", job }));
        if (isMounted.current) {
          pollJob(job.id);
        }
      }
    }
  });

  const pollJob = async (jobId: number) => {
    try {
      console.log(`[pollJob] Polling job ${jobId}...`);
      const job = await getTryOnJob(jobId);
      console.log(`[pollJob] Received job:`, job);
      if (!isMounted.current) return;
      if (job.status === "completed") {
        const url = job.result_image_path ? `${BACKEND_BASE}${job.result_image_path}` : undefined;
        console.log(`[pollJob] Job completed! Setting preview URL to:`, url);
        setState({ status: "done", job, previewUrl: url });
        console.log(`[pollJob] State updated with preview URL`);
      } else if (job.status === "failed") {
        console.error(`[pollJob] Job failed:`, job.failure_reason);
        setState({ status: "error", error: job.failure_reason ?? "Processing failed", job });
      } else {
        console.log(`[pollJob] Job status: ${job.status}, polling again in 1.5s...`);
        setTimeout(() => {
          if (isMounted.current) pollJob(jobId);
        }, 1500);
      }
    } catch (error) {
      console.error(`[pollJob] Error polling job:`, error);
      setTimeout(() => {
        if (isMounted.current) pollJob(jobId);
      }, 1500);
    }
  };

  const handleUpload = (fileList: FileList | null) => {
    const file = fileList?.[0];
    if (!file) return;
    
    console.log("File selected:", file.name, file.type, file.size);
    
    setUserImage(file);
    setModelName(file.name);
    if (portraitObjectUrlRef.current) {
      URL.revokeObjectURL(portraitObjectUrlRef.current);
    }
    const objectUrl = URL.createObjectURL(file);
    portraitObjectUrlRef.current = objectUrl;
    
    console.log("Created blob URL:", objectUrl);
    
    setPortraitPreview(objectUrl);
    // Do NOT set previewUrl here - keep Virtual try-on empty until Generate is clicked
    setState({
      status: "idle",
      job: undefined,
      previewUrl: undefined,
      error: undefined
    });
    
    console.log("Portrait preview updated, Virtual try-on cleared");
  };

  const handleGarmentUpload = (fileList: FileList | null) => {
    const file = fileList?.[0];
    if (!file) return;
    
    console.log("Garment file selected:", file.name, file.type, file.size);
    
    setGarmentFile(file);
    if (garmentObjectUrlRef.current) {
      URL.revokeObjectURL(garmentObjectUrlRef.current);
    }
    const objectUrl = URL.createObjectURL(file);
    garmentObjectUrlRef.current = objectUrl;
    
    console.log("Created garment blob URL:", objectUrl);
    
    setGarmentPreview(objectUrl);
    
    console.log("Garment preview updated");
  };


  const collections = ["My gallery (all)", "Campaign launch", "Holiday assets"];

  const garmentImageUrl = garmentPreview;

  return (
    <section id="studio" className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-6">
      <header className="flex flex-col gap-2">
        <h2 className="font-display text-4xl text-white">Try-On Studio</h2>
        <p className="max-w-3xl text-base text-white/60">
          Recreate the full virtual fitting-room workflow. Upload a model, select garments, experiment with engines, and export the
          outputs that move your campaign forward.
        </p>
      </header>

      <div className="flex flex-col gap-6 xl:flex-row">
        <aside className="flex w-full flex-col gap-4 xl:max-w-sm">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-white/50">Collection</p>
            <p className="mt-2 text-sm text-white/70">My gallery ({collections.length})</p>
            <div className="mt-4 flex flex-col gap-2">
              {collections.map((collection) => (
                <button
                  key={collection}
                  onClick={() => setActiveCollection(collection)}
                  className={clsx(
                    "rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-left text-sm text-white/70 transition hover:border-brand-500/60",
                    activeCollection === collection && "border-brand-500 bg-brand-500/10 text-white"
                  )}
                >
                  {collection}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-3xl border border-brand-500/40 bg-brand-500/10 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-white/50">Mode</p>
            <div className="mt-3 flex gap-2 rounded-full bg-black/30 p-1">
              {(Object.keys(MODE_COPY) as TryOnMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => {
                    setTryOnMode(mode);
                    setState({ status: "idle" });
                  }}
                  className={clsx(
                    "w-1/2 rounded-full px-4 py-2 text-sm transition",
                    tryOnMode === mode ? "bg-white text-black shadow-neon" : "text-white/70"
                  )}
                >
                  {MODE_COPY[mode].label}
                </button>
              ))}
            </div>
            <p className="mt-3 text-xs text-white/60">{copy.description}</p>
            {tryOnMode === "garment_swap" ? (
              <div className="mt-4">
                <p className="text-xs uppercase tracking-[0.3em] text-white/50">Garment type</p>
                <div className="mt-2 grid grid-cols-4 gap-1 rounded-2xl bg-black/30 p-1">
                  {CATEGORIES.map((option) => (
                    <button
                      key={option.value}
                      onClick={() => setCategory(option.value)}
                      className={clsx(
                        "rounded-xl px-2 py-2 text-xs transition",
                        category === option.value ? "bg-white text-black" : "text-white/70"
                      )}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          <div className="space-y-4 rounded-3xl border border-white/10 bg-white/5 p-5">
            <div className="flex items-center justify-between">
              <p className="text-xs uppercase tracking-[0.3em] text-white/50">Upload model and clothes</p>
              <button className="rounded-full border border-white/10 px-3 py-1 text-xs text-white/60 hover:border-white/30">help</button>
            </div>

            <div className="flex gap-2 rounded-full bg-black/30 p-1">
              {(["AI Engine 1", "AI Engine 2"] as const).map((engine) => (
                <button
                  key={engine}
                  onClick={() => setActiveEngine(engine)}
                  className={clsx(
                    "w-1/2 rounded-full px-4 py-2 text-sm transition",
                    activeEngine === engine ? "bg-white text-black shadow-neon" : "text-white/70"
                  )}
                >
                  {engine}
                </button>
              ))}
            </div>

            <div className="space-y-2 rounded-2xl border border-dashed border-white/10 bg-black/30 p-4">
              <p className="text-xs uppercase tracking-[0.3em] text-white/40">
                {copy.personLabel}
              </p>
              <div className="relative flex h-80 w-full overflow-hidden rounded-3xl bg-[#0c0c18]">
                {portraitPreview ? (
                  <>
                    <img 
                      src={portraitPreview} 
                      alt="Uploaded model" 
                      className="h-full w-full object-contain"
                      onError={(e) => {
                        console.error("Failed to load image:", portraitPreview);
                      }}
                      onLoad={() => {
                        console.log("Image loaded successfully:", portraitPreview);
                      }}
                    />
                    <button
                      onClick={() => {
                        setPortraitPreview(null);
                        setUserImage(null);
                        setModelName(null);
                        if (portraitObjectUrlRef.current) {
                          URL.revokeObjectURL(portraitObjectUrlRef.current);
                          portraitObjectUrlRef.current = null;
                        }
                      }}
                      className="absolute right-2 top-2 rounded-full bg-black/70 px-3 py-1 text-xs text-white/80 hover:bg-black"
                    >
                      ✕
                    </button>
                    {modelName ? (
                      <span className="absolute bottom-3 right-3 rounded-full bg-black/70 px-3 py-1 text-xs text-white/80">
                        {modelName}
                      </span>
                    ) : null}
                  </>
                ) : (
                  <label
                    htmlFor={uploadInputId}
                    className="flex h-full w-full cursor-pointer items-center justify-center border border-dashed border-white/20 rounded-3xl text-white/60 transition hover:border-white/40 hover:bg-white/5"
                  >
                    <span className="text-sm uppercase tracking-[0.3em]">
                      {copy.personLabel}
                    </span>
                  </label>
                )}
              </div>
              <input
                id={uploadInputId}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(event) => {
                  console.log("Input onChange triggered", event.target.files);
                  handleUpload(event.target.files);
                  event.target.value = "";
                }}
              />
              <div className="flex justify-between text-xs text-white/40">
                <button className="hover:text-white/70">Model templates</button>
                <button className="hover:text-white/70">Create a model</button>
              </div>
              {modelName ? <p className="text-xs text-white/40">Selected: {modelName}</p> : null}
            </div>

            <div className="flex items-center justify-between rounded-full bg-black/30 p-1">
              {(["Text", "Photo"] as const).map((option) => (
                <button
                  key={option}
                  onClick={() => setSwapSource(option)}
                  className={clsx(
                    "w-1/2 rounded-full px-4 py-2 text-sm transition",
                    swapSource === option ? "bg-white text-black shadow-neon" : "text-white/70"
                  )}
                >
                  {option}
                </button>
              ))}
            </div>

            <div className="space-y-2 rounded-2xl border border-dashed border-white/10 bg-black/30 p-4">
              <p className="text-xs uppercase tracking-[0.3em] text-white/40">
                {copy.garmentLabel}
              </p>
              <div className="relative flex h-80 w-full overflow-hidden rounded-3xl bg-[#0c0c18]">
                {garmentImageUrl ? (
                  <>
                    <img 
                      src={garmentImageUrl} 
                      alt="Garment" 
                      className="h-full w-full object-contain"
                      onError={(e) => {
                        console.error("Failed to load garment image:", garmentImageUrl);
                      }}
                      onLoad={() => {
                        console.log("Garment image loaded successfully:", garmentImageUrl);
                      }}
                    />
                    <button
                      onClick={() => {
                        setGarmentPreview(null);
                        setGarmentFile(null);
                        if (garmentObjectUrlRef.current) {
                          URL.revokeObjectURL(garmentObjectUrlRef.current);
                          garmentObjectUrlRef.current = null;
                        }
                      }}
                      className="absolute right-2 top-2 rounded-full bg-black/70 px-3 py-1 text-xs text-white/80 hover:bg-black"
                    >
                      ✕
                    </button>
                  </>
                ) : (
                  <label
                    htmlFor={garmentInputId}
                    className="flex h-full w-full cursor-pointer items-center justify-center border border-dashed border-white/20 rounded-3xl text-white/60 transition hover:border-white/40 hover:bg-white/5"
                  >
                    <span className="text-center text-sm uppercase tracking-[0.3em]">
                      {copy.garmentLabel}
                    </span>
                  </label>
                )}
              </div>
              <input
                id={garmentInputId}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(event) => {
                  console.log("Garment input onChange triggered", event.target.files);
                  handleGarmentUpload(event.target.files);
                  event.target.value = "";
                }}
              />
              <div className="flex justify-between text-xs text-white/40">
                <button className="hover:text-white/70">Item templates</button>
                <button className="hover:text-white/70">Create item</button>
              </div>
              {garmentFile ? <p className="text-xs text-white/40">Selected: {garmentFile.name}</p> : null}
            </div>

            <div className="rounded-2xl bg-lime-200/20 p-4 text-xs text-lime-200">
              <p className="font-medium text-lime-100">Note</p>
              <p className="mt-1 text-lime-100/80">
                {copy.note}
              </p>
            </div>

            <button
              onClick={() => {
                console.log("[Button] Generate Try-On clicked!");
                console.log("[Button] userImage:", userImage?.name, "garmentFile:", garmentFile?.name);
                console.log("[Button] Button disabled?", mutation.isPending || !userImage || !garmentFile);
                mutation.mutate();
              }}
              disabled={mutation.isPending || !userImage || !garmentFile}
              className="w-full rounded-full bg-brand-500 py-3 text-sm font-semibold text-white shadow-neon transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:bg-brand-500/40"
            >
              {mutation.isPending ? "Generating…" : "Generate Try-On"}
            </button>
          </div>
        </aside>

        <div className="flex flex-1 flex-col gap-4">
          <div className="flex items-center gap-2">
            {(["My gallery", "Upload", "Info"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setGalleryTab(tab)}
                className={clsx(
                  "rounded-full border border-white/10 px-4 py-2 text-sm text-white/60 transition hover:border-white/30",
                  galleryTab === tab && "border-white text-white"
                )}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
            <div className="flex items-center justify-between text-xs text-white/50">
              <span className="rounded-full bg-white/10 px-3 py-1 text-white/70">Virtual try on</span>
              <span>{activeEngine}</span>
            </div>
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
                    <img src={state.previewUrl} alt="Try-on preview" className="h-full w-full object-contain" />
                  </motion.div>
                ) : (
                  <div className="flex flex-col items-center gap-2 text-sm text-white/50">
                    <span>Upload a model and select a garment to begin.</span>
                  </div>
                )}
              </AnimatePresence>
            </div>
            <div className="mt-4 flex items-center justify-between">
              <div className="flex gap-2">
                <button
                  onClick={() => setActiveEngine("AI Engine 1")}
                  className={clsx(
                    "rounded-full border border-white/10 px-4 py-2 text-sm transition hover:border-white/30",
                    activeEngine === "AI Engine 1" && "border-brand-500 bg-brand-500/10 text-white"
                  )}
                >
                  Use
                </button>
                <button
                  onClick={() => setActiveEngine("AI Engine 2")}
                  className={clsx(
                    "rounded-full border border-white/10 px-4 py-2 text-sm transition hover:border-white/30",
                    activeEngine === "AI Engine 2" && "border-brand-500 bg-brand-500/10 text-white"
                  )}
                >
                  Use 2
                </button>
              </div>
              <div className="text-right text-xs text-white/50">
                <div>Status: {state.status.toUpperCase()}</div>
                {state.error ? <div className="text-red-400">Error: {state.error}</div> : null}
              </div>
            </div>
          </div>

          {/* Look Library hidden - user uploads their own garments */}
        </div>
      </div>
    </section>
  );
}
