## System Architecture

### Overview
The platform is composed of modular services connected via REST and event queues. All critical workloads run locally to respect privacy and enable offline experimentation. The diagram below summarizes the data flow:

```
Frontend (Next.js) ──> API Gateway (FastAPI) ──> Orchestrator
      ↑                         │                   │
      └────── WebSockets <──────┘                   │
                                                   │
                                      ┌────────────┴────────────┐
                                      │  Try-On Inference Worker │
                                      │  (PyTorch + Diffusers)   │
                                      └────────────┬────────────┘
                                                   │
                                 ┌─────────────────┴─────────────────┐
                                 │                                   │
                       Garment Store (Postgres + MinIO)     User Asset Cache
```

### Components

1. **Frontend**
   - Next.js 14 with App Router, Tailwind, Framer Motion.
   - React Query for data fetching, Zustand for lightweight state (carousel, moodboard).
   - Uses WebSockets for live job status and SSE for analytics feed.

2. **Backend API**
   - FastAPI, Pydantic v2, SQLModel for ORM.
   - Auth0-compatible JWT guard plus API keys for partner integrations.
   - Upload endpoints store originals in `storage/uploads/`, schedule inference jobs, and respond with job IDs.

3. **Inference Service**
   - Loads models lazily based on configuration. Default stack:
     - **Segmentation**: `birefnet` or `robust-video-matting`.
     - **Pose**: `openpifpaf` for 2D keypoints, optional `SMPL` for 3D.
     - **Virtual Try-On**: `tryondiffusion` pipeline via Diffusers, fallback `VITON-HD`.
   - Queue worker consumes tasks from Redis, streams partial progress (steps) to clients.

4. **Catalog & Moodboard Store**
   - PostgreSQL for metadata (`garments`, `collections`, `moodboards`, `analytics_events`).
   - MinIO/S3-compatible bucket for processed assets.

5. **LLM Studio Bridge**
   - Lightweight client hitting `LLM_STUDIO_BASE_URL/v1/chat/completions`.
   - Prompts for styling advice, moodboard summaries, and marketing copy.

### Pipelines

1. **Try-On Pipeline**
   1. Receive user portrait + garment product photo.
   2. Run segmentation to isolate person and garment masks.
   3. Estimate pose & body shape.
   4. Warp garment via TPS (thin-plate spline) guided by ControlNet.
   5. Compose output, upscale, and blend lighting.

2. **Catalog Import**
   - CLI command `python -m app.scripts.import_catalog` ingests CSVs, downloads imagery, generates embeddings.

3. **Analytics**
   - Event ingestion via `/events` endpoint. Stored for dashboards and to train conversion models.

### Deployment Modes

- **Local Dev**: Docker Compose with GPU passthrough. Hot reload for API & frontend.
- **Edge Appliance**: Systemd services with pre-fetched checkpoints and Air-gapped updates.
- **Cloud Hybrid**: Optional connectors to push aggregated metrics to cloud BI (disabled by default).

### Security & Privacy

- Images never leave the local machine; encryption at rest via filesystem-level (FileVault).
- JWT-based auth for multi-user setups. Env-var flag `SELF_HOSTED_SINGLE_USER` to disable.
- Audit trail stored in `analytics_events`.

### Extensibility

- Model registry in `app/core/models.yaml` allows mixing different pipelines.
- Plugin interface for apparel verticals (footwear, accessories) via `GarmentAdapter` class.
- Event hooks (`on_tryon_completed`, `on_moodboard_shared`) for downstream automations.


### Experimental Video Lab

- The `/api/video/clip` endpoint wraps Stability AI's Stable Video Diffusion image-to-video pipeline.   
- Results are stored in `storage/video-results/` and are served to the frontend through `/video-results/<file>`.   
- Generation requires a GPU with at least ~14 GB VRAM; CPU execution is supported but significantly slower (minutes per clip).   
- The frontend exposes this workflow at `/video-lab`, gated as an experimental surface separate from the primary try-on studio.
