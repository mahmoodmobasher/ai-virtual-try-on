## AI Virtual Try-On Platform

This project scaffolds a local-first virtual try-on experience inspired by The New Black AI. It mirrors the reference product's flows—marketing site, catalog browsing, interactive garment carousel, moodboard, and AI-assisted styling—while remaining fully self-hostable and integrating with `gpt-oss-20b` via LLM Studio.

### Core Modules
- **Frontend (`frontend/`)** – Next.js SPA recreating the landing page hero, showcase sections, carousel, try-on studio, lookbook, and CTA funnels.
- **Backend (`backend/`)** – FastAPI service orchestrating image ingestion, garment catalog management, virtual try-on inference, moodboard persistence, analytics, and LLM-backed concierge.
- **Inference (`backend/app/services/`)** – Pluggable computer vision pipelines built around open-source projects (`TryOnDiffusion`, `VITON-HD`, `ControlNet`) with fallbacks for classical segmentation. GPU-accelerated by default.
- **Assistant Integration** – Thin client for chatting with the locally served `gpt-oss-20b` to power styling suggestions, product copy, moodboard narration, and onboarding flows.

### Feature Parity Highlights
- Marketing hero with animated background, highlight cards, and C-suite CTA.
- Client logos marquee, testimonials, KPI stats, and feature deep dives.
- Try-On Studio with:
  - User upload, auto background removal, pose normalization.
  - Garment carousel, filters, search, outfit pinboard, and multi-view preview.
  - Batch rendering, downloadable contact sheet, shareable link.
- Moodboard builder and campaign planner with AI copy assist.
- Admin portal for catalog maintenance, analytics, and cohort exports.

### Getting Started
1. **System prerequisites**
   - macOS/Linux with Python 3.14 for the core application, Node.js 20+.
   - Python 3.11 and a CUDA-capable GPU or Apple Silicon for the optional ML stack.
   - Git LFS for large checkpoints.
2. **Bootstrap**
   ```bash
   cd /Users/moemahmood/builder_code/ai-virtual-try-on
   make setup
   ```
   The `Makefile` provisions Python/Node environments, installs model weights, and seeds a demo catalog.
3. **Run both services**
   ```bash
   ./run-app.sh
   ```

   Or run the services in separate terminals:
   ```bash
   make dev-backend  # starts FastAPI on :8000
   make dev-frontend # starts Next.js on :3000
   ```
   - Backend API & docs: http://localhost:8000/docs
   - Frontend experience: http://localhost:3000
   - Optional worker queue (RQ) can be added for batch rendering (see `docs/ARCHITECTURE.md`).

4. **Clothes swap (default try-on mode)**
   Clothes swap uses the hosted [FASHN](https://fashn.ai) virtual try-on API, so no GPU is needed.
   Create an API key at https://app.fashn.ai/api and add it to `backend/.env`:
   ```bash
   TRYON_FASHN_API_KEY=fa-...
   TRYON_FASHN_MODEL=tryon-v1.6   # or tryon-max for the newest (preview) model
   TRYON_FASHN_MODE=quality
   ```
   In the Try-On Studio, pick **Clothes swap**, upload your photo and a garment photo, and choose
   the garment type (auto, top, bottom, full outfit).

5. **Optional ML features**
   Install the model-backed try-on dependencies with Python 3.11 until PyTorch
   provides macOS Python 3.14 wheels:
   ```bash
   make setup-ml ML_PYTHON=python3.11
   ```

   When `backend/.venv-ml` exists, `run-app.sh` automatically uses it for the
   backend so face-swap features are available.

### Next Steps
- Configure environment variables in `backend/.env.example`.
- Add product imagery under `backend/storage/catalog/`.
- Connect LLM Studio endpoint through `LLM_STUDIO_BASE_URL`.

See `/docs/ARCHITECTURE.md` for a deep dive into data flows, pipelines, and deployment models.
