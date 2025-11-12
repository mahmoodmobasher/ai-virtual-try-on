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
   - macOS/Linux with Python 3.11+, Node.js 20+, CUDA-capable GPU (NVIDIA) or Apple Silicon.
   - Git LFS for large checkpoints.
2. **Bootstrap**
   ```bash
   cd /Users/moemahmood/Documents/AI/ai-virtual-try-on
   make setup
   ```
   The `Makefile` provisions Python/Node environments, installs model weights, and seeds a demo catalog.
3. **Run services (separate terminals)**
   ```bash
   make dev-backend  # starts FastAPI on :8000
   make dev-frontend # starts Next.js on :3000
   ```
   - Backend API & docs: http://localhost:8000/docs
   - Frontend experience: http://localhost:3000
   - Optional worker queue (RQ) can be added for batch rendering (see `docs/ARCHITECTURE.md`).

### Next Steps
- Configure environment variables in `backend/.env.example`.
- Add product imagery under `backend/storage/catalog/`.
- Connect LLM Studio endpoint through `LLM_STUDIO_BASE_URL`.

See `/docs/ARCHITECTURE.md` for a deep dive into data flows, pipelines, and deployment models.

