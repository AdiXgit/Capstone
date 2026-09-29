# Phone demo: Vercel frontend + tunneled backend

**Why not "just Vercel":** Vercel hosts static sites and short serverless functions.
Our backend is a persistent Python gRPC mesh plus a PyTorch/YOLO model — that can't
run on Vercel. So the split is:

- **Frontend → Vercel** (static Vite build, HTTPS, great on mobile)
- **Backend → your Mac**, exposed to the internet with a **Cloudflare quick tunnel** (HTTPS, no account)

The phone opens the Vercel URL and is told which backend to use via `?api=`.

```
 phone ──HTTPS──> Vercel (frontend) ──HTTPS fetch──> trycloudflare URL ──> localhost:8000 gateway ──> agent mesh + YOLO
```

## 1. Backend on your Mac
```bash
# start the mesh (orchestrator + agents + gateway on :8000)
bash scripts/run_local.sh

# in a second terminal: expose the gateway publicly (no login for quick tunnels)
brew install cloudflared
cloudflared tunnel --url http://localhost:8000
# → prints something like  https://red-fox-1234.trycloudflare.com
```
The gateway already sends `Access-Control-Allow-Origin: *`, so the Vercel page can call it.

## 2. Frontend on Vercel
From the repo:
```bash
cd frontend
npx vercel            # first run: log in, link/create a project
npx vercel --prod     # deploy → prints https://<your-app>.vercel.app
```
Vercel auto-detects Vite (`frontend/vercel.json` pins framework + SPA rewrites).
If you import via the Vercel dashboard instead, set **Root Directory = `frontend`**.

## 3. Open on your phone
```
https://<your-app>.vercel.app/?api=https://red-fox-1234.trycloudflare.com
```
The `?api=` value is saved in the browser, so later visits to the bare Vercel URL
keep using that backend until you pass a new one. (Quick-tunnel URLs change each run —
just append the new `?api=` when that happens.)

- Camera scanning needs HTTPS — Vercel provides it, so the **Scan plant** button works on the phone.

## Alternative: one URL, no Vercel
If you'd rather not manage two URLs, serve the built frontend from the gateway and
tunnel just that one origin (same-origin, no CORS, no `?api=`). Ask and I'll mount
`frontend/dist` on the FastAPI gateway.
