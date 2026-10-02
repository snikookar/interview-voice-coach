# Client (Next.js)

The browser UI for Interview Voice Coach. See the [root README](../README.md) for the full setup.

```bash
npm install
npm run dev          # http://localhost:3000 (expects the API on :7860; override with API_URL)
```

| Route | What it does |
|---|---|
| `/` | Start form (role, level, mode, job posting), progress charts, history |
| `/interview?session=<id>` | Live voice interview over WebRTC: transcript, timer, question progress, per-turn latency |
| `/report/<id>` | Scores, covered/missed key points, delivery metrics, latency breakdown, recording, transcript |

`/api/*` is proxied to the FastAPI server by `next.config.ts`, so the browser only ever talks to one origin.
