# Deploying the demo

> **Status:** the demo is not currently hosted; it runs locally (see the Quick start in the
> [README](README.md)). This guide documents how it can be deployed when needed.

The app is one Docker image: a FastAPI server with the Gradio demo page mounted on it,
serving both on port 7860. The embedding model, reranker and search index are built into
the image, so it starts without downloading anything. The only secret it needs is
`ANTHROPIC_API_KEY`, supplied at run time and never stored in the image or the repository.

Checked against the Hugging Face Spaces documentation on 2 October 2026.

## Before you start: protect your API budget

The demo calls the Anthropic API (Claude Haiku 4.5, about $0.003 per question). It has two
built-in limits:

| Setting | Default | What it does |
| --- | --- | --- |
| `RATE_LIMIT_PER_MINUTE` | 10 | Questions per visitor (IP address) per minute |
| `DAILY_LLM_CAP` | 200 | Questions answered per UTC day, across all visitors |

The daily cap is kept in memory, so **it resets whenever the container restarts** (for
example after the Space wakes from sleep). Treat it as a guard against bursts, not as a
budget. Set the real ceiling in the Anthropic Console: **Settings → Limits → monthly spend
limit** (for example $5). Use a separate API key for the demo so you can revoke it without
affecting anything else.

## Option A: Hugging Face Docker Space (recommended if you have PRO)

Docker Spaces need a **Hugging Face PRO** subscription ($9/month). The free "CPU basic"
hardware (2 vCPU, 16 GB RAM) is enough; the app uses about 2 GB.

### 1. Create the Space

1. Sign in at https://huggingface.co and subscribe to PRO (Settings → Billing).
2. Go to https://huggingface.co/new-space.
3. Name it, e.g. `uk-renting-rag`. Licence: MIT. **SDK: Docker**, template: **Blank**.
   Hardware: **CPU basic (free)**. Visibility: **Public**.
4. Click **Create Space**.

### 2. Add the API key as a secret

1. In the Space, open **Settings → Variables and secrets → New secret**.
2. Name: `ANTHROPIC_API_KEY`. Value: your key (copy it from the Anthropic Console).
3. Optional: add **variables** (not secrets) `DAILY_LLM_CAP` and `RATE_LIMIT_PER_MINUTE` to
   change the limits.

Secrets are injected as environment variables when the container runs. They are not
visible in the Space's files, logs or settings page once saved.

### 3. Push to the Space

From the project folder, with the virtual environment active:

```bash
python scripts/prepare_space.py          # assembles build/space/ with the Space README header
```

Then push it (the first time, clone the empty Space repo next to the project):

```bash
cd ..
git clone https://huggingface.co/spaces/<your-hf-username>/uk-renting-rag hf-space
cp -r UKRAG/build/space/. hf-space/      # Windows PowerShell: Copy-Item -Recurse -Force UKRAG\build\space\* hf-space\
cd hf-space
git add -A
git commit -m "Deploy uk-renting-rag v1.0.0"
git push
```

When git asks for a password, use a Hugging Face **access token** with write permission
(Settings → Access Tokens), not your account password. Do not paste the token into any file.

### 4. Wait for the build, then check it

The first build downloads PyTorch and the models and builds the index: expect 10 to 20
minutes. Watch progress in the Space's **Logs** tab (Build, then Container).

When it shows **Running**:

- Open the Space page and try an example question.
- Check `https://<your-hf-username>-uk-renting-rag.hf.space/health`: `"status": "ok"` means
  the index, models and API key are all in place. `"degraded"` with `"llm_configured": false`
  means the secret is missing or misnamed.
- The API is at `.../ask` (POST `{"question": "..."}`) and its docs at `.../docs`.

### 5. Add the link to the README

Replace the demo link placeholder at the top of `README.md` with your Space URL, commit and
push to GitHub.

### Updating the demo

Make and commit your changes in this repository, then run `python scripts/prepare_space.py`,
copy `build/space/` into `hf-space/` again, commit and push. The Space rebuilds itself.

## Option B: any Docker host

The same image runs anywhere that can run Docker with about 2 GB of memory:

```bash
docker build -t uk-renting-rag .
docker run -p 7860:7860 --env-file .env uk-renting-rag
```

Then open http://localhost:7860. The port is configurable with `-e PORT=8080 -p 8080:8080`.
CI builds this image on every push and checks that it starts and loads its index with no
network access.

## Option C: a free Gradio Space (not set up)

Free Hugging Face accounts (older than 30 days, with a verified email) can host up to two
**Gradio SDK** Spaces on ZeroGPU hardware, but not Docker Spaces. This project has not been
adapted for that, because it cannot be tested outside Hugging Face. It would need:

- a `README.md` header with `sdk: gradio`, `python_version: "3.12"` and an `app_file` that
  starts the app (the Space runs that file instead of the Dockerfile);
- PyTorch pinned to a version ZeroGPU supports (2.8 to 2.13 at the time of writing; this
  project uses 2.14);
- the index files committed to the Space (they are small) and the models listed under
  `preload_from_hub` in the header, so the app does not download them on every start;
- possibly at least one function decorated with `@spaces.GPU`, which ZeroGPU expects.

## Things visitors should know

- **Cold starts.** Free and basic Spaces go to sleep when unused; the first request after
  that waits while the container restarts and loads the models (typically under a minute).
  The demo page says so.
- **Privacy.** The app logs only anonymous metrics (outcome, timings, whether it refused),
  never the question text or IP addresses, and runs uvicorn without its access log.
  Questions are sent to Anthropic's API to generate answers. The demo page says this too.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Build fails while installing PyTorch | Temporary network problem: restart the build from the Space settings |
| `/health` shows `"index_loaded": false` | The build step `prepare_deployment.py` failed: check the Build logs |
| "The demo's answer service isn't configured" | `ANTHROPIC_API_KEY` secret missing or misnamed |
| "This demo has reached its daily limit" | `DAILY_LLM_CAP` reached; it resets at midnight UTC or on restart |
| "The answer service is unavailable" | Anthropic API error or spend limit reached: check the Console |
