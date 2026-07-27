# Publishing model-risk-lab to GitHub — step-by-step

Total time: about 5 minutes. The git repository is already initialized and
committed inside this folder — you only create the GitHub repo and push.

## 1. Create the empty repository on GitHub

1. Log in as **Arijitray2** and go to https://github.com/new
2. Repository name: **model-risk-lab** (exactly this — the website links depend on it)
3. Leave it **Public**. Do **NOT** tick "Add a README", .gitignore, or license
   (the project already has them).
4. Click **Create repository**.

## 2. Push the project

Open a terminal in this folder (the one containing this file) and run:

```bash
git remote add origin https://github.com/Arijitray2/model-risk-lab.git
git branch -M main
git push -u origin main
```

Git will ask you to authenticate (browser login or a personal access token —
GitHub walks you through it).

## 3. Turn on the website (GitHub Pages)

1. On the repo page: **Settings → Pages** (left sidebar).
2. Under **Build and deployment → Source**: choose **Deploy from a branch**.
3. Branch: **main**, folder: **/docs**. Click **Save**.
4. Wait 1–2 minutes, refresh the page — it shows your URL:
   **https://arijitray2.github.io/model-risk-lab/**

That's it. The site, the interactive demo, all figures, all data and the PDF
report are now live.

## 4. (Optional) sanity checks

- Repo page shows the README with the project description.
- The website's *Live demo* page animates when you click a preset.
- The *Results* page shows tables (they load from JSON — they need the site to be
  served by Pages, which it is).

## If something looks wrong

- **404 on the site**: Pages can take a couple of minutes on first deploy;
  also confirm the folder is `/docs` on branch `main`.
- **Tables empty on Results page**: hard-refresh (Ctrl+Shift+R) — the JSON may
  have been cached mid-deploy.
- **Want to update numbers later**: edit/re-run `scripts/run_experiments.py` and
  `scripts/make_figures.py`, copy `results/*.json` to `docs/assets/` and
  `results/figures/*.svg` to `docs/assets/figures/`, commit, push — Pages
  redeploys automatically.
