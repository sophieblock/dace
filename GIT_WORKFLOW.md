# Git Workflow for `dace` Fork (with `local-main`)

This document explains the Git setup and day-to-day workflow for working on your fork of `spcl/dace` without getting into confusing states, **including** the special `local-main` branch you use for notebooks, debugging, and untracked files.

The goals:

1. **`upstream/main` is the source of truth.**
   You never push there; you only fetch from it.
2. **`origin/main` (your fork) mirrors `upstream/main`.**
   Your fork stays clean and up to date with the original repo.
3. **`main` stays clean.**
   You don't do messy work on `main`; it tracks `origin/main` and acts as the base for everything else.
4. **`local-main` is your personal scratch branch.**
   You run notebooks, add untracked debug files, and generally mess around here, while still being able to pull in new changes from `origin/main`.
5. **Feature branches are for shareable work.**
   When you want a PR or something more polished, you branch off `main`, not `local-main`.
6. **If things get messy, you can always reset safely.**
   You keep backup branches before doing anything destructive.

---

## 1. Remote Setup

In this repo you have three remotes configured:

- `origin` – your fork on GitHub: `https://github.com/sophieblock/dace.git`
- `upstream` – the original repo: `https://github.com/spcl/dace`
- `spcl` – an additional alias pointing to the original repo: `https://github.com/spcl/dace.git`

Safety guard:

- `upstream` has **no valid push URL**, so you can't accidentally push:

  ```bash
  git remote set-url --push upstream no_push
  ```

You only ever push to `origin`.

You can confirm remotes at any time with:

```bash
cd /Users/sophieblock/dev_packages/dace

git remote -v
```

---

## 2. Branches

### `main` (clean, mirrors origin/main and upstream/main)

- Local branch that tracks `origin/main`.
- Always kept in sync with `upstream/main`.
- **No messy work here.** You don't run random notebooks or keep dirty changes on `main`.
- Used only as a clean base to:
  - Update from upstream.
  - Create `local-main` and feature branches.

### `local-main` (your personal scratch branch)

- Based on `main`, but **tracks `origin/local-main`** (a branch on your fork).
- You run notebooks here and drop untracked directories like `tutorials_local/` and `vision/`.
- You never expect `local-main` to be "clean"; that's its job.
- When upstream changes, you **pull those changes into `local-main` via `main`**.

### Feature branches (e.g. `my-feature-branch`)

- Created from `main` when you want to do real work you might share.
