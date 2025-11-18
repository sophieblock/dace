# Git Workflow for `dace` Fork

This document explains the Git setup and day-to-day workflow for working on your fork of `spcl/dace` without getting into confusing states.

The goals:

1. **`upstream/main` is the source of truth.**
   You never push there; you only fetch from it.
2. **`origin/main` (your fork) mirrors `upstream/main`.**
   Your fork stays clean and up to date with the original repo.
3. **All your work happens on feature branches off `main`.**
   You don't commit directly to `main`.
4. **If things get messy, you can always reset safely.**
   You keep backup branches before doing anything destructive.

---

## 1. Remote Setup

In this repo you have three remotes configured:

- `origin` – your fork on GitHub: `https://github.com/sophieblock/dace.git`
- `upstream` – the original repo: `https://github.com/spcl/dace`
- `spcl` – an additional alias pointing to the original repo: `https://github.com/spcl/dace.git`

Current safety guard:

- `upstream` has **no valid push URL**, so you can't accidentally push:

  ```bash
  git remote set-url --push upstream no_push
  ```

You only ever push to `origin`.

---

## 2. Branches and Their Purpose

- `main`
  - Local branch that tracks `origin/main`.
  - Always kept in sync with `upstream/main`.
  - **No direct development** here; only merges/rebases from `upstream/main`.

- `local-main`
  - Currently points to the same commit as `main`.
  - You can keep it as a sandbox or delete it if you prefer fewer branches.

- `macos-branch`

- Other branches (future feature branches)
  - Used for experiments and actual development.

**Rule of thumb:**

- `main` = clean, synced with upstream.
- "Real work" = on branches created from `main`.

---

## 3. Keeping `main` Up to Date With Upstream

When you want to get the latest changes from the original repo:

```bash
cd /Users/sophieblock/dev_packages/dace

# 1. Fetch latest upstream changes
git fetch upstream

# 2. Switch to main
git switch main

# 3. Integrate upstream/main into main
# Option A: Merge (simpler history)
git merge upstream/main

# Option B: Rebase (linear history)
# git rebase upstream/main

# 4. Push updated main to your fork
git push origin main
```

After this, the following should all point to the same commit:

- `HEAD` (when on `main`)
- `origin/main`
- `upstream/main`

You can confirm with:

```bash
git rev-parse HEAD
git rev-parse origin/main
git rev-parse upstream/main
```

---

## 4. Creating a New Feature Branch

Whenever you want to start new work (experiments, bugfixes, etc.):

```bash
cd /Users/sophieblock/dev_packages/dace

# 1. Make sure main is up to date (see section 3)
#    Then:
git switch main

# 2. Create a new feature branch based on main
git switch -c my-feature-branch
```

Now make your changes and commit as usual:

```bash
# Edit files...

git status

# Stage changes
git add <files>

# Commit
git commit -m "Describe the change"
```

When you're ready to push:

```bash
# First push with -u to set up tracking
git push -u origin my-feature-branch
```

From then on, you can simply run:

```bash
git push
```

and Git will know to push to `origin/my-feature-branch`.

---

## 5. Bringing New Upstream Changes Into a Feature Branch

If upstream changes while you're working on a feature branch, you can rebase your branch on the latest `main`.

1. Update `main` from upstream (section 3).
2. Rebase your feature branch on top of `main`:

```bash
cd /Users/sophieblock/dev_packages/dace

# 1. Make sure main is updated with upstream
git fetch upstream
git switch main
git merge upstream/main
git push origin main

# 2. Rebase your feature branch onto main
git switch my-feature-branch
git rebase main

# 3. Push rebased branch (may need --force-with-lease)
#    because rebasing rewrites commit history
git push --force-with-lease
```

This keeps your feature branch current with upstream development while avoiding complicated merge histories.

---

## 6. Using the GitHub Web UI to Update Your Fork

You *can* also use GitHub's web interface to sync your fork's `main` with upstream:

1. Go to `https://github.com/sophieblock/dace`.
2. If GitHub detects your fork is behind `spcl/dace`, it will show a banner or button like **"Sync fork"** or **"Update branch"**.
3. Click that to bring `origin/main` up to date with `spcl/main`.

After using the web UI, you still need to update your local clone:

```bash
cd /Users/sophieblock/dev_packages/dace

git switch main
git pull --ff-only origin main
```

Using the web UI is safe and convenient, but relying on a single, consistent CLI workflow (section 3) is usually less confusing.

---

## 7. Never Pushing to Upstream

To avoid ever pushing to `upstream`:

- The push URL is already invalidated:

  ```bash
  git remote -v
  # upstream https://github.com/spcl/dace (fetch)
  # upstream no_push (push)
  ```

- You **always** push to `origin` explicitly:

  ```bash
  # Good
  git push origin main
  git push origin my-feature-branch

  # Avoid
  git push upstream main
  ```

Even if you accidentally type `git push upstream main`, Git will fail because `no_push` is not a real remote.

---

## 8. Recovering if main Gets Messy Again

If `main` ever gets into a bad or confusing state, you can safely reset it using the pattern we've already used.

1. **Create a backup branch** from the current main:

   ```bash
   cd /Users/sophieblock/dev_packages/dace

   git switch main
   git branch main-backup-YYYYMMDD
   ```

2. **Reset local main to upstream/main**:

   ```bash
   git fetch upstream
   git reset --hard upstream/main
   ```

3. **Force-push to your fork** to align `origin/main` again:

   ```bash
   git push origin main --force
   ```

Now:

- `main` and `origin/main` are clean again and match `upstream/main`.
- Your old state is preserved on `main-backup-YYYYMMDD`.

You can cherry-pick from the backup branch onto new feature branches as needed.

---

## 9. Handling Local-Only Directories (`tutorials_local/`, `vision/`)

Your `git status` often shows:

```text
?? tutorials_local/
?? vision/
```

If these are just local experiments that you don't want to commit:

### Option A: Ignore them via `.gitignore`

```bash
cd /Users/sophieblock/dev_packages/dace

printf '\n# Local-only dirs\n/tutorials_local/\n/vision/\n' >> .gitignore

git add .gitignore
git commit -m "Ignore local tutorials and vision directories"
git push origin main
```

After this, these folders won't show up in `git status`.

### Option B: Leave them untracked

You can also just leave them as untracked files. They will appear in `git status` but won't affect pulls/pushes.

---

## 10. Minimal Cheat Sheet

**Update main from upstream and sync fork:**

```bash
cd /Users/sophieblock/dev_packages/dace

git fetch upstream
git switch main
git merge upstream/main
git push origin main
```

**Start new work on a feature branch:**

```bash
cd /Users/sophieblock/dev_packages/dace

git switch main
# (optionally update main first)
git switch -c my-feature-branch

# after committing
git push -u origin my-feature-branch
```

**Rebase a feature branch on latest main:**

```bash
cd /Users/sophieblock/dev_packages/dace

git fetch upstream
git switch main
git merge upstream/main
git push origin main

git switch my-feature-branch
git rebase main
git push --force-with-lease
```

With this setup and these patterns, `upstream/main` stays the canonical source, your fork's `main` stays in sync, and all your work happens on clean, manageable branches.
