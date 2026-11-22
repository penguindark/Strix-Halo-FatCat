# AGENT.md — Project Rules

## Language
- The entire project is in **English**.
- All text, code comments, documentation, commit messages, logs and identifiers
  **must be written in English**. No exceptions.

## Development language
- The development language is **C**, using the **latest standard (C23)**.
- GPU kernels may use HIP/C++ where required by ROCm; everything else stays in C.

## Target platform
- This project targets **AMD Strix Halo (gfx1151)** only. Every architecture-specific
  optimization for Strix Halo (Zen 5 CPU, RDNA 3.5 iGPU, unified LPDDR5X memory)
  is allowed and encouraged.

## Project tracking (mandatory)
- `project.md` — master development tracker (goals, roadmap, status, decisions).
  Update it whenever scope, status or decisions change.
- `development_log.md` — append-only chronological log of the whole development.
  Add an entry at the end of every working session.
- Every folder containing code, scripts or tests contains a `development.md`
  describing that folder's goal, current status, scope and next steps. Keep it
  up to date when touching the folder.

## Manuals
- `Manuals/` stores platform and library documentation (downloaded from the web
  or manually copied). Do not modify the documents themselves; add a note to
  `Manuals/README.md` when adding a new document (source URL, date, license).
