# third_party — development.md

## Goal
Vendored third-party dependencies (only when unavoidable; prefer system
libraries such as ROCm, liburing, zlib).

## Status
**Empty.** No vendored dependencies yet.

## Rules
- Every vendored library gets a subfolder plus a note here with: name, version,
  upstream URL, license, and why vendoring was necessary.
- Prefer distro/system packages (`liburing-dev`, ROCm install) over vendoring.

## Current vendored libraries
_None._
