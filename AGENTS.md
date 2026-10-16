# Development evidence

Keep generated development evidence outside this source checkout. Use the
sibling `warpSPHDevelopment/plotting` directory, or set `WARPSPH_DEVELOPMENT_ROOT`
to relocate its shared parent. Example helpers in `examples/_artifacts.py`
resolve this location. Do not overwrite significant historical runs; choose
a distinct output directory for new evidence.

Keep written findings, limitations and regeneration instructions in Markdown
under `reports/`, and runnable reproduction code in `examples/` and `tests/`.
Do not commit generated screenshots, videos, raw measurements, simulation
states/checkpoints, profiles, run logs or notebook execution outputs. Clear
notebook outputs before committing. See `reports/README.md` for the storage
contract, dependencies and exact frontend regeneration commands.
