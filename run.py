"""OncoInsights — full pipeline entry point.

Runs Modules 1-6 + 8 end to end: validate -> clean -> EDA -> features ->
build SQLite DB -> statistical analysis -> executive summary. Everything
is config-driven via config/config.yaml; nothing is hardcoded here.

Usage:
    python run.py
"""

from __future__ import annotations

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oncoinsights")

STAGES = [
    ("Module 1: Validate raw data", "src.validate"),
    ("Module 2: Clean data", "src.clean"),
    ("Module 3: Exploratory data analysis", "src.eda"),
    ("Module 4: Feature engineering", "src.features"),
    ("Module 5: Build SQLite database", "src.build_db"),
    ("Module 6: Statistical analysis", "src.stats_analysis"),
    ("Module 8: Executive summary", "src.executive_summary"),
]


def main() -> None:
    import importlib

    start = time.time()
    log.info("Starting OncoInsights pipeline (%d stages)", len(STAGES))

    for label, module_name in STAGES:
        stage_start = time.time()
        log.info("--- %s ---", label)
        try:
            module = importlib.import_module(module_name)
            module.main()
        except Exception:
            log.exception("Stage failed: %s", label)
            sys.exit(1)
        log.info("%s completed in %.1fs", label, time.time() - stage_start)

    log.info("Pipeline complete in %.1fs total.", time.time() - start)
    log.info("Dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
