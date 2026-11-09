ATHER — ACP PHASE 2 INTELLIGENCE UPGRADE
========================================

Contents
--------
core/adaptive_engine.py
    Complete replacement for the existing ACP engine. Version 0.3.0.

validate_acp_phase2.py
    Regression test for compound-condition analysis, ranking transparency,
    parameter guidance, determinism, and recommendation-only behavior.

What changed
------------
1. Adds deterministic compound profiles:
   - dark + uneven illumination
   - dark + elevated noise proxy
   - weak global + weak local contrast
   - uneven illumination + mixed dark/bright regions
   - elevated noise proxy + weak local contrast
2. Applies documented, deterministic score adjustments for those profiles.
3. Exposes score_breakdown and raw_rule_score for recommendation transparency.
4. Adds cautious, manual parameter starting-point guidance to recommendations.
5. Keeps all nine existing algorithm keys and recommendation-only behavior.
6. Does not change main.py, the real-photo validation module, or enhancement code.

Install
-------
1. Close ATHER.
2. From the project root, make a backup:
       Copy-Item core/adaptive_engine.py core/adaptive_engine_backup.py
3. Extract this ZIP directly into the ATHER project root.
4. If Windows asks, replace:
       core/adaptive_engine.py
   Do not replace main.py or core/acp_validation.py.
5. Activate the existing virtual environment and run:
       python -m py_compile core/adaptive_engine.py validate_acp_phase2.py
       python validate_acp_phase2.py
6. If tests pass, launch:
       python main.py

Important
---------
Parameter values are starting points for manual trials only. Some settings may
not be exposed by every algorithm implementation; the guidance says so where
applicable. ACP does not apply parameters or execute enhancement automatically.
The noise statistic remains an approximate residual proxy, not ground truth.
The thresholds and score weights are engineering heuristics and require
calibration on a documented real-image dataset before scientific claims.
