"""RITSMS Near-Miss 2.0 — proposal-faithful conflict-extraction pipeline.

Stage modules map 1:1 to the proposal (proposal4.pdf) processing chain:
    ingest -> detect -> track -> trajectory (+forecast) -> conflict -> outputs

Only the validated proposal math is reused from the parent package
(utils.measures, utils.site, bev_web_calibrator); the pipeline and the
protocol data outputs (§7.1 trajectory file, §7.2 conflict file) are rebuilt
here. See ritsms/config.py for all tunables.
"""

__version__ = "2.0.0"
