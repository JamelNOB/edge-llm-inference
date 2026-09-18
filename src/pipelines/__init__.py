"""
Edge Pipelines Module
=====================
Industrial end-to-end edge motion posture QA & coaching pipelines:
1. PostureCoach (Static & inflection posture QA with unified engine)
2. VideoPostureCoachPipeline (Dual-rate asynchronous video stream QA with HUD)
3. LatPulldownEdgePipeline (Biomechanical parameterized RAG + SLM micro-cue pipeline)
"""
from src.pipelines.posture_coach import EdgePostureCoach
from src.pipelines.video_coach import VideoPostureCoachPipeline
from src.pipelines.lat_pulldown_pipeline import LatPulldownEdgePipeline, RobustCueParser

__all__ = [
    "EdgePostureCoach",
    "VideoPostureCoachPipeline",
    "LatPulldownEdgePipeline",
    "RobustCueParser",
]
