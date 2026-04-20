"""Prompt templates for Gemini Pass 1 (boundaries) and Pass 2 (labels).

Both prompts request strict JSON output. The Gemini call site is responsible
for setting response_mime_type='application/json' and any response_schema."""

PASS_1_PROMPT = """\
You are reviewing a robot manipulation episode video.

Identify:
1. A short task description (one sentence, e.g. "place cube on shelf").
2. The subtask boundaries - timestamps in seconds where one mini-step ends and the next begins.

Subtask labels should be short verbs or noun phrases like:
  "grasp", "transfer", "place", "approach", "release", "retract"

For each boundary, give a confidence in [0.0, 1.0] reflecting how sure you are.

Output strict JSON matching this schema:

{
  "task": "string - one sentence",
  "boundaries": [
    {"t_s": 5.2, "label": "grasp",    "confidence": 0.91},
    {"t_s": 8.4, "label": "transfer", "confidence": 0.78}
  ]
}

Boundaries are the END timestamps of each subtask (the last subtask's end is video duration).
"""

PASS_2_PROMPT = """\
You are reviewing a robot manipulation episode video.

The subtask boundaries have already been identified (provided below).
For each subtask SEGMENT, evaluate:

  - quality (1-5):
      1 = robot failed completely or thrashed
      3 = task done but sloppy or inefficient
      5 = clean, smooth, professional execution
  - success (true/false): did the segment achieve its sub-goal?
  - mistakes: list any specific failures during this segment
      mistake types: drop, slip, miss, collision, other

For each numeric judgment, give a confidence in [0.0, 1.0].
Reasoning should be 1-2 sentences explaining the rating.

Boundaries provided:
{boundaries_json}

Output strict JSON matching this schema:

{{
  "segments": [
    {{
      "idx": 0,
      "start_s": 0.0, "end_s": 5.2,
      "quality": 4,    "quality_conf": 0.82,
      "success": true, "success_conf": 0.95,
      "mistakes": [
        {{"type": "slip", "t_s": 3.2, "note": "cube wobble", "confidence": 0.65}}
      ],
      "reasoning": "Smooth grasp, brief lateral cube motion at 3.2s."
    }}
  ]
}}
"""


def render_pass2_prompt(boundaries_json: str) -> str:
    """Inject the pass-1 boundaries JSON into the pass-2 prompt template."""
    return PASS_2_PROMPT.format(boundaries_json=boundaries_json)
