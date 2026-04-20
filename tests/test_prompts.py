import json

from vlm_annotate_verify.proposer.prompts import (
    PASS_1_PROMPT, PASS_2_PROMPT, render_pass2_prompt,
)


def test_pass_1_prompt_mentions_required_outputs():
    p = PASS_1_PROMPT
    assert "task" in p.lower()
    assert "boundaries" in p.lower()
    assert "confidence" in p.lower()


def test_pass_1_prompt_lists_expected_subtask_label_examples():
    p = PASS_1_PROMPT.lower()
    for label in ("grasp", "transfer", "place"):
        assert label in p


def test_pass_2_prompt_mentions_quality_success_mistakes():
    p = PASS_2_PROMPT.lower()
    assert "quality" in p
    assert "success" in p
    assert "mistakes" in p
    assert "reasoning" in p


def test_pass_2_prompt_includes_mistake_type_enum():
    p = PASS_2_PROMPT.lower()
    for t in ("drop", "slip", "miss", "collision", "other"):
        assert t in p


def test_pass_2_prompt_quality_scale_described():
    p = PASS_2_PROMPT
    assert "1-5" in p or "1 to 5" in p


def test_render_pass2_injects_boundaries():
    boundaries = json.dumps([{"t_s": 5.0, "label": "grasp"}])
    rendered = render_pass2_prompt(boundaries)
    assert "5.0" in rendered
    assert "grasp" in rendered
    assert "{boundaries_json}" not in rendered


def test_render_pass2_keeps_json_schema_braces():
    rendered = render_pass2_prompt("[]")
    # The schema example in the prompt uses literal {} braces - they must survive .format()
    assert '"segments"' in rendered
    assert '"quality"' in rendered
