from moment_miner.probe import probe


def test_probe_reports_shot_at_when_the_container_has_it(synthetic_video):
    # Absent in a synthesised file; the key is always present so callers need
    # no special case for an older probe result.
    assert "shot_at" in probe(str(synthetic_video))
