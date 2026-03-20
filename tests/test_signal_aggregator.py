from metaclaw.signal_aggregator import SignalAggregator, SkillEvolutionConfig
from metaclaw.skill_signal import SkillSignal


def _sig(source="conversation", signal_type="explicit_save", confidence=0.9):
    return SkillSignal(
        source=source,
        signal_type=signal_type,
        content={},
        confidence=confidence,
        timestamp="2026-03-20T10:00:00",
        session_id="s1",
    )


def test_add_and_consume():
    agg = SignalAggregator(SkillEvolutionConfig())
    agg.add([_sig(), _sig()])
    signals = agg.consume()
    assert len(signals) == 2
    assert agg.consume() == []


def test_source_filtering():
    cfg = SkillEvolutionConfig(sources=["conversation"])
    agg = SignalAggregator(cfg)
    agg.add([_sig(source="conversation"), _sig(source="prm")])
    signals = agg.consume()
    assert len(signals) == 1
    assert signals[0].source == "conversation"


def test_should_evolve_explicit_save():
    agg = SignalAggregator(SkillEvolutionConfig())
    agg.add([_sig(signal_type="explicit_save")])
    assert agg.should_evolve() is True


def test_should_evolve_correction_threshold():
    cfg = SkillEvolutionConfig(correction_threshold=3)
    agg = SignalAggregator(cfg)
    agg.add([_sig(signal_type="implicit_correction")])
    assert agg.should_evolve() is False
    agg.add([_sig(signal_type="implicit_correction")])
    assert agg.should_evolve() is False
    agg.add([_sig(signal_type="implicit_correction")])
    assert agg.should_evolve() is True


def test_should_evolve_pattern_high_confidence():
    agg = SignalAggregator(SkillEvolutionConfig(pattern_confidence_threshold=0.8))
    agg.add([_sig(signal_type="pattern_detected", confidence=0.9)])
    assert agg.should_evolve() is True


def test_should_evolve_empty():
    agg = SignalAggregator(SkillEvolutionConfig())
    assert agg.should_evolve() is False


def test_consume_atomic_swap():
    agg = SignalAggregator(SkillEvolutionConfig())
    agg.add([_sig()])
    old_buffer = agg._buffer
    consumed = agg.consume()
    assert consumed is old_buffer
    assert agg._buffer is not old_buffer
    assert agg._buffer == []
