from metaclaw.prm_signal_adapter import PRMSignalAdapter
from metaclaw.data_formatter import ConversationSample


def _sample(reward: float) -> ConversationSample:
    return ConversationSample(
        session_id="s1",
        turn_num=1,
        prompt_tokens=[1, 2, 3],
        response_tokens=[4, 5],
        response_logprobs=[-0.5, -0.3],
        loss_mask=[1, 1],
        reward=reward,
        prompt_text="hello",
        response_text="world",
    )


def test_should_adapt_failure():
    adapter = PRMSignalAdapter()
    assert adapter.should_adapt(_sample(-1.0)) is True


def test_should_adapt_success():
    adapter = PRMSignalAdapter()
    assert adapter.should_adapt(_sample(1.0)) is True


def test_should_not_adapt_ambiguous():
    adapter = PRMSignalAdapter()
    assert adapter.should_adapt(_sample(0.0)) is False


def test_adapt_failure():
    adapter = PRMSignalAdapter()
    sig = adapter.adapt(_sample(-1.0), ["skill-a"])
    assert sig.source == "prm"
    assert sig.signal_type == "prm_failure"
    assert sig.confidence == 1.0
    assert sig.content["active_skills"] == ["skill-a"]


def test_adapt_success():
    adapter = PRMSignalAdapter()
    sig = adapter.adapt(_sample(1.0), [])
    assert sig.signal_type == "prm_success"
    assert sig.confidence == 1.0
