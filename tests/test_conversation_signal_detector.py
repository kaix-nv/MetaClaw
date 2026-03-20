from metaclaw.conversation_signal_detector import ConversationSignalDetector


def _turn(user_msg, assistant_resp="OK", session_id="s1", turn_num=1):
    return {
        "session_id": session_id,
        "turn_num": turn_num,
        "user_message": user_msg,
        "assistant_response": assistant_resp,
        "active_skills": [],
    }


def test_explicit_remember_this():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("Please remember this: always use UTC timestamps"))
    assert len(signals) >= 1
    assert signals[0].signal_type == "explicit_save"
    assert signals[0].confidence >= 0.9


def test_explicit_from_now_on():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("From now on, always check logs first"))
    assert len(signals) >= 1
    assert signals[0].signal_type == "explicit_save"


def test_explicit_save_as_skill():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("Save this as a skill: verify file paths"))
    assert len(signals) >= 1
    assert signals[0].signal_type == "explicit_save"


def test_explicit_never_do():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("Never do that again, always ask before deleting"))
    assert len(signals) >= 1
    assert signals[0].signal_type == "explicit_save"


def test_implicit_correction_no_wrong():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("No, that's wrong. You should use async instead"))
    corrections = [s for s in signals if s.signal_type == "implicit_correction"]
    assert len(corrections) >= 1
    assert corrections[0].confidence >= 0.5


def test_implicit_correction_instead():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("Instead of that, use a context manager"))
    corrections = [s for s in signals if s.signal_type == "implicit_correction"]
    assert len(corrections) >= 1


def test_no_signal_on_normal_message():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("Can you help me write a function?"))
    assert len(signals) == 0


def test_no_signal_on_short_message():
    det = ConversationSignalDetector(use_llm_detection=False)
    signals = det.detect(_turn("ok"))
    assert len(signals) == 0
