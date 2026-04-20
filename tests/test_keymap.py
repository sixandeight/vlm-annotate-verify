from vlm_annotate_verify.reviewer.keymap import Action, KEYMAP, dispatch


def test_dispatch_unknown_key_returns_noop():
    assert dispatch("zzz") is Action.NOOP
    assert dispatch("") is Action.NOOP


def test_dispatch_quality_plus_minus():
    assert dispatch("+") is Action.QUALITY_INC
    assert dispatch("-") is Action.QUALITY_DEC


def test_dispatch_quality_direct_set():
    assert dispatch("1") is Action.QUALITY_SET_1
    assert dispatch("2") is Action.QUALITY_SET_2
    assert dispatch("3") is Action.QUALITY_SET_3
    assert dispatch("4") is Action.QUALITY_SET_4
    assert dispatch("5") is Action.QUALITY_SET_5


def test_dispatch_success_toggle():
    assert dispatch("s") is Action.SUCCESS_TOGGLE


def test_dispatch_mistake_keys():
    assert dispatch("m") is Action.MISTAKE_ADD
    assert dispatch("x") is Action.MISTAKE_DELETE
    assert dispatch("t") is Action.MISTAKE_CHANGE_TYPE
    assert dispatch("b") is Action.MISTAKE_CHANGE_SUBTASK
    assert dispatch("n") is Action.MISTAKE_EDIT_NOTE


def test_dispatch_task_editor():
    assert dispatch("e") is Action.EDIT_TASK


def test_dispatch_actions():
    assert dispatch("space") is Action.ACCEPT_ALL
    assert dispatch("enter") is Action.COMMIT_NEXT
    assert dispatch("r") is Action.REPROMPT


def test_dispatch_navigation():
    assert dispatch("j") is Action.NAV_NEXT
    assert dispatch("k") is Action.NAV_PREV
    assert dispatch("q") is Action.SAVE_QUIT
    assert dispatch("f") is Action.FULL_FRAME
    assert dispatch("?") is Action.HELP_TOGGLE


def test_keymap_has_no_duplicates():
    assert len(set(KEYMAP.values())) == len(KEYMAP.values())


def test_action_enum_values_unique():
    values = [a.value for a in Action]
    assert len(values) == len(set(values))
