"""Single-key dispatch table for the TUI reviewer."""
from enum import Enum


class Action(str, Enum):
    NOOP = "NOOP"

    QUALITY_INC = "QUALITY_INC"
    QUALITY_DEC = "QUALITY_DEC"
    QUALITY_SET_1 = "QUALITY_SET_1"
    QUALITY_SET_2 = "QUALITY_SET_2"
    QUALITY_SET_3 = "QUALITY_SET_3"
    QUALITY_SET_4 = "QUALITY_SET_4"
    QUALITY_SET_5 = "QUALITY_SET_5"

    SUCCESS_TOGGLE = "SUCCESS_TOGGLE"

    MISTAKE_ADD = "MISTAKE_ADD"
    MISTAKE_DELETE = "MISTAKE_DELETE"

    EDIT_TASK = "EDIT_TASK"

    ACCEPT_ALL = "ACCEPT_ALL"
    COMMIT_NEXT = "COMMIT_NEXT"
    REPROMPT = "REPROMPT"

    NAV_NEXT = "NAV_NEXT"
    NAV_PREV = "NAV_PREV"
    SAVE_QUIT = "SAVE_QUIT"
    FULL_FRAME = "FULL_FRAME"
    HELP_TOGGLE = "HELP_TOGGLE"


KEYMAP: dict[str, Action] = {
    "+": Action.QUALITY_INC,
    "-": Action.QUALITY_DEC,
    "1": Action.QUALITY_SET_1,
    "2": Action.QUALITY_SET_2,
    "3": Action.QUALITY_SET_3,
    "4": Action.QUALITY_SET_4,
    "5": Action.QUALITY_SET_5,
    "s": Action.SUCCESS_TOGGLE,
    "m": Action.MISTAKE_ADD,
    "x": Action.MISTAKE_DELETE,
    "e": Action.EDIT_TASK,
    "space": Action.ACCEPT_ALL,
    "enter": Action.COMMIT_NEXT,
    "r": Action.REPROMPT,
    "j": Action.NAV_NEXT,
    "k": Action.NAV_PREV,
    "q": Action.SAVE_QUIT,
    "f": Action.FULL_FRAME,
    "?": Action.HELP_TOGGLE,
}


def dispatch(key: str) -> Action:
    return KEYMAP.get(key, Action.NOOP)
