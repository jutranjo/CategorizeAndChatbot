import pytest
import pandas as pd
from datetime import timedelta
from Chatbot.chatbot import query_LLM_for_filters, parse_expr, setup_environment
import dateparser
from datetime import timedelta

env = setup_environment()

client = env["client"]
df = env["df"]
current_time = env["current_time"]
categories = env["categories"]
sources = env["sources"]


test_cases = [
    # Full filters: category + source
    {"query": "Show me cashout issues on livechat", "expected_category": "cashout issues", "expected_source": "livechat"},
    {"query": "Show me account issues on telegram", "expected_category": "account issues", "expected_source": "telegram"},
    {"query": "List time delays from telegram users", "expected_category": "time delays", "expected_source": "telegram"},
    {"query": "Any bonus issue messages on livechat?", "expected_category": "bonus issue", "expected_source": "livechat"},
    {"query": "General inquiry complaints from telegram", "expected_category": "general inquiry", "expected_source": "telegram"},
    {"query": "Are there any withdrawal issue reports on livechat?", "expected_category": "withdrawal issue", "expected_source": "livechat"},
    {"query": "Deposit issues reported on telegram", "expected_category": "deposit issues", "expected_source": "telegram"},
    {"query": "Game issues on livechat", "expected_category": "game issues", "expected_source": "livechat"},
    {"query": "Any freespin issues via telegram?", "expected_category": "freespin issues", "expected_source": "telegram"},
    {"query": "Not actionable messages from livechat", "expected_category": "not actionable", "expected_source": "livechat"},

    # Category only (no source specified)
    {"query": "Show me all cashout issues", "expected_category": "cashout issues", "expected_source": None},
    {"query": "What are the game issues?", "expected_category": "game issues", "expected_source": None},

    # Source only (no category specified)
    {"query": "All messages from telegram", "expected_category": None, "expected_source": "telegram"},
    {"query": "Show me livechat messages", "expected_category": None, "expected_source": "livechat"},

    # No filters
    {"query": "Show me all messages", "expected_category": None, "expected_source": None},
    {"query": "Anything new?", "expected_category": None, "expected_source": None}
]

time_filter_cases = [
    {
        "query": "Show me all messages from the last day",
        "expected_start": (current_time - timedelta(days=1)).replace(microsecond=0).isoformat(),
        "expected_end": current_time.replace(microsecond=0).isoformat()
    },
    {
        "query": "Show me issues reported in the past week",
        "expected_start": (current_time - timedelta(weeks=1)).replace(microsecond=0).isoformat(),
        "expected_end": current_time.replace(microsecond=0).isoformat()
    },
    {
        "query": "Anything in the last hour?",
        "expected_start": (current_time - timedelta(hours=1)).replace(microsecond=0).isoformat(),
        "expected_end": current_time.replace(microsecond=0).isoformat()
    },
    {
        "query": "Messages from the last minute",
        "expected_start": (current_time - timedelta(minutes=1)).replace(microsecond=0).isoformat(),
        "expected_end": current_time.replace(microsecond=0).isoformat()
    },
    {
        "query": "What was reported today?",
        "expected_start": current_time.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
        "expected_end": current_time.replace(microsecond=0).isoformat()
    },
    {
        "query": "Show me this week's messages",
        "expected_start": (current_time - timedelta(days=current_time.weekday())).replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
        "expected_end": current_time.replace(microsecond=0).isoformat()
    }
]

def check_llm_field(field_name: str, actual, expected, query: str):
    """
    Check if the actual field value from the LLM matches the expected one.
    Fails the test with a clean message if not.
    """
    if expected is None:
        if actual not in (None, "null"):
            pytest.fail(
                f"\n❌ Query: {query}\nExpected no {field_name}\nGot: '{actual}'"
            )
    else:
        if actual != expected:
            pytest.fail(
                f"\n❌ Query: {query}\nExpected {field_name}: '{expected}'\nGot: '{actual}'"
            )

# def parse_expr(expr: str, base_time: datetime) -> str:
#     if not expr:
#         return None
#     dt = dateparser.parse(expr, settings={'RELATIVE_BASE': base_time, 'RETURN_AS_TIMEZONE_AWARE': False})
#     return dt.replace(microsecond=0).isoformat() if dt else None

@pytest.mark.parametrize("case", test_cases)
def test_llm_classification_json_case(case):
    result = query_LLM_for_filters(case['query'], env)

    
    print(f"\n🔎 LLM raw result for query: '{case['query']}'\nReturned: {result}\n")

    if not isinstance(result, dict):
        pytest.fail(f"LLM response not parsed as dict for query: '{case['query']}'\nGot: {result}")

    check_llm_field("category", result.get("category"), case.get("expected_category"), case["query"])
    check_llm_field("source", result.get("source"), case.get("expected_source"), case["query"])

    
@pytest.mark.parametrize("case", time_filter_cases)
def test_llm_time_range_extraction(case):
    result = query_LLM_for_filters(case['query'], env)



    print(f"\n🔎 LLM time filter result for query '{case['query']}': {result}")

    assert isinstance(result, dict), "LLM response was not parsed as dict"

    # Extract and parse date expressions
    start_expr = result.get("start_time_expr")
    end_expr = result.get("end_time_expr")

    parsed_start = parse_expr(start_expr, current_time)
    parsed_end = parse_expr(end_expr, current_time)

    print(f"🔎 Parsed exprs: start='{start_expr}' → {parsed_start}, end='{end_expr}' → {parsed_end}")
    # Check parsed values
    assert parsed_start == case["expected_start"], \
        f"Expected start_time: {case['expected_start']}, Got: {parsed_start}"

    assert parsed_end == case["expected_end"], \
        f"Expected end_time: {case['expected_end']}, Got: {parsed_end}"
    
@pytest.mark.parametrize("expr, expected_dt", [
    ("yesterday", (current_time - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)),
    ("1 hour ago", (current_time - timedelta(hours=1)).replace(microsecond=0)),
    ("1 minute ago", (current_time - timedelta(minutes=1)).replace(microsecond=0)),
    ("today", current_time.replace(hour=0, minute=0, second=0, microsecond=0)),
    ("now", current_time.replace(microsecond=0)),
    ("3 days ago", (current_time - timedelta(days=3)).replace(microsecond=0)),
])
def test_parse_expr(expr, expected_dt):
    parsed = dateparser.parse(expr, settings={
        'RELATIVE_BASE': current_time,
        'RETURN_AS_TIMEZONE_AWARE': False
    })

    assert parsed is not None, f"Expression '{expr}' did not parse"

    parsed = parsed.replace(microsecond=0)
    expected_dt = expected_dt.replace(microsecond=0)

    print(f"🧪 Expression: '{expr}' → {parsed} (expected: {expected_dt})")
    assert parsed == expected_dt, f"Failed parsing '{expr}': got {parsed}, expected {expected_dt}"

@pytest.mark.parametrize("query, expected_reset", [
    # CLEAR resets
    ("new query: show me deposit issues", True),
    ("start over: show me cashout issues", True),
    ("reset filters and show me all messages", True),
    ("fresh search: give me freespin issues from telegram", True),

    # FOLLOW-UPS
    ("make that telegram only", False),
    ("change the source to livechat", False),
    ("now show only bonus issues", False),
    ("just change it to game issues", False),

    # AMBIGUOUS / EDGE CASES
    ("how about livechat instead", False),         # implies context
    ("actually, show freespin issues", False),     # soft change in category
    ("okay, now find account issues", False),       # arguably new intent
    ("find me messages again from yesterday", False),  # new time, unclear whether reset
    ("go back to withdrawal issues", False),       # reverting state, but continuing
    ("let's look at livechat only", False),        # sounds like refinement 
])

def test_llm_reset_flag_detection(query, expected_reset):
    result = query_LLM_for_filters(query, env)


    print(f"\n🔎 LLM result for query: '{query}' → reset: {result.get('reset')}")
    assert result.get("reset") == expected_reset, f"Expected reset={expected_reset}, got: {result.get('reset')}"
