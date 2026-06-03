def attempt_json_recovery(broken_json: str) -> str:
    """Attempts to close a truncated JSON string by balancing brackets.

    Args:
        broken_json (str): The incomplete JSON string

    Returns:
        str: A potentially valid JSON string with brackets closed
    """
    open_braces = broken_json.count('{') - broken_json.count('}')
    open_brackets = broken_json.count('[') - broken_json.count(']')

    last_complete = max(
        broken_json.rfind('}'),
        broken_json.rfind(']')
    )
    if last_complete != -1:
        broken_json = broken_json[:last_complete + 1]

    open_braces = broken_json.count('{') - broken_json.count('}')
    open_brackets = broken_json.count('[') - broken_json.count(']')

    broken_json += ']' * open_brackets
    broken_json += '}' * open_braces

    return broken_json