"""Rule-based instruction parser: "pick up the <target> <relation> <reference> and place it on the <goal>"."""
import re

RELATION_KEYWORDS = {
    "in the top drawer of": "in", "in the bottom drawer of": "in",
    "between": "between", "next to": "near", "on top of": "above",
    "from table center": "at", "on": "on", "in": "in", "at": "at",
    "left of": "left", "to the left of": "left",
    "right of": "right", "to the right of": "right",
    "above": "above", "over": "above",
    "below": "below", "under": "below", "beneath": "below",
    "in front of": "front", "behind": "behind",
    "near": "near", "close to": "near", "far from": "far",
}
SORTED_RELATIONS = sorted(RELATION_KEYWORDS.keys(), key=len, reverse=True)
TABLE_CENTER_NAMES = {"table center", "table centre", "center of the table", "centre of the table"}


def _find_relation(text):
    for phrase in SORTED_RELATIONS:
        if phrase in text:
            return phrase, RELATION_KEYWORDS[phrase]
    return None, None


def _clean(text):
    text = text.strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text.strip(" .")


def parse_instruction(instruction):
    """-> {"target", "reference" (str, or a list of two for 'between'), "relation", "raw"}"""
    text = instruction.lower().strip().rstrip(".")
    pick_clause = text.split(" and place")[0]
    pick_clause = re.sub(r"^(pick up|put|place|move|set)\s+", "", pick_clause)

    relation_phrase, relation = _find_relation(pick_clause)
    if relation is None:
        return {"target": None, "reference": None, "relation": None, "raw": instruction}

    idx = pick_clause.find(relation_phrase)
    target = _clean(pick_clause[:idx])
    remainder = pick_clause[idx + len(relation_phrase):]
    remainder = re.sub(r"^\s*(of|to)\s+", "", remainder)

    if relation == "at" and not remainder.strip():
        reference = "table center"
    else:
        reference = _clean(remainder)

    if relation == "between" and " and " in reference:
        reference = [_clean(p) for p in reference.split(" and ")]

    return {"target": target or None, "reference": reference or None, "relation": relation, "raw": instruction}


def parse_goal(instruction):
    """the second half: 'place it on the plate' -> {"object": "plate", "relation": "on"}"""
    text = instruction.lower().strip().rstrip(".")
    m = re.search(r"\b(?:place|put|set|move)\s+(?:it|them)\s+(.*)$", text)
    if not m:
        return {"object": None, "relation": None}
    rest = m.group(1)
    rel_phrase, rel = _find_relation(rest)
    if rel is None:
        return {"object": _clean(rest) or None, "relation": None}
    obj = rest[rest.find(rel_phrase) + len(rel_phrase):]
    obj = _clean(re.sub(r"^\s*(of|to)\s+", "", obj))
    return {"object": obj or None, "relation": rel}


def is_table_center(phrase):
    return phrase is not None and phrase.strip().lower() in TABLE_CENTER_NAMES


def as_list(ref):
    if ref is None:
        return []
    return list(ref) if isinstance(ref, (list, tuple)) else [ref]
