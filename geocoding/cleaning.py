"""Правила отсева странных/неполных адресов для пайплайна геокодинга."""
from __future__ import annotations

import re
from typing import Any


def _parts(addr: str) -> list[str]:
    """Разбить адрес по запятой/точке с запятой, убрать пустые."""
    return [p.strip() for p in re.split(r"[,;]", addr) if p.strip()]


def looks_like_number_range_only(addr: str) -> bool:
    """Адрес из диапазонов чисел без улицы/города (мусор)."""
    if not addr:
        return True
    s = addr.strip()
    if not re.search(r"[а-яёa-z]{2,}", s, re.IGNORECASE):
        if re.match(r"^[\d\s\-;,.]+$", s):
            return True
    return False


def is_invalid_or_placeholder(addr: str) -> bool:
    """Служебные/некорректные значения: fixme, todo и т.п."""
    if not addr or len(addr) < 3:
        return True
    if re.search(r"fixme|todo|undefined|unknown|tba|\.\.\.", addr, re.IGNORECASE):
        return True
    return False


def is_ru_only_or_ru_plus_one(addr: str) -> bool:
    """Только страна или страна + один элемент (город/регион) — нет улицы."""
    s = addr.strip()
    parts = _parts(s)
    if not parts or parts[0].upper() != "RU":
        return False
    return len(parts) <= 2


def is_ru_plus_postcode_only(addr: str) -> bool:
    """Формат «RU, 123456» — только индекс."""
    return bool(re.match(r"^RU\s*,\s*\d{5,6}\s*$", addr.strip(), re.IGNORECASE))


def has_no_digit(addr: str) -> bool:
    """Нет ни одной цифры — нет номера дома, адрес слишком общий."""
    return not re.search(r"\d", addr)


def more_digits_than_letters(addr: str) -> bool:
    """В адресе больше цифр, чем букв — похоже на мусор (списки чисел, коды)."""
    if not addr:
        return True
    s = addr.strip()
    letters = sum(1 for c in s if c.isalpha())
    digits = sum(1 for c in s if c.isdigit())
    return digits > letters


def is_bad_address(addr: str) -> bool:
    """Итоговое правило: True = адрес выкидываем из пайплайна."""
    if not addr:
        return True
    s = addr.strip()
    if len(s) < 5:
        return True
    if looks_like_number_range_only(s):
        return True
    if is_invalid_or_placeholder(s):
        return True
    if is_ru_only_or_ru_plus_one(s):
        return True
    if is_ru_plus_postcode_only(s):
        return True
    if has_no_digit(s):
        return True
    if more_digits_than_letters(s):
        return True
    return False


def cleaning_reason(addr: str) -> str | None:
    """Возвращает причину отсева или None, если адрес ок."""
    if not addr:
        return "empty"
    s = addr.strip()
    if len(s) < 5:
        return "too_short"
    if looks_like_number_range_only(s):
        return "number_range_only"
    if is_invalid_or_placeholder(s):
        return "invalid_placeholder"
    if is_ru_only_or_ru_plus_one(s):
        return "ru_only_or_ru_plus_one"
    if is_ru_plus_postcode_only(s):
        return "ru_postcode_only"
    if has_no_digit(s):
        return "no_digit"
    if more_digits_than_letters(s):
        return "more_digits_than_letters"
    return None


def filter_rows(rows: list[dict[str, Any]], address_key: str = "address") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Разделить строки на прошедшие и отсеянные. Возвращает (kept, dropped)."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for r in rows:
        addr = r.get(address_key) or ""
        reason = cleaning_reason(addr)
        if reason is None:
            kept.append(r)
        else:
            dropped.append({**r, "_drop_reason": reason})
    return kept, dropped
