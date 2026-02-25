"""Аугментации адресов: перестановки слов, мусорные слова, опечатки (SAGE опционально)."""
from __future__ import annotations

import random
from typing import Any

# Слова/фразы для добавления или удаления (мусор в контексте геокодинга по одному адресу)
JUNK_PREFIX_SUFFIX = [
    "Санкт-Петербург",
    "СПб",
    "RU",
    "Ленинградская область",
    "Россия",
    "г. Санкт-Петербург",
]

_sage_char_corruptor: Any = None
_sage_available: bool | None = None


def _get_sage_corruptor():
    global _sage_char_corruptor, _sage_available
    if _sage_available is None:
        try:
            from sage.spelling_corruption import CharAugConfig, CharAugCorruptor

            config = CharAugConfig(
                unit_prob=0.15,
                min_aug=1,
                max_aug=4,
            )
            _sage_char_corruptor = CharAugCorruptor.from_config(config)
            _sage_available = True
        except Exception:
            _sage_char_corruptor = None
            _sage_available = False
    return _sage_char_corruptor


def shuffle_parts(text: str, sep: str = ",", prob: float = 0.2, rng: random.Random | None = None) -> str:
    """Разбивает по sep, с вероятностью prob перемешивает части."""
    rng = rng or random
    parts = [p.strip() for p in text.split(sep) if p.strip()]
    if len(parts) <= 1 or rng.random() > prob:
        return text
    random.shuffle(parts)
    return ", ".join(parts)


def add_remove_junk(text: str, prob_add: float = 0.15, prob_remove: float = 0.2, rng: random.Random | None = None) -> str:
    """С вероятностью удаляет вхождения мусорных слов; с вероятностью добавляет одно в начало/конец."""
    rng = rng or random
    out = text
    for junk in JUNK_PREFIX_SUFFIX:
        if junk in out and rng.random() < prob_remove:
            out = out.replace(junk, "").strip()
            out = out.replace(",,", ",").strip()
    if rng.random() < prob_add and JUNK_PREFIX_SUFFIX:
        word = rng.choice(JUNK_PREFIX_SUFFIX)
        if rng.random() < 0.5:
            out = f"{word}, {out}"
        else:
            out = f"{out}, {word}"
    return out


def apply_typos_sage(text: str, seed: int | None = None) -> str:
    """Применяет SAGE CharAugCorruptor если доступен. Иначе возвращает text."""
    corruptor = _get_sage_corruptor()
    if corruptor is None:
        return text
    try:
        # Разные версии SAGE: seed как keyword или только text
        return corruptor.corrupt(text, seed=seed or 0)
    except TypeError:
        try:
            return corruptor.corrupt(text)
        except Exception:
            return text
    except Exception:
        return text


class AddressAugmentor:
    """Комбинирует перестановки, мусорные слова и опечатки (SAGE опционально)."""

    def __init__(
        self,
        shuffle_prob: float = 0.2,
        junk_add_prob: float = 0.15,
        junk_remove_prob: float = 0.2,
        typo_prob: float = 0.3,
        typo_use_sage: bool = True,
        seed: int | None = None,
    ):
        self.shuffle_prob = shuffle_prob
        self.junk_add_prob = junk_add_prob
        self.junk_remove_prob = junk_remove_prob
        self.typo_prob = typo_prob
        self.typo_use_sage = typo_use_sage
        self._rng = random.Random(seed) if seed is not None else random

    def __call__(self, address: str, seed: int | None = None) -> str:
        rng = random.Random(seed) if seed is not None else self._rng
        text = address
        text = shuffle_parts(text, prob=self.shuffle_prob, rng=rng)
        text = add_remove_junk(text, prob_add=self.junk_add_prob, prob_remove=self.junk_remove_prob, rng=rng)
        if self.typo_prob > 0 and rng.random() < self.typo_prob and self.typo_use_sage:
            text = apply_typos_sage(text, seed=seed)
        return text.strip() or address
