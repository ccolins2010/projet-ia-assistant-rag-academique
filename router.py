from __future__ import annotations
import re

"""
router.py — Détection des intentions
-----------------------------------

Décide si la question doit aller :
- vers smalltalk
- vers un outil (calc, weather, todo, web)
- vers le RAG

Priorité :
1. météo
2. calculatrice
3. todo
4. recherche web explicite
5. smalltalk simple
6. rag par défaut
"""

# Détecte les mots qui indiquent une recherche météo
_WEATHER_RE = re.compile(r"\b(meteo|météo|weather|temperature|température|temps)\b", re.I)

# Détecte si l'utilisateur veut explicitement une recherche web
# → uniquement si la phrase COMMENCE par "cherche" / "recherche" / "search"
_WEB_RE = re.compile(r"^\s*(cherche|recherche|search)\b", re.I)

# Détecte les commandes TODO (ajout / termine)
_TODO_ADD_RE = re.compile(r"\b(ajoute|ajouter|add)\b", re.I)
_TODO_DONE_RE = re.compile(r"\b(termine|fini|finis|done)\b", re.I)

# Détection des commandes de reset TODO
_TODO_CLEAR_RE = re.compile(
    r"\b(vide tout|vide la liste|reset|clear|efface tout|supprime tout)\b",
    re.I,
)

# Détecte formulaire de calcul
_MATH_HINT_RE = re.compile(r"[0-9+\-*/^()]")
_MATH_FUNC_RE = re.compile(r"\b(sin|cos|tan|sqrt|log|log10|ln|exp|pi|π)\b", re.I)


def _looks_like_math(text: str) -> bool:
    """
    Détection STRICTE des calculs :
    - au moins un chiffre
    - présence d'un opérateur (+ - * / ^) OU d'une fonction math
    - ET correspond à notre regex math
    → Empêche la phrase 'les 7 couches du modèle OSI' d'aller en calculatrice

    Supporte aussi :
    - sin45
    - log100
    - sqrt16
    - etc.
    """
    if not text:
        return False

    t = text.replace(" ", "").lower()

    # 1. Doit contenir un chiffre
    if not any(c.isdigit() for c in t):
        return False

    # 2. Doit contenir un opérateur ou une fonction math
    if not re.search(r"[+\-*/^]", t) and not _MATH_FUNC_RE.search(t):
        return False

    # 3. Structure compatible math
    if not _MATH_HINT_RE.search(text):
        return False

    return True


def route(text: str):
    """
    Retourne un tuple (intent, payload)
    intent ∈ {"smalltalk", "calc", "weather", "todo", "web", "rag"}
    """
    t = (text or "").strip().lower()

    # 1) METEO
    if _WEATHER_RE.search(t):
        return "weather", t

    # 2) CALCUL
    if _looks_like_math(text):
        return "calc", text

    # 3) TODO (ADD / DONE / CLEAR / LIST)
    if _TODO_ADD_RE.search(t) or _TODO_DONE_RE.search(t) or _TODO_CLEAR_RE.search(t):
        return "todo", text

    if t in {"liste", "list"}:
        return "todo", text

    # 4) Recherche web explicite (doit commencer par "cherche"/"recherche"/"search")
    if _WEB_RE.search(t):
        return "web", text

    # 5) Smalltalk (phrases basiques)
    if t in {"bonjour", "salut", "hello", "hi"}:
        return "smalltalk", text

    # 6) Par défaut → RAG
    return "rag", text
