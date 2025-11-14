from __future__ import annotations
import re
from typing import Literal, Tuple, Optional

"""
router.py
---------
Routage simple et robuste vers :

- "calc"      : calculatrice (sin, cos, sqrt, +, -, ^, etc.)
- "weather"   : météo
- "web"       : recherche web
- "todo"      : gestion TODO
- "rag"       : questions sur les documents internes
- "smalltalk" : salutations / conversation légère

Idées clés :
- On reconnaît d'abord les outils non ambigus (météo, web, todo).
- La calculatrice NE se déclenche que s'il y a vraiment des maths
  (chiffres + opérateurs/fonctions).
- Une phrase comme "c est quoi la cuisine" ira vers RAG + web,
  pas vers la calculatrice.
"""

# ───────────────────────── Patterns par intention ─────────────────────────

# Mots-clés pour météo, web, todo
_PATTERNS = {
    "weather": re.compile(
        r"\b(meteo|météo|temperature|température|fait-il|fait il|temps|vent|pluie|ensoleillé|neige)\b",
        re.I,
    ),
    "web": re.compile(
        r"\b(recherche|cherche sur (le )?web|internet|google|news|actualit|article|info)\b",
        re.I,
    ),
    "todo": re.compile(
        r"\b(todo|tâche|tache|ajoute|ajouter|add:|done:|list)\b",
        re.I,
    ),
}

# Calcul : verbes explicites
_CALC_MAIN_RE = re.compile(
    r"\b(calcule|calcul|résous|resous|résoudre|resoudre)\b",
    re.I,
)

# Indices d'expressions mathématiques
#   - 2+3, 12 / 4, 3*5, 2^8
#   - 23², 10³
#   - sin45, cos(30), sqrt16, log10(100), pi, π
_MATH_HINT_RE = re.compile(
    r"""
    (                                   # une des formes suivantes :
        \d+\s*[\+\-\*/%^]\s*\d+      |  # 2+3, 4 * 5, 2^8
        \d+\s*[²³]                   |  # 23², 10³
        (sin|cos|tan|sqrt|log10?|ln|exp)\s*\d* |  # sin45, sqrt16, exp2
        \bpi\b | π                      # pi, π
    )
    """,
    re.I | re.X,
)

# smalltalk (salutations)
_GREET = re.compile(r"^\s*(bonjour|salut|coucou|bonsoir|hello|hey)\b", re.I)

# Questions "de cours" (explicites)
_DOC_QUE = re.compile(
    r"\b(c('|\s*)est\s+quoi|définis|definition|définition|explique|selon le cours|dans le cours)\b",
    re.I,
)


# ───────────────────────── Helpers ─────────────────────────


def _looks_like_math(text: str) -> bool:
    """
    Retourne True si 'text' contient vraiment une expression mathématique.

    Conditions :
      - il doit y avoir au moins un chiffre
      - ET un des patterns de _MATH_HINT_RE

    Ça évite par exemple que :
      "c est quoi la cuisine"
    soit pris pour du calcul.
    """
    if not text:
        return False
    if not any(ch.isdigit() for ch in text):
        return False
    return _MATH_HINT_RE.search(text) is not None


# ───────────────────────── Noyau de routage ─────────────────────────


def detect_intent(
    text: str,
) -> Literal["weather", "calc", "web", "todo", "rag", "smalltalk"]:
    """
    Analyse la question en langage naturel et renvoie une intention.

    Logique :
      1) Outils spécifiques (météo, web, todo)
      2) Calculatrice (verbe explicite ou expression math)
      3) Smalltalk (salutations)
      4) Questions de cours explicites
      5) Par défaut : RAG (documents internes)
    """
    q = (text or "").strip()

    # 1) Outils "non ambigus" d'abord (météo, web, todo)
    #    -> évite que "météo à Paris" parte dans la calculatrice
    if _PATTERNS["weather"].search(q) and not _CALC_MAIN_RE.search(q):
        return "weather"
    if _PATTERNS["web"].search(q):
        return "web"
    if _PATTERNS["todo"].search(q):
        return "todo"

    # 2) Calculatrice : verbes explicites OU vraie expression math
    if _CALC_MAIN_RE.search(q) and any(ch.isdigit() for ch in q):
        # ex : "résous moi ceci sin45 + sqrt16 + 23²"
        return "calc"
    if _looks_like_math(q):
        # ex : "2+3*4", "sin45", "23²"
        return "calc"

    # 3) Salutations → smalltalk
    if _GREET.search(q):
        return "smalltalk"

    # 4) Questions de cours → RAG
    if _DOC_QUE.search(q):
        return "rag"

    # 5) Par défaut → RAG (on privilégie les documents internes)
    return "rag"


def route(text: str) -> Tuple[str, Optional[str]]:
    """
    Retourne (intent, payload).

    Ici, le payload est simplement le texte brut original, que
    l'application (app.py) transmettra ensuite au bon outil ou au RAG.
    """
    return detect_intent(text), text
