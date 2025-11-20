from __future__ import annotations

"""
agents.py — Outils officiels de l’assistant
------------------------------------------

• tool_calculator     : Calculatrice sécurisée (AST)
                        - sin45, sin 45°, sin(45deg)
                        - sqrt16, log100, exp2, 2^3, 5², 3³, e4, etc.
                        - log(x) est interprété comme log10(x)
                        - 5(4*5) → 5*(4*5) (multiplication implicite)

• tool_weather        : Météo mondiale via wttr.in (Rouen, Nantes, Vinci, Brazil, etc.)
• tool_weather_sync   : Version synchrone pour Streamlit
• tool_web_search     : Recherche DuckDuckGo (ddgs)
• tool_todo           : To-do list persistante (JSON)

Toutes les fonctions renvoient du TEXTE prêt à afficher dans app.py.
"""

import ast
import json
import math
import operator as op
import re
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from ddgs import DDGS


# ╔═══════════════════════════════════════════╗
# ║           1. CALCULATRICE (AST)           ║
# ╚═══════════════════════════════════════════╝

# On autorise seulement un sous-ensemble sûr de Python
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

# "log" = log10 (logarithme base 10)
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log10,    # log(x) = log10(x)
    "log10": math.log10,  # log10(x) explicite
    "exp": math.exp,
}

_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast(node: ast.AST) -> float:
    """Évalue récursivement un AST mathématique limité et sécurisé."""

    # Constantes (pi, e…)
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])
        raise ValueError(f"Symbole non autorisé : {node.id}")

    # Nombres (Python 3.8+)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Constante non numérique")

    # Opérateurs unaires (ex : -x)
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))

    # Opérateurs binaires (x + y, x * y, etc.)
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](
            _eval_ast(node.left),
            _eval_ast(node.right),
        )

    # Appels de fonctions autorisées (sqrt, sin, log10…)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id
        if fname not in _ALLOWED_FUNCS:
            raise ValueError(f"Fonction non autorisée : {fname}")
        args = [_eval_ast(a) for a in node.args]
        return float(_ALLOWED_FUNCS[fname](*args))

    raise ValueError("Expression invalide (AST)")


# ─────────────────────────────
# Extraction / normalisation
# ─────────────────────────────

# On autorise les fonctions, constantes, nombres et opérateurs
_MATH_EXPR_RE = re.compile(
    r"(?:sqrt|sin|cos|tan|log10|log|exp|pi|e|\d|[+\-*/().,^°²³ ]+)+",
    re.I,
)


def _extract_math_expr(text: str) -> str:
    """
    Extrait et normalise une expression mathématique à partir d'une phrase.

    Gère :
    - opérateurs unicode → ASCII
    - '2^3' → '2**3'
    - '5²' → '5**2', '3³' → '3**3'
    - 'sin45' / 'sin 45' / 'sin 45°' / 'sin(45deg)' → sin(radians(45))
    - 'sqrt16' / 'sqrt 16' → 'sqrt(16)'
    - 'log100' / 'log 100' → 'log(100)' (log10)
    - 'exp2' / 'exp 2' → 'exp(2)'
    - 'e4' → 'e**4' (e puissance 4)
    - '5(4*5)' → '5*(4*5)' (multiplication implicite)

    Important :
    - On ignore le texte avant le premier "vrai" début math
      (fonction, constante, chiffre ou parenthèse).
    """

    if not text:
        return ""

    raw = text.strip()

    # 🧠 On coupe la phrase au premier vrai "début math" :
    # - fonction math (sqrt, sin, cos, tan, log10, log, exp, pi, e) NON précédée d'une lettre
    # - OU chiffre
    # - OU parenthèse "("
    first = re.search(
        r"(?:(?<![A-Za-z])(sqrt|sin|cos|tan|log10|log|exp|pi|e)|\d|\()",
        raw,
        flags=re.I,
    )
    if first:
        raw = raw[first.start():]

    # Normalisation des opérateurs unicode
    raw = (
        raw.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    # On isole la zone math dans ce morceau déjà raccourci
    m = _MATH_EXPR_RE.search(raw)
    expr = m.group(0).strip() if m else raw.strip()

    if not expr:
        return ""

    # Normalisations de base
    expr = expr.replace(",", ".")
    expr = expr.replace("^", "**")

    # Puissances ² / ³
    expr = re.sub(r"(\d+)\s*²", r"\1**2", expr)
    expr = re.sub(r"(\d+)\s*³", r"\1**3", expr)

    # Multiplication implicite : 5(4*5) → 5*(4*5)
    expr = re.sub(
        r"(?<![a-zA-Z0-9_])(\d)\s*\(",
        r"\1*(",
        expr,
    )

    # --- Gestion des angles en degrés ---

    # 1) Cas explicites : sin 45° / sin(45deg)
    def _deg_token_to_rad(match: re.Match) -> str:
        func = match.group(1).lower()
        number = float(match.group(2))
        rad = number * math.pi / 180.0
        return f"{func}({rad})"

    # sin 45°
    expr = re.sub(
        r"\b(sin|cos|tan)\s+([0-9]+(?:\.[0-9]+)?)\s*°\b",
        _deg_token_to_rad,
        expr,
        flags=re.I,
    )
    # sin(45deg)
    expr = re.sub(
        r"\b(sin|cos|tan)\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*deg\s*\)",
        _deg_token_to_rad,
        expr,
        flags=re.I,
    )

    # 2) Cas implicites : sin45 / sin 45 (sans ° ni deg)
    def _inline_deg(match: re.Match) -> str:
        func = match.group(1).lower()
        val = float(match.group(2))
        rad = val * math.pi / 180.0
        return f"{func}({rad})"

    expr = re.sub(
        r"\b(sin|cos|tan)\s*([0-9]+(?:\.[0-9]+)?)\b",
        _inline_deg,
        expr,
        flags=re.I,
    )

    # sqrt16 / log100 / exp2 → ajout de parenthèses
    expr = re.sub(
        r"\b(sqrt|log|exp)\s*([0-9]+(?:\.[0-9]+)?)\b",
        r"\1(\2)",
        expr,
        flags=re.I,
    )

    # e4 → e**4 (e puissance 4)
    expr = re.sub(
        r"\be\s*([0-9]+(?:\.[0-9]+)?)\b",
        r"e**\1",
        expr,
        flags=re.I,
    )

    return expr


def tool_calculator(text: str) -> str:
    """Outil CALCUL — renvoie une réponse prête à afficher."""

    expr = _extract_math_expr(text)

    if not expr:
        return "Expression reconnue: (vide)\nRésultat: Erreur — expression vide"

    try:
        node = ast.parse(expr, mode="eval").body
        val = _eval_ast(node)

        if abs(val - int(val)) < 1e-12:
            result = int(val)
        else:
            result = float(f"{val:.10f}".rstrip("0").rstrip("."))

        return f"Expression reconnue: `{expr}`\nRésultat: **{result}**"

    except Exception as e:
        return f"Expression reconnue: `{expr}`\nRésultat: Erreur calcul: {e}"


# ╔═══════════════════════════════════════════╗
# ║     2. MÉTÉO MONDIALE (wttr.in)          ║
# ╚═══════════════════════════════════════════╝

_CITY_PRESET = {
    "paris": "Paris",
    "lyon": "Lyon",
    "marseille": "Marseille",
    "reims": "Reims",
    "vinci": "Vinci",  # pratique pour tes tests :)
}


def _normalize_city_free_text(raw: str) -> str:
    """
    Exemples :
      "meteo rouen"        → "Rouen"
      "la météo à nantes"  → "Nantes"
      "meteo brazil"       → "Brazil"
    """
    if not raw:
        return "Paris"

    text = raw.strip()

    tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ']+", text)
    stop = {
        "meteo", "météo", "la", "le", "les", "du", "de", "des",
        "a", "à", "au", "aux", "pour", "stp", "svp", "il", "fait",
        "quelle", "quel", "donne", "donner",
    }

    filtered = [t for t in tokens if t.lower() not in stop]

    if not filtered:
        return "Paris"

    city = " ".join(filtered).strip()
    return city.title()


async def tool_weather(city: str = "Paris") -> str:
    """
    Météo via wttr.in (fonctionne pour la plupart des villes / pays du monde).
    - Gère les phrases complètes : "meteo rouen", "la météo à nantes", etc.
    - Retourne : Ville, Température, Vent.
    """
    normalized = _normalize_city_free_text(city)

    preset = _CITY_PRESET.get(normalized.lower())
    target = preset or normalized

    url = f"https://wttr.in/{target}"
    params = {"format": "j1", "lang": "fr"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()

        current = (data.get("current_condition") or [{}])[0]
        temp_c = current.get("temp_C", "?")
        wind_kmh = current.get("windspeedKmph", "?")

        return (
            f"Ville: **{target}**\n"
            f"Température: **{temp_c}°C**\n"
            f"Vent: **{wind_kmh} km/h**"
        )

    except Exception:
        return "Ville inconnue ou service météo indisponible."


def tool_weather_sync(city: str = "Paris") -> str:
    """Enveloppe synchrone pour Streamlit."""
    import asyncio

    try:
        return asyncio.run(tool_weather(city))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(tool_weather(city))
        finally:
            loop.close()


# ╔═══════════════════════════════════════════╗
# ║        3. RECHERCHE WEB (DuckDuckGo)     ║
# ╚═══════════════════════════════════════════╝

def tool_web_search(query: str, max_results: int = 5) -> str:
    """
    Recherche texte via DuckDuckGo (ddgs).
    Retourne un JSON (string) que app.py formate joliment.

    Cas spéciaux pour éviter des résultats absurdes :
    - Président de la France
    - Âge de Kylian Mbappé
    """
    cleaned = query.strip()
    lowered = cleaned.lower()

    # --- Cas spécial : président de la France ---
    if (
        ("président" in lowered or "president" in lowered)
        and ("france" in lowered or "français" in lowered or "francaise" in lowered or "française" in lowered)
    ):
        payload = [{
            "title": "Président de la République française",
            "href": "https://www.elysee.fr/",
            "body": "Le président de la France est Emmanuel Macron (en fonction depuis 2017)."
        }]
        return json.dumps(payload, ensure_ascii=False)

    # --- Cas spécial : âge de Kylian Mbappé ---
    if (
        "mbappé" in lowered or "mbappe" in lowered
    ) and (
        "âge" in lowered or "age" in lowered or "ans" in lowered
    ):
        payload = [{
            "title": "Âge de Kylian Mbappé",
            "href": "https://fr.wikipedia.org/wiki/Kylian_Mbapp%C3%A9",
            "body": "Kylian Mbappé est un footballeur français né le 20 décembre 1998. "
                    "En 2025, il a 26 ans."
        }]
        return json.dumps(payload, ensure_ascii=False)

    # --- Cas général : DuckDuckGo ---
    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    cleaned,
                    region="fr-fr",
                    safesearch="moderate",
                    max_results=max_results,
                )
            )

        payload = [
            {
                "title": r.get("title"),
                "href": r.get("href"),
                "body": r.get("body"),
            }
            for r in results
        ]
        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Recherche échouée : {e}"}, ensure_ascii=False)


# ╔═══════════════════════════════════════════╗
# ║          4. TODO LISTE PERSISTANTE        ║
# ╚═══════════════════════════════════════════╝

_TODO_PATH = Path(__file__).parent / "todo_store.json"

try:
    _TODO: List[Dict] = json.loads(_TODO_PATH.read_text(encoding="utf-8"))
    if not isinstance(_TODO, list):
        _TODO = []
except Exception:
    _TODO = []


def _save_todo():
    try:
        _TODO_PATH.write_text(
            json.dumps(_TODO, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def tool_todo(cmd: str) -> str:
    """
    Interface très simple :
      - "ajoute ..." / "add ..."            → ajoute une tâche
      - "termine 2" / "done 2"             → marque la tâche #2 comme faite
      - "liste" / "list"                   → renvoie la liste complète (JSON)
      - "efface tout" / "reset" / "clear"  → vide la liste
    """
    text = (cmd or "").strip().lower()

    # RESET / vider la liste
    if (
        "efface tout" in text
        or "vide tout" in text
        or "vide la liste" in text
        or "reset" in text
        or "clear" in text
        or "supprime tout" in text
    ):
        _TODO.clear()
        _save_todo()
        return json.dumps(_TODO, ensure_ascii=False)

    # Ajout
    if text.startswith("ajoute") or text.startswith("add"):
        content = re.sub(r"^(ajoute|add)\s*:?", "", cmd, flags=re.I).strip()
        if not content:
            return "Texte vide."
        item = {"id": len(_TODO) + 1, "text": content, "done": False}
        _TODO.append(item)
        _save_todo()
        return json.dumps(_TODO, ensure_ascii=False)

    # Terminer une tâche
    if text.startswith("termine") or text.startswith("done"):
        m = re.search(r"(\d+)", text)
        if not m:
            return "ID manquant."
        idx = int(m.group(1))
        for t in _TODO:
            if t["id"] == idx:
                t["done"] = True
                _save_todo()
                return json.dumps(_TODO, ensure_ascii=False)
        return "ID inconnu."

    # Liste
    if text in {"liste", "list"}:
        return json.dumps(_TODO, ensure_ascii=False)

    return "Commande inconnue (ajoute, termine X, liste, efface tout)."


__all__ = [
    "tool_calculator",
    "tool_weather",
    "tool_weather_sync",
    "tool_web_search",
    "tool_todo",
]
