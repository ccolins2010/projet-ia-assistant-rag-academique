from __future__ import annotations
"""
app.py
------
Interface Streamlit pour l'assistant académique RAG + Agents.

Flux :
 1) On lit la question de l'utilisateur.
 2) Si on attend une réponse "oui/non" pour la recherche web → on gère ça d'abord.
 3) Sinon, on teste commande e-mail.
 4) Sinon, on route vers :
      - smalltalk (LLM local)
      - outils (calc / météo / todo / web)
      - RAG (documents internes)
 5) Si RAG ne sait pas → demande de consentement pour recherche web.
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # charge .env (SMTP_* etc.)

from rag_core import answer_question, reindex, DOCS_DIR
from router import route
from agents import (
    tool_calculator,
    tool_weather_sync,
    tool_web_search,
    tool_todo,
)

from langchain_ollama import ChatOllama

# ───────────────────────── Config & Mémoire ─────────────────────────

st.set_page_config(
    page_title="Assistant académique (RAG + Agents)",
    page_icon="🎓",
    layout="centered",
)
st.title("🎓 Assistant académique — RAG + Agents (Ollama + Chroma + DuckDuckGo)")

# CSS simple pour la sidebar
st.markdown(
    """
<style>
div[data-testid="stSidebar"] .stButton>button {
    padding: 0.4rem 0.6rem;
    font-size: 0.9rem;
    border-radius: 6px;
    width: 100%;
    margin-bottom: 0.35rem;
}
</style>
""",
    unsafe_allow_html=True,
)

ROOT = Path(__file__).parent
MEMORY_PATH = ROOT / "memory_store.json"

MAX_TURNS = 30  # nombre max de messages en historique


def load_memory() -> List[Dict[str, str]]:
    """Charge l'historique persistant depuis disque."""
    if MEMORY_PATH.exists():
        try:
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [
                    t for t in data
                    if isinstance(t, dict) and "role" in t and "content" in t
                ]
        except Exception:
            pass
    return []


def save_memory(history: List[Dict[str, str]]) -> None:
    """Sauvegarde l'historique sur disque."""
    try:
        MEMORY_PATH.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def trim_history():
    """Garde au plus MAX_TURNS messages."""
    if len(st.session_state.chat_history) > MAX_TURNS:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_TURNS:]


def get_smalltalk_llm():
    """LLM local pour smalltalk via Ollama."""
    return ChatOllama(model="llama3.2:3b", temperature=0.5)


# ──────────────────────── État de session ──────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_memory()

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_mode" not in st.session_state:
    st.session_state.last_mode = None

# pending_web est utilisé lorsqu'on attend "oui/non" pour la recherche web
if "pending_web" not in st.session_state:
    st.session_state.pending_web = None


# ───────────────────────── Sidebar (contrôles) ─────────────────────────

st.sidebar.header("⚙️ Contrôles")

if st.sidebar.button(
    "🆕 Nouveau chat",
    help="Réinitialise la discussion (mais garde la mémoire sur disque).",
):
    st.session_state.chat_history = []
    st.session_state.last_sources = []
    st.session_state.last_mode = None
    st.session_state.pending_web = None
    st.success("Nouvelle discussion démarrée ✅")

if st.sidebar.button(
    "🧹 Effacer historique",
    help="Vide la session et efface l'historique persistant.",
):
    st.session_state.chat_history = []
    try:
        MEMORY_PATH.write_text("[]", encoding="utf-8")
    except Exception:
        pass
    st.success("Historique effacé ✅")

if st.sidebar.button(
    "🔄 Réindexer",
    help="Reconstruit l’index à partir des documents internes.",
):
    reindex()
    st.sidebar.success("Index reconstruit avec succès ✅")
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": (
                "ℹ️ Index reconstruit. Les prochaines réponses utiliseront la "
                "dernière version des documents internes."
            ),
        }
    )
    trim_history()
    save_memory(st.session_state.chat_history)


# ───────────────────── Affichage historique ────────────────────────

st.subheader("💬 Discussion")
for turn in st.session_state.chat_history:
    with st.chat_message("user" if turn["role"] == "user" else "assistant"):
        st.markdown(turn["content"])


# ──────────────────── Helper d’affichage web ───────────────────────

def render_web_results(json_payload: str) -> str:
    """
    Mise en forme des résultats DuckDuckGo.
    Si le JSON est invalide ou erreur, on montre le JSON brut.
    """
    try:
        data = json.loads(json_payload)
        if isinstance(data, dict) and "error" in data:
            return f"Erreur de recherche : {data['error']}"
        if not isinstance(data, list):
            return f"Résultats (brut):\n\n```json\n{json_payload}\n```"

        lines = ["**Résultats web :**"]
        for i, item in enumerate(data[:8], 1):
            title = item.get("title") or "(sans titre)"
            href = item.get("href") or ""
            body = item.get("body") or ""
            if href:
                lines.append(f"- {i}. [{title}]({href})  \n  {body}")
            else:
                lines.append(f"- {i}. {title}  \n  {body}")
        return "\n".join(lines)
    except Exception:
        return f"Résultats (brut):\n\n```json\n{json_payload}\n```"


# ──────────────────── E-mail : détection + envoi SMTP ──────────────

EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")


def detect_email_command(text: str) -> Optional[str]:
    """
    Détecte une commande du type :
      - "envoie la réponse à nom@domaine.com"
      - "mail cette réponse à ..."
    Retourne l’adresse e-mail ou None.
    """
    t = (text or "").lower()
    if any(k in t for k in ["envoie", "envoies", "mail", "email", "e-mail", "envoyer"]):
        m = EMAIL_RE.search(text)
        if m:
            return m.group(1)
    return None


def send_email_smtp(to_addr: str, subject: str, body: str) -> Tuple[bool, str]:
    """
    Envoie un e-mail texte via SMTP.
    Variables attendues dans .env :
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM
    """
    import smtplib
    from email.mime.text import MIMEText

    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM", user)

    if not (user and pwd and from_addr):
        return False, "SMTP non configuré (.env incomplet)."

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port) as s:
            s.starttls()
            s.login(user, pwd)
            s.send_message(msg)
        return True, "E-mail envoyé ✅"
    except Exception as e:
        return False, f"Échec envoi e-mail: {e}"


YES_RE = re.compile(r"^\s*(oui|o|yes|y)\b", re.I)
NO_RE = re.compile(r"^\s*(non|n|no)\b", re.I)


def handle_user_query(user_text: str):
    """
    Route la requête :
      - consentement web (oui/non) si pending_web
      - commande e-mail
      - smalltalk
      - outils
      - RAG (documents)
      - demande de consentement web si RAG ne sait pas
    """

    # 0) Consentement web en cours ?
    if st.session_state.pending_web is not None:
        original_query = st.session_state.pending_web.get("query", "")
        if YES_RE.search(user_text or ""):
            raw_json = tool_web_search(original_query)
            answer_md = (
                "🛠️ **Recherche Web (suite à ton consentement)**\n\n"
                + render_web_results(raw_json)
            )
            mode = "web"
            sources = []
            st.session_state.pending_web = None

            st.session_state.chat_history.append({"role": "user", "content": user_text})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer_md}
            )
            trim_history()
            save_memory(st.session_state.chat_history)
            st.session_state.last_sources = sources
            st.session_state.last_mode = mode
            with st.chat_message("assistant"):
                st.markdown(answer_md)
            return

        if NO_RE.search(user_text or ""):
            answer_md = (
                "👍 D'accord, je reste sur tes documents internes. "
                "Comment puis-je t'aider autrement ?"
            )
            mode = "rag"
            sources = []
            st.session_state.pending_web = None

            st.session_state.chat_history.append({"role": "user", "content": user_text})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer_md}
            )
            trim_history()
            save_memory(st.session_state.chat_history)
            st.session_state.last_sources = sources
            st.session_state.last_mode = mode
            with st.chat_message("assistant"):
                st.markdown(answer_md)
            return

        # Ni oui ni non
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        answer_md = (
            "Je n’ai pas compris. Souhaites-tu que je cherche **sur le web** ? "
            "Réponds par **oui** ou **non**."
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer_md}
        )
        trim_history()
        save_memory(st.session_state.chat_history)
        st.session_state.last_sources = []
        st.session_state.last_mode = "rag"
        with st.chat_message("assistant"):
            st.markdown(answer_md)
        return

    # 0-bis) Commande e-mail ?
    to_addr = detect_email_command(user_text)
    if to_addr:
        st.session_state.chat_history.append({"role": "user", "content": user_text})

        last_assistant = ""
        for t in reversed(st.session_state.chat_history):
            if t["role"] == "assistant":
                last_assistant = t["content"]
                break

        if not last_assistant:
            answer_md = "Je n’ai pas de réponse précédente à envoyer."
        else:
            ok, info = send_email_smtp(
                to_addr,
                subject="Réponse de l'assistant",
                body=last_assistant,
            )
            answer_md = info

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer_md}
        )
        trim_history()
        save_memory(st.session_state.chat_history)
        st.session_state.last_sources = []
        st.session_state.last_mode = "email"
        with st.chat_message("assistant"):
            st.markdown(answer_md)
        return

    # 1) Parcours normal
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    intent, payload = route(user_text)

    # 1.a) Smalltalk
    if intent == "smalltalk":
        llm = get_smalltalk_llm()
        msgs = [
            {"role": "system", "content": "Tu es un assistant amical et bref."},
            {"role": "user", "content": user_text},
        ]
        ai = llm.invoke(msgs)
        answer_md = getattr(ai, "content", str(ai))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer_md}
        )
        trim_history()
        save_memory(st.session_state.chat_history)
        st.session_state.last_sources = []
        st.session_state.last_mode = "smalltalk"
        with st.chat_message("assistant"):
            st.markdown(answer_md)
        return

    # 1.b) Outils
    if intent in {"calc", "weather", "todo", "web"}:
        try:
            if intent == "calc":
                # ⚠️ On passe l'input complet de l'utilisateur,
                # pas seulement payload (qui pourrait être transformé).
                out = tool_calculator(user_text)
                answer_md = f"🛠️ **Outil Calculatrice**\n\n{out}"
            elif intent == "weather":
                out = tool_weather_sync(payload)
                answer_md = f"🛠️ **Outil Météo**\n\n{out}"
            elif intent == "todo":
                out = tool_todo(payload)
                answer_md = f"🛠️ **Outil TODO**\n\n{out}"
            else:  # web explicite
                raw_json = tool_web_search(payload)
                answer_md = (
                    "🛠️ **Outil Recherche Web (DuckDuckGo)**\n\n"
                    + render_web_results(raw_json)
                )

            mode, sources = intent, []

        except Exception as e:
            mode, sources = "error", []
            answer_md = f"⚠️ Erreur: {e}"

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer_md}
        )
        trim_history()
        save_memory(st.session_state.chat_history)
        st.session_state.last_sources = sources
        st.session_state.last_mode = mode
        with st.chat_message("assistant"):
            st.markdown(answer_md)
        return

    # 2) Sinon → RAG en premier
    try:
        with st.spinner("🔎 Recherche dans les documents internes..."):
            res = answer_question(
                user_text,
                chat_history=st.session_state.chat_history,
            )

        answer_text = res.get("answer", "")
        sources = res.get("source_documents", []) or []
        found = (len(sources) > 0) and (
            "je ne sais pas" not in (answer_text or "").lower()
        )

        if found:
            top_src = None
            for d in sources:
                meta = getattr(d, "metadata", {}) or {}
                if meta.get("source"):
                    top_src = meta["source"]
                    break

            if top_src:
                answer_md = f"{answer_text}\n\n---\n📎 **Source** : `{top_src}`"
            else:
                answer_md = answer_text

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer_md}
            )
            trim_history()
            save_memory(st.session_state.chat_history)
            st.session_state.last_sources = sources
            st.session_state.last_mode = "rag"
            with st.chat_message("assistant"):
                st.markdown(answer_md)
            return

    except Exception as e:
        st.warning(f"RAG indisponible: {e}")

    # 3) RAG n'a rien trouvé → consentement web
    st.session_state.pending_web = {"query": user_text}
    answer_md = (
        "Je n’ai rien trouvé dans **les documents internes**.\n\n"
        "👉 Veux-tu que je cherche **sur le web** ? Réponds par **oui** ou **non**."
    )
    mode, sources = "rag", []

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer_md}
    )
    trim_history()
    save_memory(st.session_state.chat_history)
    st.session_state.last_sources = sources
    st.session_state.last_mode = mode
    with st.chat_message("assistant"):
        st.markdown(answer_md)


# Champ de saisie utilisateur
question = st.chat_input("Pose ta question (cours, calcul, météo, web, todo...)")
if question:
    handle_user_query(question)
