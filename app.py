from __future__ import annotations

# """
# app.py — Version finale stable pour Assistant Académique RAG + Agents
# --------------------------------------------------------------------

# Pipeline général :

# 1) Gestion du consentement de recherche web (oui / non)
# 2) Détection e-mail → envoi SMTP via ton compte Gmail (config .env)
# 3) Smalltalk (discussion simple)
# 4) Agents (calcul, météo, TODO, recherche web explicite)
# 5) RAG interne (réponses strictes basées sur tes documents)
# 6) Si RAG ne sait pas → demande de consentement pour recherche web

# Points clés :

# - Détection d'e-mails robuste, y compris :
#   "envoi la réponse à ...", "envoie un mail à ...", "envoi mail ...", etc.
# - Utilisation d'un mot de passe d'application Gmail via .env
# - Historique de conversation persistant dans memory_store.json
# - Aucune modification des parties déjà fonctionnelles (RAG, TODO, météo, calc)
# """

import json
import os
import re
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag_core import answer_question, reindex, LLM_MODEL
from router import route
from agents import (
    tool_calculator,
    tool_weather_sync,
    tool_web_search,
    tool_todo,
)

from langchain_ollama import ChatOllama


# ─────────────────────────────────────────────
# CONFIG STREAMLIT
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Assistant académique",
    page_icon="🎓",
    layout="centered",
)
st.title("🎓 Assistant Académique — RAG + Agents")

ROOT = Path(__file__).parent
MEMORY_PATH = ROOT / "memory_store.json"
MAX_TURNS = 30


# ─────────────────────────────────────────────
# HISTORIQUE LOCAL (FICHIER JSON)
# ─────────────────────────────────────────────

def load_memory():
    """
    Charge l'historique depuis memory_store.json si présent.
    Ignore les entrées invalides.
    """
    try:
        if MEMORY_PATH.exists():
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict) and "role" in x and "content" in x]
    except Exception:
        pass
    return []


def save_memory(history):
    """
    Sauvegarde l'historique complet sur disque.
    """
    try:
        MEMORY_PATH.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


# ─────────────────────────────────────────────
# SMALLTALK
# ─────────────────────────────────────────────

def get_smalltalk_llm():
    """
    Modèle Ollama dédié au smalltalk (température légèrement plus élevée).
    """
    return ChatOllama(model=LLM_MODEL, temperature=0.4)


# ─────────────────────────────────────────────
# INIT DES ÉTATS DE SESSION
# ─────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_memory()

if "pending_web" not in st.session_state:
    # pending_web = {"query": "..."} quand on attend un "oui/non"
    st.session_state.pending_web = None


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.header("⚙️ Options")

if st.sidebar.button("🆕 Nouveau Chat"):
    st.session_state.chat_history = []
    st.session_state.pending_web = None
    save_memory([])
    st.success("Nouvelle conversation créée.")

if st.sidebar.button("🧹 Effacer historique"):
    st.session_state.chat_history = []
    MEMORY_PATH.write_text("[]", encoding="utf-8")
    st.success("Historique effacé.")

if st.sidebar.button("🔄 Réindexer"):
    reindex()
    msg = "ℹ️ Index reconstruit. Les réponses s'appuieront sur les documents internes mis à jour."
    st.session_state.chat_history.append({"role": "assistant", "content": msg})
    save_memory(st.session_state.chat_history)
    st.success("Documents réindexés.")


# ─────────────────────────────────────────────
# AFFICHAGE DU CHAT
# ─────────────────────────────────────────────

st.subheader("💬 Discussion")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─────────────────────────────────────────────
# RÉSULTATS WEB (FORMATAGE)
# ─────────────────────────────────────────────

def render_web_results(json_payload: str) -> str:
    """
    Transforme le JSON renvoyé par tool_web_search()
    en markdown lisible.
    """
    try:
        data = json.loads(json_payload)
        if isinstance(data, dict) and "error" in data:
            return f"Erreur : {data['error']}"

        if not isinstance(data, list):
            return f"Résultats bruts :\n```json\n{json_payload}\n```"

        out = ["**🌐 Résultats Web :**"]
        for r in data[:5]:
            title = r.get("title", "(titre inconnu)")
            body = r.get("body", "")
            out.append(f"- **{title}**\n  {body}")
        return "\n".join(out)

    except Exception:
        return f"Résultats bruts :\n```json\n{json_payload}\n```"


# ─────────────────────────────────────────────
# EMAIL — DÉTECTION & ENVOI SMTP
# ─────────────────────────────────────────────

# Regex pour détecter une adresse email dans une phrase
EMAIL_RE = re.compile(r"\b([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})\b")


def detect_email_command(text: str) -> Optional[str]:
    """
    Détection robuste des commandes e-mail.

    Exemples détectés :
      - "envoie la réponse à ccolins2010@yahoo.fr"
      - "envoi la reponse à mon mail : xxx@yy.com"
      - "peux-tu envoyer un email à toto@test.org ?"
      - "envoi un mail vers ccolins2010@yahoo;fr"

    Logique :
      1) On vérifie qu'il y a un "trigger" type mail/email/envoi/envoie…
      2) On corrige les ';' en '.' pour yahoo;fr → yahoo.fr
      3) On extrait la première adresse trouvée via regex
    """
    if not text:
        return None

    t_low = text.lower()

    # ⚠️ On inclut explicitement "envoi" (ton cas), ainsi que
    # différentes formes autour de "envoyer".
    triggers = [
        "mail",
        "email",
        "courriel",
        "envoi",      # <--- IMPORTANT : ton cas
        "envoie",
        "envoyer",
        "envoies",
        "envoi un mail",
        "envoie un mail",
    ]

    if not any(trig in t_low for trig in triggers):
        return None

    # Correction de petites fautes de frappe type yahoo;fr → yahoo.fr
    cleaned = text.replace(";", ".").replace(",", ".")

    m = EMAIL_RE.search(cleaned)
    if m:
        return m.group(1)

    return None


def send_email_smtp(to_addr: str, subject: str, body: str):
    """
    Envoi d'un e-mail en SMTP via Gmail (ou autre)
    en utilisant les variables définies dans le fichier .env :

      SMTP_HOST=smtp.gmail.com
      SMTP_PORT=587
      SMTP_USER=...
      SMTP_PASS=...
      SMTP_FROM=...

    Retourne (success: bool, message: str)
    """
    import smtplib
    from email.mime.text import MIMEText

    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", 587))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    frm = os.getenv("SMTP_FROM", user)

    # Vérification de la config de base
    if not all([host, user, pwd, frm]):
        return False, "❌ SMTP non configuré correctement dans le fichier `.env`."

    msg = MIMEText(body or "", "plain", "utf-8")
    msg["From"] = frm
    msg["To"] = to_addr
    msg["Subject"] = subject

    try:
        server = smtplib.SMTP(host, port)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(user, pwd)
        server.sendmail(frm, [to_addr], msg.as_string())
        server.quit()

        return True, f"✅ Email envoyé avec succès à **{to_addr}**"

    except smtplib.SMTPAuthenticationError as e:
        return False, (
            f"❌ Erreur d'authentification SMTP : {e}\n"
            "➡️ Vérifie ton mot de passe d'application et les identifiants dans `.env`."
        )

    except Exception as e:
        return False, f"❌ Erreur SMTP : {e}"


# ─────────────────────────────────────────────
# LOGIQUE GLOBALE DE TRAITEMENT
# ─────────────────────────────────────────────

YES = re.compile(r"^\s*(oui|o|yes|y)\b", re.I)
NO = re.compile(r"^\s*(non|no|n)\b", re.I)


def handle_user_query(user_text: str):
    """
    Gère une nouvelle entrée utilisateur selon le pipeline défini en haut.
    """

    # ───── 0) Gestion d'une réponse OUI/NON pour la recherche web ─────
    if st.session_state.pending_web:
        if YES.search(user_text):
            q = st.session_state.pending_web["query"]
            raw = tool_web_search(q)
            resp = "🛠️ **Recherche Web**\n\n" + render_web_results(raw)

            st.session_state.chat_history += [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": resp},
            ]
            st.session_state.pending_web = None
            save_memory(st.session_state.chat_history)
            st.chat_message("assistant").markdown(resp)
            return

        if NO.search(user_text):
            resp = "OK 👍 Je reste sur tes documents internes."
            st.session_state.chat_history += [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": resp},
            ]
            st.session_state.pending_web = None
            save_memory(st.session_state.chat_history)
            st.chat_message("assistant").markdown(resp)
            return

        # Réponse invalide (ni oui ni non)
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        ans = "Je n’ai pas compris. Réponds **oui** ou **non**."
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        save_memory(st.session_state.chat_history)
        st.chat_message("assistant").markdown(ans)
        return

    # ───── 1) Détection d'une commande d'envoi d'e-mail ─────
    to_addr = detect_email_command(user_text)
    if to_addr:
        # On ajoute d'abord le message utilisateur à l'historique
        st.session_state.chat_history.append({"role": "user", "content": user_text})

        # On récupère la dernière réponse assistant pour l'envoyer par mail
        last_answer = next(
            (m["content"] for m in reversed(st.session_state.chat_history)
             if m["role"] == "assistant"),
            "",
        )

        ok, msg = send_email_smtp(to_addr, "Réponse de l'assistant", last_answer)
        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        save_memory(st.session_state.chat_history)
        st.chat_message("assistant").markdown(msg)
        return

    # ───── 2) Ajout de la question à l'historique ─────
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    # ───── 3) Routing via router.py (smalltalk / tools / rag) ─────
    intent, payload = route(user_text)

    # 3.a Smalltalk
    if intent == "smalltalk":
        llm = get_smalltalk_llm()
        out = llm.invoke([
            {"role": "system", "content": "Tu es un assistant amical, bref et poli."},
            {"role": "user", "content": user_text},
        ])
        answer = out.content
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        save_memory(st.session_state.chat_history)
        st.chat_message("assistant").markdown(answer)
        return

    # 3.b Outils (calculatrice, météo, todo, recherche web explicite)
    if intent in {"calc", "weather", "todo", "web"}:
        try:
            if intent == "calc":
                msg = "🛠️ **Outil Calculatrice**\n\n" + tool_calculator(user_text)

            elif intent == "weather":
                msg = "🛠️ **Outil Météo**\n\n" + tool_weather_sync(payload)

            elif intent == "todo":
                raw = tool_todo(payload)
                # Essaye d'interpréter la réponse comme une liste JSON
                try:
                    tasks = json.loads(raw)
                    if isinstance(tasks, list):
                        lines = ["**📋 Liste des tâches :**"]
                        if not tasks:
                            lines.append("_Aucune tâche._")
                        else:
                            for t in tasks:
                                icon = "✅" if t.get("done") else "⬜"
                                lines.append(f"- {icon} #{t['id']} — {t['text']}")
                        msg = "🛠️ **Outil TODO**\n\n" + "\n".join(lines)
                    else:
                        msg = "🛠️ **Outil TODO**\n\n" + raw
                except Exception:
                    msg = "🛠️ **Outil TODO**\n\n" + raw

            elif intent == "web":
                raw = tool_web_search(payload)
                msg = "🛠️ **Outil Recherche Web**\n\n" + render_web_results(raw)

        except Exception as e:
            msg = f"⚠️ Erreur outil : {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        save_memory(st.session_state.chat_history)
        st.chat_message("assistant").markdown(msg)
        return

    # ───── 4) RAG interne ─────
    with st.spinner("🔎 Recherche dans les documents internes…"):
        res = answer_question(user_text)

    answer = res["answer"]
    docs = res["source_documents"]

    # Si le RAG sait répondre (et n'a pas dit "Je ne sais pas.")
    if docs and answer.lower().strip() != "je ne sais pas.":  
        src = docs[0].metadata.get("source", "inconnu")
        msg = f"{answer}\n\n---\n📎 **Source :** `{src}`"
        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        save_memory(st.session_state.chat_history)
        st.chat_message("assistant").markdown(msg)
        return

    # ───── 5) Si RAG ne sait pas → demande consentement web ─────
    st.session_state.pending_web = {"query": user_text}
    msg = (
        "Je n’ai rien trouvé dans les documents internes.\n\n"
        "👉 Souhaites-tu que je cherche **sur le web** ? (oui / non)"
    )

    st.session_state.chat_history.append({"role": "assistant", "content": msg})
    save_memory(st.session_state.chat_history)
    st.chat_message("assistant").markdown(msg)


# ─────────────────────────────────────────────
# INPUT UTILISATEUR
# ─────────────────────────────────────────────

query = st.chat_input("Pose ta question...")
if query:
    handle_user_query(query)
