# smart_bot.py
# Telegram FAQ bot with SQLite DB, fuzzy matching (TF-IDF), and admin-locked /add.
# Polling mode (works locally and for 24/7 cloud like Render).

import os
import sqlite3
import threading
from typing import List, Tuple, Optional

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== CONFIG ==========
TOKEN = os.getenv("BOT_TOKEN")          # set this in your shell or Render
ADMIN_ID = 8075615491                   # <-- YOUR Telegram numeric ID (admin lock)
DB = "kb.db"                            # SQLite database file (auto-created)
SIMILARITY_THRESHOLD = 0.35             # 0.25 more lenient, 0.45 stricter
# ===========================

# ---------- Database ----------
def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS faqs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                q  TEXT NOT NULL UNIQUE,
                a  TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS users(
                user_id    INTEGER PRIMARY KEY,
                username   TEXT,
                first_name TEXT,
                last_name  TEXT,
                joined_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.commit()

def upsert_user(update: Update) -> None:
    u = update.effective_user
    if not u:
        return
    with sqlite3.connect(DB) as c:
        c.execute(
            "INSERT OR IGNORE INTO users(user_id, username, first_name, last_name) VALUES (?,?,?,?)",
            (u.id, u.username, u.first_name, u.last_name),
        )
        c.commit()

def add_faq(q: str, a: str) -> None:
    with sqlite3.connect(DB) as c:
        c.execute("INSERT OR REPLACE INTO faqs(q, a) VALUES(?, ?)", (q.strip(), a.strip()))
        c.commit()

def get_all_faqs() -> List[Tuple[int, str, str]]:
    with sqlite3.connect(DB) as c:
        return c.execute("SELECT id, q, a FROM faqs ORDER BY id").fetchall()

# ---------- “Kinda smart” search (TF-IDF) ----------
_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
_lock = threading.Lock()
_cached_questions: List[str] = []
_cached_answers: List[str] = []
_cached_matrix = None

def _rebuild_index():
    global _cached_questions, _cached_answers, _cached_matrix
    rows = get_all_faqs()
    _cached_questions = [r[1] for r in rows]
    _cached_answers = [r[2] for r in rows]
    if _cached_questions:
        _cached_matrix = _vectorizer.fit_transform(_cached_questions)
    else:
        _cached_matrix = None

def smart_lookup(user_text: str, threshold: float = SIMILARITY_THRESHOLD) -> Optional[str]:
    with _lock:
        if _cached_matrix is None:
            _rebuild_index()
        if _cached_matrix is None:
            return None
        q_vec = _vectorizer.transform([user_text])
        sims = cosine_similarity(q_vec, _cached_matrix)[0]
        best_i = sims.argmax()
        best = sims[best_i]
        if best >= threshold:
            return _cached_answers[best_i]
        return None

# ---------- Handlers ----------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user(update)
    await update.message.reply_text(
        "Hi! I’m your smart-ish bot 🤖\n"
        "• Ask me anything in my knowledge base.\n"
        "• Add knowledge: /add question | answer (admin only)\n"
        "• List knowledge: /list"
    )

async def add_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Admin lock
    if not update.effective_user or update.effective_user.id != ADMIN_ID:
        return await update.message.reply_text("Sorry, admin only.")
    # Expect: /add question | answer
    raw = update.message.text
    parts = raw.split(" ", 1)
    if len(parts) < 2 or "|" not in parts[1]:
        return await update.message.reply_text("Usage:\n/add question | answer")
    q, a = [s.strip() for s in parts[1].split("|", 1)]
    if not q or not a:
        return await update.message.reply_text("Both question and answer are required.")
    add_faq(q, a)
    with _lock:
        _rebuild_index()
    await update.message.reply_text("Saved ✅")

async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = get_all_faqs()
    if not rows:
        return await update.message.reply_text("No entries yet. Add with /add question | answer")
    text = "\n".join(f"{rid}. {q}" for rid, q, _ in rows[:200])
    if len(rows) > 200:
        text += f"\n…(+{len(rows) - 200} more)"
    await update.message.reply_text("Knowledge Base:\n" + text)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user(update)
    message = (update.message.text or "").strip()
    if not message:
        return
    ans = smart_lookup(message)
    if ans:
        return await update.message.reply_text(ans)
    await update.message.reply_text(
        "Hmm, I don’t know that yet.\n"
        "Admin can teach me with:\n"
        "/add question | answer"
    )

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN not set. Export BOT_TOKEN before running.")
    init_db()
    # Preload on first run
    if not get_all_faqs():
        add_faq("price", "Our base price is $25. Promos every Friday.")
        add_faq("cashapp", "Yes, we accept CashApp. Send $ to $YourTag and DM the receipt.")
        add_faq("support contact", "Email support@example.com or DM @YourHandle.")
        _rebuild_index()
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("add", add_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    print("Bot running… Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()

