# smart_bot.py
# Telegram bot: hybrid retrieval (BM25 + TF-IDF), structured entities + fuzzy names,
# optional LLM fallback (OpenAI), Flask health (Render Free), NumPy 2–compatible.

import os, re, sqlite3, threading
from typing import List, Tuple, Optional, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz

# Numpy (NumPy 2 safe helpers)
import numpy as np

# Optional: FAISS + OpenAI (semantic + LLM)
try:
    import faiss  # provided by 'faiss-cpu'
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# Flask health server (keeps Render Free alive)
from flask import Flask
from threading import Thread

# =================== CONFIG ===================
TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional
ADMIN_ID = 8075615491

DB = "kb.db"
FAQ_THRESHOLD = 0.60              # confidence threshold for direct FAQ answer
NAME_SIM_THRESHOLD = 85           # RapidFuzz name similarity (0..100)
LLM_MODEL = "gpt-4o-mini"         # OpenAI chat model (optional)
EMB_MODEL = "text-embedding-3-small"
EMB_DIM = 1536
# =============================================

# --------- Flask (Render Free health) ---------
app_flask = Flask(__name__)
@app_flask.get("/")
def health():
    return "ok", 200

def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app_flask.run(host="0.0.0.0", port=port)

# ---------------- Utilities ----------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _minmax01(arr: np.ndarray) -> np.ndarray:
    """NumPy 2 safe [0,1] normalization."""
    a = np.asarray(arr, dtype=float)
    rng = np.ptp(a)  # == a.max() - a.min()
    if rng == 0:
        return np.zeros_like(a, dtype=float)
    return (a - a.min()) / (rng + 1e-9)

# ---------------- Database -----------------
def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS faqs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            q  TEXT NOT NULL UNIQUE,
            a  TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            user_id INTEGER PRIMARY KEY, username TEXT, first_name TEXT, last_name TEXT,
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS entities(
            kind TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL,
            PRIMARY KEY(kind,name))""")
        c.execute("""CREATE TABLE IF NOT EXISTS aliases(
            canonical TEXT NOT NULL, alias TEXT NOT NULL,
            PRIMARY KEY(canonical,alias))""")
        c.execute("""CREATE TABLE IF NOT EXISTS synonyms(
            term TEXT NOT NULL, alt TEXT NOT NULL, PRIMARY KEY(term,alt))""")
        c.execute("""CREATE TABLE IF NOT EXISTS faq_embeddings(
            faq_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL,
            FOREIGN KEY(faq_id) REFERENCES faqs(id) ON DELETE CASCADE)""")
        c.commit()

def upsert_user(update: Update):
    u = update.effective_user
    if not u: return
    with sqlite3.connect(DB) as c:
        c.execute("INSERT OR IGNORE INTO users(user_id,username,first_name,last_name) VALUES(?,?,?,?)",
                  (u.id, u.username, u.first_name, u.last_name))
        c.commit()

def add_faq(q: str, a: str) -> int:
    with sqlite3.connect(DB) as c:
        c.execute("INSERT OR REPLACE INTO faqs(q,a) VALUES(?,?)", (q.strip(), a.strip()))
        c.commit()
        row = c.execute("SELECT id FROM faqs WHERE q=?", (q.strip(),)).fetchone()
        return row[0]

def get_all_faqs() -> List[Tuple[int,str,str]]:
    with sqlite3.connect(DB) as c:
        return c.execute("SELECT id,q,a FROM faqs ORDER BY id").fetchall()

def add_entity(kind: str, name: str, value: str):
    with sqlite3.connect(DB) as c:
        c.execute("INSERT OR REPLACE INTO entities(kind,name,value) VALUES(?,?,?)",
                  (norm(kind), name.strip(), value.strip()))
        c.commit()

def get_entities(kind: Optional[str]=None):
    with sqlite3.connect(DB) as c:
        if kind:
            return c.execute("SELECT kind,name,value FROM entities WHERE kind=?",(norm(kind),)).fetchall()
        return c.execute("SELECT kind,name,value FROM entities").fetchall()

def add_alias(canonical: str, alias: str):
    with sqlite3.connect(DB) as c:
        c.execute("INSERT OR IGNORE INTO aliases(canonical,alias) VALUES(?,?)",(canonical.strip(),alias.strip()))
        c.commit()

def get_alias_map() -> Dict[str, List[str]]:
    with sqlite3.connect(DB) as c:
        rows = c.execute("SELECT canonical,alias FROM aliases").fetchall()
    mp: Dict[str, List[str]] = {}
    for can, al in rows:
        mp.setdefault(can, []).append(al)
    return mp

def add_synonym(term: str, alt: str):
    with sqlite3.connect(DB) as c:
        c.execute("INSERT OR IGNORE INTO synonyms(term,alt) VALUES(?,?)",(norm(term),norm(alt)))
        c.commit()

def get_synonyms() -> Dict[str, List[str]]:
    with sqlite3.connect(DB) as c:
        rows = c.execute("SELECT term,alt FROM synonyms").fetchall()
    mp: Dict[str, List[str]] = {}
    for t,a in rows:
        mp.setdefault(t, []).append(a)
    return mp

# ------------- Retrieval indexes -------------
_vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
_lock = threading.Lock()
_cached_ids: List[int] = []
_cached_qs:  List[str] = []
_cached_as:  List[str] = []
_tfidf = None
_bm25 = None
_faiss_index = None  # semantic index (optional)

def rebuild_indices_and_embeddings():
    """Build TF-IDF, BM25, and semantic index (if OpenAI+FAISS available)."""
    global _cached_ids, _cached_qs, _cached_as, _tfidf, _bm25, _faiss_index
    rows = get_all_faqs()
    _cached_ids = [r[0] for r in rows]
    _cached_qs  = [r[1] for r in rows]
    _cached_as  = [r[2] for r in rows]

    if _cached_qs:
        _tfidf = _vectorizer.fit_transform(_cached_qs)
        _bm25  = BM25Okapi([q.split() for q in _cached_qs])
    else:
        _tfidf = None
        _bm25 = None

    # Semantic index (optional)
    if OPENAI_API_KEY and _HAS_OPENAI and _HAS_FAISS and _cached_qs:
        client = OpenAI(api_key=OPENAI_API_KEY)
        with sqlite3.connect(DB) as c:
            existing = {row[0] for row in c.execute("SELECT faq_id FROM faq_embeddings").fetchall()}
            need = [(fid, q) for fid, q in zip(_cached_ids, _cached_qs) if fid not in existing]
            for fid, q in need:
                emb = client.embeddings.create(model=EMB_MODEL, input=q).data[0].embedding
                c.execute("INSERT OR REPLACE INTO faq_embeddings(faq_id,embedding) VALUES(?,?)",
                          (fid, np.asarray(emb, dtype=np.float32).tobytes()))
            c.commit()
            embs = []
            for fid in _cached_ids:
                row = c.execute("SELECT embedding FROM faq_embeddings WHERE faq_id=?", (fid,)).fetchone()
                if row:
                    embs.append(np.frombuffer(row[0], dtype=np.float32))
        if embs:
            mat = np.vstack(embs).astype("float32")
            faiss.normalize_L2(mat)
            _faiss_index = faiss.IndexFlatIP(EMB_DIM)
            _faiss_index.add(mat)
        else:
            _faiss_index = None
    else:
        _faiss_index = None

def hybrid_rank(query: str) -> Tuple[Optional[int], float]:
    """Return (best_index, score in 0..1) using BM25 + TF-IDF."""
    if not _cached_qs:
        return (None, 0.0)
    bm = _bm25.get_scores(query.split()) if _bm25 else np.zeros(len(_cached_qs))
    cs = cosine_similarity(_vectorizer.transform([query]), _tfidf)[0] if _tfidf is not None else np.zeros(len(_cached_qs))
    bm_n = _minmax01(bm)
    cs_n = _minmax01(cs)
    score = 0.6 * bm_n + 0.4 * cs_n
    idx = int(score.argmax())
    return (idx, float(score[idx]))

def semantic_topk(query: str, k=3):
    """Top-k via FAISS cosine (if available); returns [(idx, score), ...]."""
    if not (_faiss_index is not None and OPENAI_API_KEY and _HAS_OPENAI and _HAS_FAISS and _cached_qs):
        return []
    client = OpenAI(api_key=OPENAI_API_KEY)
    emb = client.embeddings.create(model=EMB_MODEL, input=query).data[0].embedding
    vec = np.asarray(emb, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    D, I = _faiss_index.search(vec, min(k, len(_cached_qs)))
    return [(int(i), float(d)) for d, i in zip(D[0], I[0])]

# ------------- Intent routing & entities -------------
def expand_synonyms(text: str) -> str:
    syn = get_synonyms()
    toks = text.split()
    out=[]
    for t in toks:
        t0 = norm(t); rep=[t]
        for k,alts in syn.items():
            if t0==k or t0 in alts:
                rep = list(set([t]+[k]+alts))
                break
        out.append(" / ".join(rep))
    return " ".join(out)

def detect_birthday(text: str) -> Optional[str]:
    m = re.search(r"\b(bday|birthday)\b(?:\s+of|\s*[:\-]?\s*)\s*([A-Za-z][\w\s'.-]{0,40})\??$", text, re.I)
    if not m:
        m = re.search(r"when\s+is\s+([A-Za-z][\w\s'.-]{0,40})\s*(birthday|bday)\??$", text, re.I)
    if m:
        return (m.group(2) if m.lastindex and m.lastindex>=2 else m.group(1)).strip()
    return None

def entity_lookup(kind: str, name_query: str) -> Optional[str]:
    rows = get_entities(kind)
    if not rows: return None
    names = [r[1] for r in rows]
    alias_map = get_alias_map()
    comps, idx = [], []
    for i,can in enumerate(names):
        comps.append(can); idx.append(i)
        for al in alias_map.get(can, []):
            comps.append(al); idx.append(i)
    best = process.extractOne(name_query, comps, scorer=fuzz.WRatio)
    if best and best[1] >= NAME_SIM_THRESHOLD:
        canonical_idx = idx[best[2]]
        return rows[canonical_idx][2]
    return None

# ------------------- LLM -------------------
def llm_answer(query: str, passages: List[Tuple[str, str]]) -> str:
    """Grounded LLM answer; never crashes on key/quota errors."""
    if not (OPENAI_API_KEY and _HAS_OPENAI):
        return "I’m not sure yet."
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        sys_prompt = (
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "If the answer is not present, say 'I don't know' and suggest teaching with /teach."
        )
        ctx = "\n\n".join(f"[{i}] {t}\n{c}" for i,(t,c) in enumerate(passages,1)) or "No context."
        resp = client.chat.completions.create(
            model=LLM_MODEL, temperature=0.2,
            messages=[
                {"role":"system","content":sys_prompt},
                {"role":"user","content":f"Question: {query}\n\nContext:\n{ctx}"}
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "I couldn’t reach the language model right now. Try again later or teach me with /teach question | answer."

# ----------------- Handlers -----------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user(update)
    await update.message.reply_text(
        "Hi! I’m your smart bot 🤖\n"
        "• Ask FAQs (price, refund policy, etc.)\n"
        "• Birthdays: 'bday of Emmy' (aliases supported)\n"
        "• Teach: /teach question | answer (admin)\n"
        "• Entities: /entity kind | name | value (admin)\n"
        "• Alias: /alias canonical | alias (admin)\n"
    )

async def teach_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or update.effective_user.id != ADMIN_ID:
        return await update.message.reply_text("Admin only.")
    parts = update.message.text.split(" ",1)
    if len(parts)<2 or "|" not in parts[1]:
        return await update.message.reply_text("Usage:\n/teach question | answer")
    q,a = [s.strip() for s in parts[1].split("|",1)]
    add_faq(q,a)
    rebuild_indices_and_embeddings()
    await update.message.reply_text("Learned ✅")

async def entity_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or update.effective_user.id != ADMIN_ID:
        return await update.message.reply_text("Admin only.")
    parts = update.message.text.split(" ",1)
    if len(parts)<2 or parts[1].count("|")<2:
        return await update.message.reply_text("Usage:\n/entity kind | name | value")
    kind,name,val = [s.strip() for s in parts[1].split("|",2)]
    add_entity(kind,name,val)
    await update.message.reply_text(f"Saved {kind}: {name} → {val} ✅")

async def alias_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or update.effective_user.id != ADMIN_ID:
        return await update.message.reply_text("Admin only.")
    parts = update.message.text.split(" ",1)
    if len(parts)<2 or "|" not in parts[1]:
        return await update.message.reply_text("Usage:\n/alias canonical | alias")
    can, al = [s.strip() for s in parts[1].split("|",1)]
    add_alias(can, al)
    await update.message.reply_text(f"Alias added: {al} → {can} ✅")

async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = get_all_faqs()
    if not rows:
        return await update.message.reply_text("No FAQs yet. /teach question | answer")
    await update.message.reply_text("FAQ list:\n" + "\n".join(f"{rid}. {q}" for rid,q,_ in rows[:200]))

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user(update)
    raw = (update.message.text or "").strip()
    if not raw: return

    # 1) Entities (birthday)
    text_expanded = expand_synonyms(norm(raw))
    name = detect_birthday(text_expanded)
    if name:
        val = entity_lookup("birthday", name)
        if val:
            return await update.message.reply_text(f"{name}: {val} 🎂")

    # 2) Hybrid retrieval
    idx, score = hybrid_rank(raw)
    if idx is not None and score >= FAQ_THRESHOLD:
        return await update.message.reply_text(_cached_as[idx])

    # 3) Semantic + LLM fallback (optional)
    topk = semantic_topk(raw, k=3)
    passages = [(_cached_qs[i], _cached_as[i]) for i,_ in topk]
    if not passages and idx is not None:
        passages.append((_cached_qs[idx], _cached_as[idx]))
    answer = llm_answer(raw, passages)
    await update.message.reply_text(answer)

# ---------------- Bootstrap ----------------
def preload():
    # seed some data once
    if not get_all_faqs():
        add_faq("price", "Our base price is $25. Promos every Friday.")
        add_faq("cashapp", "Yes, we accept CashApp. Send $ to $YourTag and DM the receipt.")
        add_faq("support contact", "Email support@example.com or DM @YourHandle.")
    add_synonym("birthday","bday")
    add_synonym("price","rate")
    # sample birthdays
    add_entity("birthday","Emmy","January 14")
    add_entity("birthday","Gigi","January 18")
    add_alias("Emmy","Emmi")
    add_alias("Gigi","Georgina")

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN not set")
    if not OPENAI_API_KEY:
        print("NOTE: OPENAI_API_KEY not set — running in retrieval-only mode (no LLM fallback).")

    init_db()
    preload()
    rebuild_indices_and_embeddings()

    # Keep-alive web server for Render Free
    Thread(target=run_flask, daemon=True).start()

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("teach", teach_cmd))
    app.add_handler(CommandHandler("entity", entity_cmd))
    app.add_handler(CommandHandler("alias", alias_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    print("Smart bot running…")
    app.run_polling()

if __name__ == "__main__":
    main()
