# smart_bot.py — LLM-RAG smart bot
# - SQLite KB + Entities
# - Hybrid retrieval (BM25 + TF-IDF) + semantic (FAISS)
# - LLM fallback (OpenAI gpt-4o-mini) with strict grounding
# - Admin: /teach, /entity, /alias
# - Flask health endpoint for Render Free
# - Admin-locked by ADMIN_ID

import os, re, sqlite3, json, threading
from typing import List, Tuple, Optional, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz

# Embeddings + vector search
import faiss
import numpy as np
from openai import OpenAI

# Flask health
from threading import Thread
from flask import Flask

# ==== CONFIG ====
TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required for LLM/embeddings
ADMIN_ID = 8075615491
DB = "kb.db"
EMB_DIM = 1536    # text-embedding-3-small
FAQ_THRESHOLD = 0.60    # confidence threshold for direct FAQ answer
NAME_SIM_THRESHOLD = 85 # RapidFuzz name match
LLM_MODEL = "gpt-4o-mini"
EMB_MODEL = "text-embedding-3-small"
# ===============

# Flask health (Render Free Web Service)
app_flask = Flask(__name__)
@app_flask.get("/")
def health():
    return "ok", 200
def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app_flask.run(host="0.0.0.0", port=port)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

# ---------- DB ----------
def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS faqs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            q TEXT NOT NULL UNIQUE, a TEXT NOT NULL)""")
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
def get_alias_map():
    with sqlite3.connect(DB) as c:
        rows = c.execute("SELECT canonical,alias FROM aliases").fetchall()
    mp: Dict[str,List[str]] = {}
    for can,al in rows:
        mp.setdefault(can,[]).append(al)
    return mp

def add_synonym(term:str, alt:str):
    with sqlite3.connect(DB) as c:
        c.execute("INSERT OR IGNORE INTO synonyms(term,alt) VALUES(?,?)",(norm(term),norm(alt)))
        c.commit()
def get_synonyms():
    with sqlite3.connect(DB) as c:
        rows = c.execute("SELECT term,alt FROM synonyms").fetchall()
    mp: Dict[str,List[str]] = {}
    for t,a in rows:
        mp.setdefault(t,[]).append(a)
    return mp

# ---------- Retrieval indexes ----------
_vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
_lock = threading.Lock()
cached_qs: List[str] = []
cached_as: List[str] = []
cached_ids: List[int] = []
tfidf_m = None
bm25 = None
faiss_index = None

def rebuild_indices_and_embeddings():
    global cached_qs,cached_as,cached_ids,tfidf_m,bm25,faiss_index
    rows = get_all_faqs()
    cached_ids = [r[0] for r in rows]
    cached_qs  = [r[1] for r in rows]
    cached_as  = [r[2] for r in rows]

    # TF-IDF + BM25
    if cached_qs:
        tfidf_m = _vectorizer.fit_transform(cached_qs)
        bm25 = BM25Okapi([q.split() for q in cached_qs])
    else:
        tfidf_m = None
        bm25 = None

    # Embeddings (ensure entries exist)
    if OPENAI_API_KEY and cached_qs:
        client = OpenAI(api_key=OPENAI_API_KEY)
        with sqlite3.connect(DB) as c:
            existing = {row[0] for row in c.execute("SELECT faq_id FROM faq_embeddings").fetchall()}
            need = [ (fid, q) for fid,q in zip(cached_ids, cached_qs) if fid not in existing ]
            for fid, q in need:
                emb = client.embeddings.create(model=EMB_MODEL, input=q).data[0].embedding
                b = np.asarray(emb, dtype=np.float32).tobytes()
                c.execute("INSERT OR REPLACE INTO faq_embeddings(faq_id,embedding) VALUES(?,?)",(fid,b))
            c.commit()
            # build FAISS
            embs = []
            for fid in cached_ids:
                ebytes = c.execute("SELECT embedding FROM faq_embeddings WHERE faq_id=?",(fid,)).fetchone()
                if ebytes:
                    emb = np.frombuffer(ebytes[0], dtype=np.float32)
                    embs.append(emb)
            if embs:
                mat = np.vstack(embs).astype('float32')
                faiss_index = faiss.IndexFlatIP(EMB_DIM)
                # normalize for cosine
                faiss.normalize_L2(mat)
                faiss_index.add(mat)
            else:
                faiss_index = None

def hybrid_rank(query: str):
    """Return best FAQ index and a normalized score (0..1)."""
    if not cached_qs:
        return None, 0.0
    # BM25
    bm = bm25.get_scores(query.split()) if bm25 else np.zeros(len(cached_qs))
    # TF-IDF cosine
    cs = cosine_similarity(_vectorizer.transform([query]), tfidf_m)[0] if tfidf_m is not None else np.zeros(len(cached_qs))
    # normalize
    bm = (bm - bm.min()) / (bm.ptp() + 1e-9)
    cs = (cs - cs.min()) / (cs.ptp() + 1e-9)
    score = 0.6*bm + 0.4*cs
    i = int(score.argmax())
    return i, float(score[i])

def semantic_topk(query: str, k=3):
    """Return top-k indices via FAISS cosine similarity + scores."""
    if faiss_index is None or not OPENAI_API_KEY or not cached_qs:
        return []
    client = OpenAI(api_key=OPENAI_API_KEY)
    emb = client.embeddings.create(model=EMB_MODEL, input=query).data[0].embedding
    vec = np.asarray(emb, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    D, I = faiss_index.search(vec, min(k, len(cached_qs)))
    out = []
    for d,i in zip(D[0], I[0]):
        out.append((int(i), float(d)))
    return out

# ---------- Intent routing ----------
def get_synonym_map_text(text: str) -> str:
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
        name = m.group(2) if m.lastindex and m.lastindex>=2 else m.group(1)
        return name.strip()
    return None

def entity_lookup(kind: str, name_query: str) -> Optional[str]:
    rows = get_entities(kind)
    if not rows: return None
    names = [r[1] for r in rows]
    alias_map = get_alias_map()
    comps=[]; idx=[]
    for i,can in enumerate(names):
        comps.append(can); idx.append(i)
        for al in alias_map.get(can,[]):
            comps.append(al); idx.append(i)
    best = process.extractOne(name_query, comps, scorer=fuzz.WRatio)
    if best and best[1] >= NAME_SIM_THRESHOLD:
        canonical_idx = idx[best[2]]
        return rows[canonical_idx][2]
    return None

# ---------- LLM ----------
def llm_answer(query: str, passages: List[Tuple[str,str]]) -> str:
    """
    passages: list of (title, content). We ground the LLM strictly.
    """
    if not OPENAI_API_KEY:
        return "I’m not sure yet."
    sys_prompt = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the answer is not present, say 'I don't know' and suggest teaching with /teach."
    )
    ctx_blocks = []
    for i,(title,content) in enumerate(passages,1):
        ctx_blocks.append(f"[{i}] {title}\n{content}")
    context_str = "\n\n".join(ctx_blocks) if ctx_blocks else "No context."
    client = OpenAI(api_key=OPENAI_API_KEY)
    msg = [
        {"role":"system","content":sys_prompt},
        {"role":"user","content":f"Question: {query}\n\nContext:\n{context_str}"}
    ]
    resp = client.chat.completions.create(model=LLM_MODEL, messages=msg, temperature=0.2)
    return resp.choices[0].message.content.strip()

# ---------- Handlers ----------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user(update)
    await update.message.reply_text(
        "Hi! I’m your very smart bot 🤖\n"
        "• Teach FAQ: /teach question | answer (admin)\n"
        "• Entities: /entity kind | name | value (admin)\n"
        "• Alias: /alias canonical | alias (admin)\n"
        "• Examples: 'bday of Emmy', 'price', 'refund policy'\n"
    )

async def teach_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or update.effective_user.id != ADMIN_ID:
        return await update.message.reply_text("Admin only.")
    parts = update.message.text.split(" ",1)
    if len(parts)<2 or "|" not in parts[1]:
        return await update.message.reply_text("Usage:\n/teach question | answer")
    q,a = [s.strip() for s in parts[1].split("|",1)]
    fid = add_faq(q,a)
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
    can,al = [s.strip() for s in parts[1].split("|",1)]
    add_alias(can,al)
    await update.message.reply_text(f"Alias added: {al} → {can} ✅")

async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = get_all_faqs()
    if not rows:
        return await update.message.reply_text("No FAQs yet. /teach question | answer")
    text = "\n".join(f"{rid}. {q}" for rid,q,_ in rows[:200])
    await update.message.reply_text("FAQ list:\n"+text)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user(update)
    raw = (update.message.text or "").strip()
    if not raw: return
    text = norm(raw)
    text_expanded = get_synonym_map_text(text)

    # 1) Entities first (birthday demo)
    name = detect_birthday(text_expanded)
    if name:
        val = entity_lookup("birthday", name)
        if val:
            return await update.message.reply_text(f"{name}: {val} 🎂")

    # 2) Hybrid retrieval (BM25 + TF-IDF)
    idx, score = hybrid_rank(raw)
    if idx is not None and score >= FAQ_THRESHOLD:
        return await update.message.reply_text(cached_as[idx])

    # 3) Semantic top-k + LLM fallback
    topk = semantic_topk(raw, k=3)
    passages=[]
    for i,_ in topk:
        passages.append((cached_qs[i], cached_as[i]))
    if not passages and idx is not None:
        # include the best hybrid as context as a last resort
        passages.append((cached_qs[idx], cached_as[idx]))
    answer = llm_answer(raw, passages)
    await update.message.reply_text(answer)

def preload():
    # minimal defaults
    if not get_all_faqs():
        add_faq("price", "Our base price is $25. Promos every Friday.")
        add_faq("cashapp", "Yes, we accept CashApp. Send $ to $YourTag and DM the receipt.")
        add_faq("support contact", "Email support@example.com or DM @YourHandle.")
    add_synonym("birthday","bday")
    add_synonym("price","rate")
    add_entity("birthday","Emmy","January 14")
    add_entity("birthday","Gigi","January 18")
    add_alias("Emmy","Emmi")
    add_alias("Gigi","Georgina")

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN not set")
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set — LLM fallback disabled")
    init_db()
    preload()
    rebuild_indices_and_embeddings()

    # start Flask health
    Thread(target=run_flask, daemon=True).start()

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("teach", teach_cmd))
    app.add_handler(CommandHandler("entity", entity_cmd))
    app.add_handler(CommandHandler("alias", alias_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    print("LLM-RAG bot running…")
    app.run_polling()

if __name__ == "__main__":
    main()
