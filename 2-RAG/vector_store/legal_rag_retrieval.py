"""
Legal RAG Retrieval System
===========================

سیستم جستجوی هوشمند برای پرسش‌وپاسخ حقوقی با استفاده از:
- Metadata-aware retrieval (نام قانون + شماره ماده/اصل)
- Semantic search (bge-m3 embeddings روی Qdrant)
- Cohere reranking (rerank-multilingual-v3.0)

نویسنده: OMID Moradi
تاریخ: دسامبر 2025
"""

from typing import List, Dict
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
import cohere
import re
import os
import numpy as np

# ===========================
# تنظیمات
# ===========================
QDRANT_URL = "http://localhost:6333"

COLLECTION_LAWS = "legal_laws"
COLLECTION_UNITY = "legal_votes_unity"
COLLECTION_DADNAMEH = "legal_votes_dadnameh"

MODEL_EMBED = "baai/bge-m3"

# ===========================
# API Keys
# ===========================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_EMBEDDINGS_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_EMBEDDINGS_API_KEY تنظیم نشده است")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY تنظیم نشده است")

# ===========================
# اتصال به سرویس‌ها
# ===========================
qdrant = QdrantClient(url=QDRANT_URL)

client_embed = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

co = cohere.Client(api_key=COHERE_API_KEY)

print("✓ Qdrant متصل شد")
print("✓ Cohere آماده است")
print("✓ Embedding client آماده است")

# ===========================
# Embedding function
# ===========================
def embed_query(text: str) -> np.ndarray:
    """تبدیل متن سؤال به Dense embedding"""
    resp = client_embed.embeddings.create(
        model=MODEL_EMBED,
        input=[text]
    )
    emb = np.array(resp.data[0].embedding, dtype="float32")
    return emb

# ===========================
# بارگذاری فایل‌های متادیتا (ساده)
# ===========================
QAVANIN_PATH = os.path.join(os.path.dirname(__file__), "qavanin_karbordi.txt")

# فقط QAVANIN_LIST (بدون LAW_TO_DOMAIN)
with open(QAVANIN_PATH, "r", encoding="utf-8") as f:
    QAVANIN_TEXT = f.read()
    lines = [line.strip() for line in QAVANIN_TEXT.split('\n') if line.strip()]
   
    QAVANIN_LIST = [
        line for line in lines
        if line.startswith(('قانون', 'لایحه', 'آیین‌نامه', 'آیین نامه', 'دستورالعمل', 'احکام'))
    ]

print(f"✓ بارگذاری شد: {len(QAVANIN_LIST)} قانون کاربردی")
print("\n" + "="*80)

def normalize_for_match(text: str) -> str:
    """
    نرمال‌سازی موقت فقط برای مقایسه (حفظ متن اصلی با نیم‌فاصله)
    تبدیل نیم‌فاصله به فاصله و یکسان‌سازی فاصله‌ها
    """
    t = text.replace('\u200c', ' ')  # نیم‌فاصله → فاصله
    t = re.sub(r'\s+', ' ', t)       # چند فاصله → یک فاصله
    return t.strip().lower()

# ===========================
# تبدیل اعداد فارسی به انگلیسی (کامل)
# ===========================
FARSI_ONES = {
    'صفر': 0, 'یک': 1, 'دو': 2, 'سه': 3, 'چهار': 4, 'پنج': 5,
    'شش': 6, 'هفت': 7, 'هشت': 8, 'نه': 9
}

FARSI_TENS = {
    'ده': 10, 'یازده': 11, 'دوازده': 12, 'سیزده': 13, 'چهارده': 14, 'پانزده': 15,
    'شانزده': 16, 'هفده': 17, 'هجده': 18, 'نوزده': 19,
    'بیست': 20, 'سی': 30, 'چهل': 40, 'پنجاه': 50,
    'شصت': 60, 'هفتاد': 70, 'هشتاد': 80, 'نود': 90
}

FARSI_HUNDREDS = {
    'یکصد': 100, 'صد': 100, 'دویست': 200, 'سیصد': 300, 'چهارصد': 400,
    'پانصد': 500, 'ششصد': 600, 'هفتصد': 700, 'هشتصد': 800, 'نهصد': 900
}

FARSI_ORDINAL_SIMPLE = {
    'اول': '1', 'دوم': '2', 'سوم': '3', 'چهارم': '4', 'پنجم': '5',
    'ششم': '6', 'هفتم': '7', 'هشتم': '8', 'نهم': '9', 'دهم': '10',
    'یازدهم': '11', 'دوازدهم': '12', 'سیزدهم': '13', 'چهاردهم': '14', 'پانزدهم': '15',
    'شانزدهم': '16', 'هفدهم': '17', 'هجدهم': '18', 'نوزدهم': '19', 'بیستم': '20',
    'نخست': '1', 'سی‌ام': '30', 'چهلم': '40', 'پنجاهم': '50', 'شصتم': '60',
    'هفتادم': '70', 'هشتادم': '80', 'نودم': '90', 'یکصدم': '100'
}

def farsi_word_to_number(text: str) -> int:
    """تبدیل عدد فارسی (ترکیبی) به عدد انگلیسی"""
    text = text.strip().replace('‌', ' ')  # تبدیل نیم‌فاصله به فاصله
    
    if text.isdigit():
        return int(text)
    
    if text in FARSI_ORDINAL_SIMPLE:
        return int(FARSI_ORDINAL_SIMPLE[text])
    
    parts = text.split()
    total = 0
    current = 0
    
    for part in parts:
        if part == 'و':
            continue
        
        if part in FARSI_HUNDREDS:
            current += FARSI_HUNDREDS[part]
        elif part in FARSI_TENS:
            current += FARSI_TENS[part]
        elif part in FARSI_ONES:
            current += FARSI_ONES[part]
        elif part.endswith('م') or part.endswith('مین'):
            base = part.rstrip('مین').rstrip('م')
            if base in FARSI_TENS:
                current += FARSI_TENS[base]
            elif base in FARSI_ONES:
                current += FARSI_ONES[base]
    
    total += current
    return total if total > 0 else None

def normalize_article_number(text: str) -> str:
    """تبدیل شماره فارسی (ساده یا ترکیبی) به انگلیسی"""
    text = text.strip()
    
    if text.isdigit():
        return text
    
    try:
        number = farsi_word_to_number(text)
        return str(number) if number else text
    except:
        return text

def parse_question_metadata_simple(question: str) -> dict:
    """
    استخراج ساده: فقط نام قانون + اصل/ماده
   
    Returns:
    - law_name: نام قانون یا None
    - article_number: شماره یا None  
    - article_type: "اصل" یا "ماده" یا None
    - has_metadata: True اگر حداقل یکی پیدا شد
    """
    q = question.strip()
    q_norm = normalize_for_match(q)
   
    result = {
        "law_name": None,
        "article_number": None,
        "article_type": None,
        "has_metadata": False
    }
   
    # ===========================
    # 1. نام قانون (دقیق + فازی)
    # ===========================
    # تطبیق دقیق
    for law in QAVANIN_LIST:
        if normalize_for_match(law) in q_norm:
            result["law_name"] = law
            result["has_metadata"] = True
            break
   
    # تطبیق فازی
    if not result["law_name"]:
        candidates = []
        for law in QAVANIN_LIST:
            law_clean = law.replace('لایحه قانونی', '').replace('قانون', '').replace('آیین‌نامه', '').strip()
            law_keywords_raw = [w for w in law_clean.split() if len(w) > 3][:4]
           
            if law_keywords_raw:
                law_keywords_norm = [normalize_for_match(kw) for kw in law_keywords_raw]
                matches = sum(1 for kw_norm in law_keywords_norm if kw_norm in q_norm)
               
                if matches >= 1:
                    match_ratio = matches / len(law_keywords_norm)
                    score = (match_ratio * 100) + len(law_keywords_norm)
                    candidates.append({'law': law, 'matches': matches, 'score': score})
       
        if candidates:
            candidates.sort(key=lambda x: (x['matches'], x['score']), reverse=True)
            best = candidates[0]
            if best['matches'] >= 2 or len([w for w in best['law'].split() if len(w) > 3]) == 1:
                result["law_name"] = best['law']
                result["has_metadata"] = True
   
    # ===========================
    # 2. اصل/ماده
    # ===========================
    patterns = [
        (r'اصل\s+([آ-ی\s]+(?:م|مین)?|\d+)', 'اصل'),
        (r'(?:ماده|مواد)\s+(\d+(?:\s*مکرر)?(?:\s*تا\s*\d+)?(?:\s*و\s*\d+)*)', 'ماده'),
    ]
   
    for pattern, article_type in patterns:
        match = re.search(pattern, q)
        if match:
            raw_number = match.group(1).strip()
            result["article_number"] = normalize_article_number(raw_number)
            result["article_type"] = article_type
            result["has_metadata"] = True
            break
   
    return result

# ==========================================
#  Retrieval بدون متادیتا (Semantic Only)
# ==========================================
def retrieve_semantic_only(query_text: str, top_k_total: int = 20):
    """
    جستجوی صرفاً معنایی (بدون استفاده از متادیتا) روی ۳ کالکشن:
    - COLLECTION_LAWS
    - COLLECTION_UNITY
    - COLLECTION_DADNAMEH
    """
    # 1) Embed کردن سؤال
    query_vec = embed_query(query_text).tolist()
    all_results = []

    # 2) توزیع ساده top_k بین سه کالکشن (با وزن بیشتر برای قوانین)
    laws_limit = max(8, top_k_total // 2)
    unity_limit = max(4, top_k_total // 4)
    dadnameh_limit = max(4, top_k_total // 4)

    collections = [
        (COLLECTION_LAWS, "قانون", laws_limit),
        (COLLECTION_UNITY, "وحدت رویه", unity_limit),
        (COLLECTION_DADNAMEH, "دادنامه", dadnameh_limit),
    ]

    for collection_name, source_type, limit in collections:
        try:
            search_result = qdrant.query_points(
                collection_name=collection_name,
                query=query_vec,
                limit=limit,
                with_payload=True,
                with_vectors=False  # فقط امتیاز و payload کافی است
            )

            for hit in search_result.points:
                all_results.append({
                    "text": hit.payload.get("page_content", "")[:3500],
                    "source_type": source_type,
                    "collection": collection_name,
                    "retrieval_score": float(hit.score),
                    "metadata": dict(hit.payload),
                    "matched_via": "semantic_only"
                })
        except Exception as e:
            print(f"⚠️ خطا در جستجو روی کالکشن {collection_name}: {e}")

    # 3) مرتب‌سازی بر اساس امتیاز و محدود کردن به top_k_total
    all_results.sort(key=lambda x: x["retrieval_score"], reverse=True)
    final_results = all_results[:top_k_total]

    print(f"🔍 Semantic only: {len(final_results)} نتیجه (از {len(all_results)} نتیجه خام)")
    return final_results

def retrieve_with_metadata(query_text: str, top_k_total: int = 20):
    """
    جستجو با استفاده از متادیتا (نام قانون + شماره اصل/ماده)

    اولویت‌ها:
      1) law+article : قانون + شماره دقیق اصل/ماده (با Range روی فیلد عددی)
      2) law_only    : فقط قانون (بدون شماره مشخص)
      3) semantic    : جستجوی صرفاً معنایی در همه کالکشن‌ها
    """
    # 1️⃣ استخراج متادیتا ساده از سؤال
    parsed = parse_question_metadata_simple(query_text)
    query_vec = embed_query(query_text).tolist()
    all_results = []

    law_name = parsed["law_name"]
    article_num = parsed["article_number"]
    article_type = parsed["article_type"]

    print(f"🎯 Metadata: law='{law_name}' | {article_type} {article_num}")

    # اولویت مرتب‌سازی نهایی
    priority = {"law+article": 0, "law_only": 1, "semantic": 2}

    # ===========================
    # 1️⃣ قانون + شماره اصل/ماده (با Range)
    # ===========================
    if law_name and article_num:
        print("🔍 فیلتر قانون + شماره (با Range)...")

        # انتخاب فیلد صحیح در payload بر اساس نوع
        if article_type == "اصل":
            field_name = "principle_number"
        elif article_type == "ماده":
            field_name = "article_number"
        else:
            field_name = None

        if field_name:
            # تلاش برای تبدیل به float تا با هر دو نوع int/float در payload مچ شود
            article_num_float = None
            try:
                article_num_float = float(article_num)
            except Exception:
                article_num_float = None

            try:
                filters = [
                    FieldCondition(
                        key="law_name",
                        match=MatchValue(value=law_name)
                    )
                ]

                # اگر توانستیم به عدد تبدیل کنیم، از Range استفاده می‌کنیم
                if article_num_float is not None:
                    filters.append(
                        FieldCondition(
                            key=field_name,
                            range=Range(
                                gte=article_num_float - 0.1,
                                lte=article_num_float + 0.1
                            )
                        )
                    )

                hits = qdrant.query_points(
                    collection_name=COLLECTION_LAWS,
                    query=query_vec,
                    query_filter=Filter(must=filters),
                    limit=10,
                    with_payload=True,
                    with_vectors=False
                ).points

                print(f"   ✓ {len(hits)} سند (law+{field_name})")

                for hit in hits:
                    meta = dict(hit.payload)
                    # متن اصلی برای قوانین: text → page_content
                    main_text = meta.get("text") or meta.get("page_content", "")
                    all_results.append({
                        "text": main_text[:3500],
                        "source_type": "قانون",
                        "collection": COLLECTION_LAWS,
                        "retrieval_score": float(hit.score),
                        "metadata": meta,
                        "matched_via": "law+article"
                    })
            except Exception as e:
                print(f"⚠️ خطا در law+article: {e}")

    # ===========================
    # 2️⃣ فقط قانون (اگر هنوز نتایج کافی نداریم)
    # ===========================
    if law_name and len(all_results) < top_k_total // 2:
        print("🔍 فیلتر فقط قانون...")

        try:
            hits = qdrant.query_points(
                collection_name=COLLECTION_LAWS,
                query=query_vec,
                query_filter=Filter(must=[
                    FieldCondition(
                        key="law_name",
                        match=MatchValue(value=law_name)
                    )
                ]),
                limit=10,
                with_payload=True,
                with_vectors=False
            ).points


            print(f"   ✓ {len(hits)} سند (law_only)")

            for hit in hits:
                meta = dict(hit.payload)
                main_text = meta.get("text") or meta.get("page_content", "")
                all_results.append({
                    "text": main_text[:3500],
                    "source_type": "قانون",
                    "collection": COLLECTION_LAWS,
                    "retrieval_score": float(hit.score),
                    "metadata": meta,
                    "matched_via": "law_only"
                })
        except Exception as e:
            print(f"⚠️ خطا در law_only: {e}")

    # ===========================
    # 3️⃣ Semantic fallback (همه کالکشن‌ها)
    # ===========================
    if len(all_results) < top_k_total:
        print("🔍 Semantic fallback...")

        collections = [
            (COLLECTION_LAWS, "قانون", 8),
            (COLLECTION_UNITY, "وحدت رویه", 6),
            (COLLECTION_DADNAMEH, "دادنامه", 6)
        ]

        for collection_name, source_type, limit in collections:
            try:
                hits = qdrant.query_points(
                    collection_name=collection_name,
                    query=query_vec,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                ).points

                for hit in hits:
                    meta = dict(hit.payload)
                    # انتخاب متن مناسب بر اساس نوع منبع
                    if source_type == "قانون":
                        main_text = meta.get("text") or meta.get("page_content", "")
                    else:
                        # برای آراء وحدت رویه و دادنامه: vote_text → text → page_content
                        main_text = meta.get("vote_text") or meta.get("text") or meta.get("page_content", "")

                    all_results.append({
                        "text": main_text[:3500],
                        "source_type": source_type,
                        "collection": collection_name,
                        "retrieval_score": float(hit.score),
                        "metadata": meta,
                        "matched_via": "semantic"
                    })
            except Exception as e:
                print(f"⚠️ خطا در semantic {source_type}: {e}")

    # ===========================
    # 🔎 حذف تکراری + اولویت‌دهی + برش نهایی
    # ===========================
    seen = set()
    unique_results = []
    for r in all_results:
        h = r["text"][:200]
        if h not in seen:
            seen.add(h)
            unique_results.append(r)

    unique_results.sort(
        key=lambda x: (
            priority.get(x["matched_via"], 99),
            -x["retrieval_score"]
        )
    )

    final_results = unique_results[:top_k_total]
    print(f"✅ نهایی: {len(final_results)} نتیجه")
    return final_results

def rerank_with_cohere_smart(query: str, results: list, top_k: int = 5):
    """
    Reranking هوشمند با Cohere:
      - اگر سندی با matched_via == "law+article" داریم، آنها را در اولویت نگه می‌دارد.
      - بقیه نتایج با مدل rerank-multilingual-v3.0 مرتب می‌شوند.
      - اگر متادیتا نداشتیم، همه چیز صرفاً بر اساس امتیاز Cohere مرتب می‌شود.
    """
    if not results:
        return []

    # گروه‌بندی بر اساس matched_via
    law_article = [r for r in results if r.get("matched_via") == "law+article"]
    others = [r for r in results if r.get("matched_via") != "law+article"]

    # اگر هیچ law+article نداریم → همه را معمولی rerank کن
    if not law_article:
        documents = [r["text"][:4000] for r in results]

        try:
            print(f"🔄 Reranking {len(documents)} سند (بدون law+article)...")
            response = co.rerank(
                query=query,
                documents=documents,
                model="rerank-multilingual-v3.0",
                top_n=min(top_k, len(documents))
            )

            reranked = []
            for item in response.results:
                result = results[item.index].copy()
                result["rerank_score"] = float(item.relevance_score)
                reranked.append(result)

            return reranked
        except Exception as e:
            print(f"⚠️ خطا در reranking: {e}")
            # در صورت خطا، همان top_k اولیه را برگردان
            for r in results[:top_k]:
                r["rerank_score"] = r.get("retrieval_score", 0.0)
            return results[:top_k]

    # اگر law+article داریم → استراتژی ترکیبی
    print(f"🎯 {len(law_article)} نتیجه law+article یافت شد")

    # به نتایج law+article یک امتیاز بالا می‌دهیم تا در صدر بمانند
    for dm in law_article:
        dm["rerank_score"] = 1.0  # بالاتر از هر امتیاز نرمال‌شده‌ی Cohere

    # تعداد جای خالی برای بقیه نتایج
    available_slots = max(top_k - len(law_article), 0)

    reranked_others = []
    if others and available_slots > 0:
        documents = [r["text"][:4000] for r in others]

        try:
            print(f"🔄 Reranking {len(documents)} سند دیگر...")
            response = co.rerank(
                query=query,
                documents=documents,
                model="rerank-multilingual-v3.0",
                top_n=min(available_slots, len(documents))
            )

            for item in response.results:
                result = others[item.index].copy()
                result["rerank_score"] = float(item.relevance_score)
                reranked_others.append(result)
        except Exception as e:
            print(f"⚠️ خطا در reranking: {e}")
            # fallback: استفاده از retrieval_score
            reranked_others = sorted(
                others,
                key=lambda x: x.get("retrieval_score", 0.0),
                reverse=True
            )[:available_slots]

    # ترکیب نهایی: اول law+article، بعد نتایج rerank شده
    final_results = law_article + reranked_others

    # اگر هنوز کمتر از top_k است، از بقیه others طبق retrieval_score پر کن
    if len(final_results) < top_k:
        remaining = [r for r in others if r not in reranked_others]
        remaining_sorted = sorted(
            remaining,
            key=lambda x: x.get("retrieval_score", 0.0),
            reverse=True
        )
        for r in remaining_sorted[: top_k - len(final_results)]:
            r = r.copy()
            r["rerank_score"] = r.get("retrieval_score", 0.0)
            final_results.append(r)

    return final_results[:top_k]

# ==========================================
# تابع اصلی (Main API)
# ==========================================
def legal_rag_retrieve(
    query: str,
    method: str = "auto",
    top_k: int = 5,
    use_rerank: bool = True,
    verbose: bool = False
) -> List[Dict]:
    """
    تابع اصلی برای جستجوی هوشمند در اسناد حقوقی

    Parameters
    ----------
    query : str
        متن سؤال یا query
    method : str, default="auto"
        روش جستجو:
        - "auto": انتخاب خودکار (metadata اگر موجود باشد، وگرنه semantic)
        - "metadata": استفاده از metadata (شماره ماده/اصل، نام قانون)
        - "semantic": فقط جستجوی معنایی
    top_k : int, default=5
        تعداد نتایج برگشتی
    use_rerank : bool, default=True
        استفاده از Cohere reranking
    verbose : bool, default=False
        نمایش جزئیات

    Returns
    -------
    List[Dict]
        لیست نتایج شامل:
        - text: متن سند
        - source_type: نوع منبع (قانون، وحدت رویه، دادنامه)
        - retrieval_score: امتیاز اولیه جستجو
        - rerank_score: امتیاز rerank (اگر use_rerank=True)
        - metadata: اطلاعات متادیتا
        - matched_via: روش match شدن (law+article, law_only, semantic)

    Examples
    --------
    >>> results = legal_rag_retrieve(
    ...     "تعریف اموال غیرمنقول در قانون مدنی چیست؟",
    ...     method="auto",
    ...     top_k=3
    ... )
    >>> print(results[0]['text'][:100])

    >>> results = legal_rag_retrieve(
    ...     "طبق اصل 57 قانون اساسی...",
    ...     method="metadata",
    ...     use_rerank=True
    ... )
    """
    if verbose:
        print(f"📝 Query: {query[:80]}...")

    # ===========================
    # 1. انتخاب روش
    # ===========================
    if method == "auto":
        parsed = parse_question_metadata_simple(query)
        if parsed.get("article_number") or parsed.get("law_name"):
            method = "metadata"
            if verbose:
                print("🎯 روش انتخاب شده: Metadata-aware")
        else:
            method = "semantic"
            if verbose:
                print("🔍 روش انتخاب شده: Semantic search")

    # ===========================
    # 2. Retrieve
    # ===========================
    if method == "metadata":
        # کمی بزرگ‌تر از top_k برای اینکه rerank فضای مانور داشته باشد
        results = retrieve_with_metadata(query, top_k_total=top_k * 4)
    else:
        results = retrieve_semantic_only(query, top_k_total=top_k * 4)

    if verbose:
        print(f"   ✓ Retrieved: {len(results)} documents")

    # ===========================
    # 3. Rerank (optional)
    # ===========================
    if use_rerank and len(results) > 0:
        try:
            results = rerank_with_cohere_smart(query, results, top_k=top_k)
            if verbose:
                print(f"   ✓ Reranked: top {len(results)} documents")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Rerank failed: {e}")
            results = results[:top_k]
    else:
        results = results[:top_k]

    return results

# ==========================================
# تابع کمکی: Format کردن نتایج برای LLM
# ==========================================

def format_results_for_llm(results: List[Dict], include_metadata: bool = True) -> str:
    """
    تبدیل نتایج به فرمت مناسب برای ورودی LLM

    Parameters
    ----------
    results : List[Dict]
        خروجی legal_rag_retrieve()
    include_metadata : bool, default=True
        شامل کردن اطلاعات metadata (نام قانون، شماره ماده/اصل و نوع منبع)

    Returns
    -------
    str
        متن فرمت‌شده برای context در LLM

    Examples
    --------
    >>> results = legal_rag_retrieve("سؤال...", top_k=3)
    >>> context = format_results_for_llm(results)
    >>> prompt = f"سؤال: ...\\n\\nمتن قانون:\\n{context}"
    """
    formatted = []

    # فیلدهایی که اگر باشند، نمایش داده می‌شوند
    FIELDS_ORDER = [
        "article_number", "principle_number",
        "volume", "book", "bab", "chapter", "section", "paragraph",
        "text_section", "title", "issuer",
        "text", "vote_text"
    ]

    for i, result in enumerate(results, 1):
        source_type = result.get("source_type", "نامشخص")
        meta = result.get("metadata", {}) if include_metadata else {}
        law_name = meta.get("law_name")

        # هدر اصلی منبع
        header_lines = [f"[منبع {i}] ({source_type})"]
        if law_name:
            header_lines[0] += f" - {law_name}"

        # شماره ماده / اصل (اگر وجود داشته باشد)
        principle = meta.get("principle_number")
        article = meta.get("article_number")
        if principle is not None:
            header_lines.append(f"شماره_اصل: {principle}")
        if article is not None:
            header_lines.append(f"شماره_ماده: {article}")

        # سایر فیلدها طبق ترتیب مشخص‌شده
        body_lines = []
        if include_metadata:
            for field in FIELDS_ORDER:
                if field in meta:
                    value = meta[field]
                    # متن اصلی (text / vote_text) را محدود می‌کنیم تا خیلی طولانی نشود
                    if field in ("text", "vote_text") and isinstance(value, str):
                        snippet = value.strip()
                        if len(snippet) > 1500:
                            snippet = snippet[:1500] + " ..."
                        body_lines.append(f"{field}: {snippet}")
                    else:
                        body_lines.append(f"{field}: {value}")

        # اگر به هر دلیلی متن اصلی result["text"] با متادیتا متفاوت بود، می‌توانیم آن را هم اضافه کنیم
        # ولی فقط اگر include_metadata=False باشد یا text در meta نباشد
        if not include_metadata or "text" not in meta:
            main_text = result.get("text", "")
            if main_text:
                snippet = main_text.strip()
                if len(snippet) > 1500:
                    snippet = snippet[:1500] + " ..."
                body_lines.append(f"content: {snippet}")

        # مونتاژ قطعه نهایی برای این منبع
        block = "\n".join(header_lines)
        if body_lines:
            block += "\n" + "\n".join(body_lines)

        formatted.append(block)

    return "\n---\n\n".join(formatted)
