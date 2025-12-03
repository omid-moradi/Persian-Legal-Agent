"""
Legal RAG Retrieval System
===========================
سیستم بازیابی هوشمند اطلاعات حقوقی با استفاده از:
- Metadata-aware retrieval
- Semantic search
- Cohere reranking

نویسنده: [نام شما]
تاریخ: دسامبر 2025
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
import cohere


# ==========================================
# تنظیمات پیش‌فرض
# ==========================================

# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_LAWS = "legal_laws"
COLLECTION_UNITY = "legal_votes_unity"
COLLECTION_DADNAMEH = "legal_votes_dadnameh"

# Embedding Model
MODEL_EMBED = "baai/bge-m3"

# مسیرهای پیش‌فرض فایل‌ها
DEFAULT_KEYWORDS_PATH = r"F:\Thesis\project\2-RAG\raw_laws\Unifying Precedent Decisions\keywords.txt"
DEFAULT_QAVANIN_PATH = r"F:\Thesis\project\2-RAG\raw_laws\Unifying Precedent Decisions\qavanin_karbordi.txt"
DEFAULT_METADATA_INDEX_PATH = r"F:\Thesis\project\2-RAG\vector_store\metadata_reverse_index.json"


# ==========================================
# متغیرهای Global
# ==========================================

qdrant = None
client_embed = None
co = None
KEYWORDS = set()
QAVANIN_LIST = []
REVERSE_INDEX = {}


# ==========================================
# تابع Setup (اجباری برای اجرا)
# ==========================================

def setup_legal_rag(
    qdrant_url: str = QDRANT_URL,
    openrouter_api_key: Optional[str] = None,
    cohere_api_key: Optional[str] = None,
    keywords_path: Optional[str] = None,
    qavanin_path: Optional[str] = None,
    metadata_index_path: Optional[str] = None
):
    """
    راه‌اندازی اولیه سیستم Legal RAG
    
    Parameters
    ----------
    qdrant_url : str
        آدرس Qdrant server
    openrouter_api_key : str, optional
        کلید API برای OpenRouter (اگر None باشد، از environment variable استفاده می‌شود)
    cohere_api_key : str, optional
        کلید API برای Cohere (اگر None باشد، از environment variable استفاده می‌شود)
    keywords_path : str, optional
        مسیر فایل keywords.txt
    qavanin_path : str, optional
        مسیر فایل qavanin_karbordi.txt
    metadata_index_path : str, optional
        مسیر فایل metadata_reverse_index.json
    
    Returns
    -------
    bool
        True اگر setup موفق بود
    
    Examples
    --------
    >>> setup_legal_rag(
    ...     openrouter_api_key="YOUR_KEY",
    ...     cohere_api_key="YOUR_KEY"
    ... )
    """
    
    global qdrant, client_embed, co, KEYWORDS, QAVANIN_LIST, REVERSE_INDEX
    
    # ===========================
    # 1. دریافت API Keys
    # ===========================
    if openrouter_api_key is None:
        openrouter_api_key = os.environ.get("OPENROUTER_EMBEDDINGS_API_KEY")
    if cohere_api_key is None:
        cohere_api_key = os.environ.get("COHERE_API_KEY")
    
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_EMBEDDINGS_API_KEY تنظیم نشده است")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY تنظیم نشده است")
    
    # ===========================
    # 2. اتصال به سرویس‌ها
    # ===========================
    qdrant = QdrantClient(url=qdrant_url)
    
    client_embed = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key
    )
    
    co = cohere.Client(api_key=cohere_api_key)
    
    print("✓ Qdrant متصل شد")
    print("✓ Cohere آماده است")
    print("✓ Embedding client آماده است")
    
    # ===========================
    # 3. بارگذاری فایل‌های متادیتا
    # ===========================
    
    # استفاده از مسیرهای پیش‌فرض اگر مسیر داده نشده
    keywords_path = keywords_path or DEFAULT_KEYWORDS_PATH
    qavanin_path = qavanin_path or DEFAULT_QAVANIN_PATH
    metadata_index_path = metadata_index_path or DEFAULT_METADATA_INDEX_PATH
    
    # بارگذاری keywords
    if os.path.exists(keywords_path):
        with open(keywords_path, "r", encoding="utf-8") as f:
            KEYWORDS = set(line.strip() for line in f if line.strip())
        print(f"✓ بارگذاری شد: {len(KEYWORDS)} keyword")
    else:
        print(f"⚠️ فایل keywords یافت نشد: {keywords_path}")
    
    # بارگذاری قوانین کاربردی
    if os.path.exists(qavanin_path):
        with open(qavanin_path, "r", encoding="utf-8") as f:
            qavanin_text = f.read()
            QAVANIN_LIST = [
                line.strip() 
                for line in qavanin_text.split('\n') 
                if line.strip() and (line.strip().startswith('قانون') or line.strip().startswith('لایحه'))
            ]
        print(f"✓ بارگذاری شد: {len(QAVANIN_LIST)} قانون کاربردی")
    else:
        print(f"⚠️ فایل qavanin یافت نشد: {qavanin_path}")
    
    # بارگذاری metadata reverse index
    if os.path.exists(metadata_index_path):
        with open(metadata_index_path, "r", encoding="utf-8") as f:
            REVERSE_INDEX = json.load(f)
        print(f"✓ بارگذاری شد: Metadata Reverse Index ({len(REVERSE_INDEX)} کلیدواژه)")
    else:
        print(f"⚠️ فایل metadata_reverse_index.json یافت نشد: {metadata_index_path}")
        REVERSE_INDEX = {}
    
    print("\n" + "="*80)
    print("✅ Setup کامل شد!")
    print("="*80 + "\n")
    
    return True


# ==========================================
# تابع کمکی: Embed Query
# ==========================================

def embed_query(text: str) -> np.ndarray:
    """
    تبدیل متن به embedding vector
    
    Parameters
    ----------
    text : str
        متن ورودی
    
    Returns
    -------
    np.ndarray
        بردار embedding
    """
    
    if client_embed is None:
        raise RuntimeError("لطفاً ابتدا setup_legal_rag() را اجرا کنید")
    
    response = client_embed.embeddings.create(
        input=text,
        model=MODEL_EMBED
    )
    
    return np.array(response.data[0].embedding)

# ==========================================
# Parse Metadata & Helper Functions
# ==========================================

# تبدیل اعداد فارسی به انگلیسی (کامل)
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
    """
    تبدیل عدد فارسی (ترکیبی) به عدد انگلیسی
    
    Parameters
    ----------
    text : str
        عدد فارسی (مثل "یکصد و دوازده")
    
    Returns
    -------
    int
        عدد انگلیسی (مثل 112)
    
    Examples
    --------
    >>> farsi_word_to_number("یکصد و دوازده")
    112
    >>> farsi_word_to_number("بیست و سوم")
    23
    """
    text = text.strip().replace('‌', ' ')
    
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
    """
    تبدیل شماره فارسی (ساده یا ترکیبی) به انگلیسی
    
    Parameters
    ----------
    text : str
        شماره به فارسی یا انگلیسی
    
    Returns
    -------
    str
        شماره به انگلیسی
    """
    text = text.strip()
    
    if text.isdigit():
        return text
    
    try:
        number = farsi_word_to_number(text)
        return str(number) if number else text
    except:
        return text


def parse_question_metadata(question: str) -> dict:
    """
    استخراج متادیتا از سؤال حقوقی
    
    این تابع موارد زیر را از سؤال استخراج می‌کند:
    - نام قانون
    - شماره ماده/اصل (با پشتیبانی اعداد فارسی ترکیبی)
    - نوع (ماده، اصل، تبصره، ...)
    - ساختار (فصل، باب، ...)
    - حوزه حقوقی
    
    Parameters
    ----------
    question : str
        متن سؤال
    
    Returns
    -------
    dict
        دیکشنری حاوی metadata استخراج شده
    
    Examples
    --------
    >>> parse_question_metadata("طبق اصل 57 قانون اساسی...")
    {'law_name': 'قانون اساسی جمهوری اسلامی ایران',
     'article_number': '57',
     'article_type': 'اصل',
     ...}
    """
    
    if client_embed is None:
        raise RuntimeError("لطفاً ابتدا setup_legal_rag() را اجرا کنید")
    
    q = question.strip()
    result = {
        "law_name": None,
        "article_number": None,
        "article_type": None,
        "domain": None,
        "structure_info": {}
    }
    
    # 1. جستجوی نام قانون
    for law in QAVANIN_LIST:
        law_clean = law.replace('لایحه قانونی', '').replace('قانون', '').strip()
        law_keywords = [w for w in law_clean.split() if len(w) > 3][:4]
        
        matches = sum(1 for kw in law_keywords if kw in q)
        if matches >= 2 or (len(law_keywords) == 1 and law_keywords[0] in q):
            result["law_name"] = law
            break
    
    # 2. استخراج شماره (با پشتیبانی اعداد ترکیبی)
    patterns = [
        (r'اصل\s+([آ-ی\s]+(?:م|مین)?|\d+)', 'اصل'),
        (r'(?:ماده|مواد)\s+(\d+(?:\s*مکرر)?(?:\s*تا\s*\d+)?(?:\s*و\s*\d+)*)', 'ماده'),
        (r'تبصره\s+([آ-ی\s]+(?:م)?|\d+)', 'تبصره'),
        (r'بند\s+([آ-ی\s]+(?:م)?|[آ-ی]|\d+)', 'بند'),
        (r'فقره\s+([آ-ی\s]+(?:م)?|\d+)', 'فقره'),
    ]
    
    for pattern, article_type in patterns:
        match = re.search(pattern, q)
        if match:
            raw_number = match.group(1).strip()
            result["article_number"] = normalize_article_number(raw_number)
            result["article_type"] = article_type
            break
    
    # 3. استخراج ساختار
    structure_patterns = [
        (r'فصل\s+([آ-ی\s]+(?:م)?|\d+)(?:\s*[-:]\s*([^\n،\.؛]+))?', 'فصل'),
        (r'باب\s+([آ-ی\s]+(?:م)?|\d+)(?:\s*[-:]\s*([^\n،\.؛]+))?', 'باب'),
        (r'بخش\s+([آ-ی\s]+(?:م)?|\d+)(?:\s*[-:]\s*([^\n،\.؛]+))?', 'بخش'),
        (r'کتاب\s+([آ-ی\s]+(?:م)?|\d+)(?:\s*[-:]\s*([^\n،\.؛]+))?', 'کتاب'),
        (r'جلد\s+([آ-ی\s]+(?:م)?|\d+)(?:\s*[-:]\s*([^\n،\.؛]+))?', 'جلد'),
        (r'مبحث\s+([آ-ی\s]+(?:م)?|\d+)(?:\s*[-:]\s*([^\n،\.؛]+))?', 'مبحث'),
    ]
    
    for pattern, struct_type in structure_patterns:
        match = re.search(pattern, q)
        if match:
            raw_number = match.group(1).strip()
            number = normalize_article_number(raw_number)
            title = match.group(2).strip() if match.lastindex >= 2 and match.group(2) else None
            result["structure_info"][struct_type] = {
                "number": number,
                "title": title
            }
    
    # 4. جستجوی حوزه
    for keyword in KEYWORDS:
        if keyword in q:
            result["domain"] = keyword
            break
    
    if not result["domain"] and result["law_name"]:
        if "مجازات" in result["law_name"] or "کیفری" in result["law_name"]:
            result["domain"] = "کیفری"
        elif "اساسی" in result["law_name"]:
            result["domain"] = "اساسی"
        elif "مدنی" in result["law_name"]:
            result["domain"] = "مدنی"
    
    return result

# ==========================================
# Semantic-Only Retrieval
# ==========================================

def retrieve_semantic_only(query_text: str, top_laws: int = 10, top_unity: int = 5, top_dadnameh: int = 5) -> List[Dict]:
    """
    جستجوی معنایی خالص (بدون استفاده از metadata filtering)
    
    این تابع بدون فیلتر metadata، صرفاً بر اساس شباهت معنایی
    embedding ها در سه collection جستجو می‌کند.
    
    Parameters
    ----------
    query_text : str
        متن سؤال
    top_laws : int, default=10
        تعداد نتایج از collection قوانین
    top_unity : int, default=5
        تعداد نتایج از collection وحدت رویه
    top_dadnameh : int, default=5
        تعداد نتایج از collection دادنامه‌ها
    
    Returns
    -------
    List[Dict]
        لیست نتایج شامل:
        - text: متن سند
        - source_type: نوع منبع
        - retrieval_score: امتیاز شباهت
        - metadata: اطلاعات metadata
        - matched_via: "semantic_only"
    
    Examples
    --------
    >>> results = retrieve_semantic_only(
    ...     "تعریف اموال غیرمنقول",
    ...     top_laws=5,
    ...     top_unity=2
    ... )
    >>> print(len(results))
    7
    """
    
    if qdrant is None or client_embed is None:
        raise RuntimeError("لطفاً ابتدا setup_legal_rag() را اجرا کنید")
    
    # Embed کردن سؤال
    query_vec = embed_query(query_text).tolist()
    
    all_results = []
    
    # 1. جستجو در قوانین
    try:
        hits_laws = qdrant.query_points(
            collection_name=COLLECTION_LAWS,
            query=query_vec,
            limit=top_laws,
            with_payload=True
        ).points
        
        for hit in hits_laws:
            all_results.append({
                "text": hit.payload.get("page_content", "")[:4000],
                "source_type": "قانون",
                "retrieval_score": float(hit.score),
                "metadata": hit.payload,
                "matched_via": "semantic_only"
            })
    except Exception as e:
        print(f"⚠️ خطا در جستجوی قوانین: {e}")
    
    # 2. جستجو در آراء وحدت رویه
    try:
        hits_unity = qdrant.query_points(
            collection_name=COLLECTION_UNITY,
            query=query_vec,
            limit=top_unity,
            with_payload=True
        ).points
        
        for hit in hits_unity:
            all_results.append({
                "text": hit.payload.get("page_content", "")[:4000],
                "source_type": "وحدت رویه",
                "retrieval_score": float(hit.score),
                "metadata": hit.payload,
                "matched_via": "semantic_only"
            })
    except Exception as e:
        print(f"⚠️ خطا در جستجوی وحدت رویه: {e}")
    
    # 3. جستجو در دادنامه‌ها
    try:
        hits_dadnameh = qdrant.query_points(
            collection_name=COLLECTION_DADNAMEH,
            query=query_vec,
            limit=top_dadnameh,
            with_payload=True
        ).points
        
        for hit in hits_dadnameh:
            all_results.append({
                "text": hit.payload.get("page_content", "")[:4000],
                "source_type": "دادنامه",
                "retrieval_score": float(hit.score),
                "metadata": hit.payload,
                "matched_via": "semantic_only"
            })
    except Exception as e:
        print(f"⚠️ خطا در جستجوی دادنامه‌ها: {e}")
    
    # مرتب‌سازی بر اساس score (نزولی)
    all_results.sort(key=lambda x: x["retrieval_score"], reverse=True)
    
    return all_results

# ==========================================
# Metadata-Aware Retrieval
# ==========================================

def retrieve_with_metadata(query_text: str, top_k_per_source: int = 10) -> List[Dict]:
    """
    جستجوی هوشمند با استفاده از metadata (شماره ماده/اصل، نام قانون و عناوین)
    
    Parameters
    ----------
    query_text : str
        متن سؤال
    top_k_per_source : int, default=10
        تعداد نتایج در هر منبع برای بخش semantic عمومی
    
    Returns
    -------
    List[Dict]
        لیست نتایج یکتا، مرتب‌شده بر اساس اولویت و امتیاز
    """
    
    if qdrant is None or client_embed is None:
        raise RuntimeError("لطفاً ابتدا setup_legal_rag() را اجرا کنید")
    
    parsed_metadata = parse_question_metadata(query_text)
    query_vec = embed_query(query_text).tolist()
    all_results: List[Dict] = []
    
    # ===========================
    # بخش 1: فیلتر مستقیم شماره اصل/ماده
    # ===========================
    direct_match_found = False
    
    if parsed_metadata.get("article_number") and parsed_metadata.get("article_type"):
        article_num = parsed_metadata["article_number"]
        article_type = parsed_metadata["article_type"]
        
        print(f"🎯 جستجوی مستقیم: {article_type} {article_num}")
        
        if article_type == "اصل":
            collection = COLLECTION_LAWS
            field_name = "principle_number"
        elif article_type == "ماده":
            collection = COLLECTION_LAWS
            field_name = "article_number"
        else:
            collection = None
            field_name = None
        
        if collection and field_name:
            try:
                # تبدیل به float برای استفاده در Range
                try:
                    article_num_float = float(article_num)
                except (ValueError, TypeError):
                    article_num_float = None
                
                filters = []
                if article_num_float is not None:
                    # استفاده از Range به‌جای MatchValue
                    filters.append(
                        FieldCondition(
                            key=field_name,
                            range=Range(
                                gte=article_num_float - 0.1,
                                lte=article_num_float + 0.1
                            )
                        )
                    )
                
                # فیلتر قانون اساسی برای اصول (در صورت عدم ذکر صریح نام قانون)
                if article_type == "اصل" and not parsed_metadata.get("law_name"):
                    if "قانون اساسی" in query_text or "اساسی" in query_text:
                        filters.append(
                            FieldCondition(
                                key="law_name",
                                match=MatchValue(value="قانون اساسی جمهوری اسلامی ایران")
                            )
                        )
                elif parsed_metadata.get("law_name"):
                    filters.append(
                        FieldCondition(
                            key="law_name",
                            match=MatchValue(value=parsed_metadata["law_name"])
                        )
                    )
                
                if filters:
                    hits = qdrant.query_points(
                        collection_name=collection,
                        query=query_vec,
                        query_filter=Filter(must=filters),
                        limit=5,
                        with_payload=True
                    ).points
                    
                    print(f"   ✓ یافت شد: {len(hits)} سند")
                    
                    if len(hits) > 0:
                        direct_match_found = True
                        for hit in hits:
                            all_results.append({
                                "text": hit.payload.get("page_content", "")[:4000],
                                "source_type": "قانون",
                                "retrieval_score": float(hit.score),
                                "metadata": hit.payload,
                                "matched_via": f"direct: {article_type} {article_num}"
                            })
                    else:
                        print(f"   ⚠️ هیچ سندی با {field_name}={article_num} پیدا نشد")
            
            except Exception as e:
                print(f"⚠️ خطا در فیلتر {article_type}: {e}")
    
    # ===========================
    # Fallback: اگر direct match پیدا نشد ولی نام قانون داریم
    # ===========================
    if not direct_match_found and parsed_metadata.get("law_name"):
        law_name = parsed_metadata["law_name"]
        print(f"🔄 Fallback: جستجو در {law_name[:50]}...")
        
        try:
            hits = qdrant.query_points(
                collection_name=COLLECTION_LAWS,
                query=query_vec,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="law_name",
                            match=MatchValue(value=law_name)
                        )
                    ]
                ),
                limit=8,
                with_payload=True
            ).points
            
            print(f"   ✓ یافت شد: {len(hits)} سند")
            
            for hit in hits:
                all_results.append({
                    "text": hit.payload.get("page_content", "")[:4000],
                    "source_type": "قانون",
                    "retrieval_score": float(hit.score),
                    "metadata": hit.payload,
                    "matched_via": f"law_fallback: {law_name[:30]}..."
                })
        
        except Exception as e:
            print(f"⚠️ خطا در fallback: {e}")
    
    # ===========================
    # بخش 2: metadata titles (استفاده از REVERSE_INDEX)
    # ===========================
    if parsed_metadata.get("matched_metadata"):
        print(f"🎯 یافت شد {len(parsed_metadata['matched_metadata'])} metadata title")
        
        for matched in parsed_metadata["matched_metadata"][:2]:
            try:
                filters = []
                
                if matched.get("law_name"):
                    filters.append(
                        FieldCondition(
                            key="law_name",
                            match=MatchValue(value=matched["law_name"])
                        )
                    )
                
                if matched.get("metadata_field") and matched.get("full_text"):
                    filters.append(
                        FieldCondition(
                            key=matched["metadata_field"],
                            match=MatchValue(value=matched["full_text"])
                        )
                    )
                
                if filters:
                    hits = qdrant.query_points(
                        collection_name=matched["collection"],
                        query=query_vec,
                        query_filter=Filter(must=filters),
                        limit=3,
                        with_payload=True
                    ).points
                    
                    for hit in hits:
                        all_results.append({
                            "text": hit.payload.get("page_content", "")[:4000],
                            "source_type": (
                                "قانون" if matched["collection"] == COLLECTION_LAWS
                                else "وحدت رویه" if matched["collection"] == COLLECTION_UNITY
                                else "دادنامه"
                            ),
                            "retrieval_score": float(hit.score),
                            "metadata": hit.payload,
                            "matched_via": f"title: {matched['title'][:30]}..."
                        })
            
            except Exception:
                # در صورت خطا در یک title خاص، ادامه می‌دهیم
                pass
    
    # ===========================
    # بخش 3: semantic عمومی
    # ===========================
    collections = [
        (COLLECTION_LAWS, "قانون", top_k_per_source),
        (COLLECTION_UNITY, "وحدت رویه", max(3, top_k_per_source // 2)),
        (COLLECTION_DADNAMEH, "دادنامه", max(3, top_k_per_source // 2))
    ]
    
    for collection_name, source_type, limit in collections:
        try:
            hits = qdrant.query_points(
                collection_name=collection_name,
                query=query_vec,
                limit=limit,
                with_payload=True
            ).points
            
            for hit in hits:
                all_results.append({
                    "text": hit.payload.get("page_content", "")[:4000],
                    "source_type": source_type,
                    "retrieval_score": float(hit.score),
                    "metadata": hit.payload,
                    "matched_via": "semantic_general"
                })
        
        except Exception as e:
            print(f"⚠️ خطا در {source_type}: {e}")
    
    # ===========================
    # حذف تکراری و مرتب‌سازی
    # ===========================
    seen_texts = set()
    unique_results: List[Dict] = []
    
    for result in all_results:
        text_hash = result["text"][:200]
        if text_hash not in seen_texts:
            seen_texts.add(text_hash)
            unique_results.append(result)
    
    priority_order = {
        "direct": 0,
        "title": 1,
        "law_fallback": 2,
        "semantic_general": 3
    }
    
    unique_results.sort(
        key=lambda x: (
            priority_order.get(x["matched_via"].split(":")[0], 99),
            -x["retrieval_score"]
        )
    )
    
    return unique_results

# ==========================================
# Reranking با استراتژی هوشمند
# ==========================================

def rerank_with_cohere_smart(query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Reranking هوشمند نتایج که در آن:
    - اسنادی که به صورت مستقیم با شماره ماده/اصل match شده‌اند (direct)
      در اولویت حفظ می‌شوند.
    - سایر اسناد با استفاده از مدل rerank چندزبانه Cohere مرتب‌سازی مجدد می‌شوند.
    
    Parameters
    ----------
    query : str
        متن سؤال
    results : List[Dict]
        لیست نتایج اولیه (خروجی retrieve_with_metadata یا retrieve_semantic_only)
    top_k : int, default=5
        تعداد نتایج نهایی مورد نیاز
    
    Returns
    -------
    List[Dict]
        لیست نتایج rerank شده، حداکثر به طول top_k، شامل فیلد جدید rerank_score (در صورت موفقیت)
    """
    
    if not results:
        return []
    
    if co is None:
        raise RuntimeError("لطفاً ابتدا setup_legal_rag() را اجرا کنید تا Cohere مقداردهی شود")
    
    # جدا کردن direct matches
    direct_matches = [r for r in results if "direct" in r.get("matched_via", "")]
    other_results = [r for r in results if "direct" not in r.get("matched_via", "")]
    
    # اگر direct match نداریم، rerank ساده روی همه‌ی نتایج
    if not direct_matches:
        documents = [r["text"][:4000] for r in results]
        
        try:
            print(f"🔄 Reranking {len(documents)} سند...")
            response = co.rerank(
                query=query,
                documents=documents,
                model="rerank-multilingual-v3.0",
                top_n=min(top_k, len(documents))
            )
            
            reranked: List[Dict] = []
            for item in response.results:
                result = results[item.index].copy()
                result["rerank_score"] = float(item.relevance_score)
                reranked.append(result)
            
            return reranked
        
        except Exception as e:
            print(f"⚠️ خطا در reranking: {e}")
            return results[:top_k]
    
    # اگر direct match داریم، استراتژی ترکیبی
    print(f"🎯 {len(direct_matches)} direct match یافت شد")
    
    # Rerank کردن بقیه نتایج
    if other_results:
        documents = [r["text"][:4000] for r in other_results]
        
        try:
            print(f"🔄 Reranking {len(documents)} سند دیگر...")
            response = co.rerank(
                query=query,
                documents=documents,
                model="rerank-multilingual-v3.0",
                top_n=min(max(top_k - len(direct_matches), 0), len(documents))
            )
            
            reranked_others: List[Dict] = []
            for item in response.results:
                result = other_results[item.index].copy()
                result["rerank_score"] = float(item.relevance_score)
                reranked_others.append(result)
        
        except Exception as e:
            print(f"⚠️ خطا در reranking: {e}")
            reranked_others = other_results[:max(top_k - len(direct_matches), 0)]
    else:
        reranked_others = []
    
    # ترکیب: direct matches اول، سپس نتایج reranked
    for dm in direct_matches:
        # امتیاز پیش‌فرض بالا برای direct matches
        dm.setdefault("rerank_score", 0.9)
    
    final_results = direct_matches + reranked_others
    
    # هشدار در صورت وجود نتیجه‌ی دیگر با امتیاز بسیار بالاتر
    if reranked_others and reranked_others[0].get("rerank_score", 0) > 0.7:
        print(f"   ⚠️ توجه: نتیجه دیگری با امتیاز بالا ({reranked_others[0]['rerank_score']:.3f}) یافت شد")
        print("      ممکن است نتیجه مرتبط‌تری نسبت به direct match وجود داشته باشد.")
    
    return final_results[:top_k]

"""
Legal RAG Retrieval System
===========================

سیستم جستجوی هوشمند برای پرسش‌وپاسخ حقوقی با استفاده از:
- Metadata-aware retrieval
- Semantic search
- Cohere reranking

نویسنده: OMID Moradi
تاریخ: دسامبر 2025
"""

import time
from typing import List, Dict, Optional


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
    تابع اصلی برای جستجوی هوشمند در اسناد حقوقی.

    Parameters
    ----------
    query : str
        متن سؤال یا query.
    method : str, default="auto"
        روش جستجو:
        - "auto": انتخاب خودکار (استفاده از metadata در صورت وجود، در غیر این صورت semantic).
        - "metadata": استفاده از metadata (شماره ماده/اصل، نام قانون).
        - "semantic": فقط جستجوی معنایی.
    top_k : int, default=5
        تعداد نتایج نهایی.
    use_rerank : bool, default=True
        استفاده از Cohere reranking.
    verbose : bool, default=False
        نمایش جزئیات فرایند.

    Returns
    -------
    List[Dict]
        لیست نتایج شامل:
        - text: متن سند.
        - source_type: نوع منبع (قانون، وحدت رویه، دادنامه).
        - retrieval_score: امتیاز retrieval.
        - rerank_score: امتیاز rerank (اگر use_rerank=True و موفق).
        - metadata: اطلاعات متادیتا.
        - matched_via: روش match شدن.
    """
    if verbose:
        print(f"📝 Query: {query[:80]}...")

    # 1. انتخاب روش
    if method == "auto":
        parsed = parse_question_metadata(query)
        if parsed.get("article_number") or parsed.get("law_name"):
            method = "metadata"
            if verbose:
                print("🎯 روش انتخاب‌شده: Metadata-aware")
        else:
            method = "semantic"
            if verbose:
                print("🔍 روش انتخاب‌شده: Semantic search")

    # 2. Retrieve
    if method == "metadata":
        results = retrieve_with_metadata(query, top_k_per_source=top_k * 3)
    else:
        results = retrieve_semantic_only(
            query,
            top_laws=top_k * 2,
            top_unity=max(2, top_k // 2),
            top_dadnameh=max(2, top_k // 2),
        )

    if verbose:
        print(f"   ✓ Retrieved: {len(results)} documents")

    # 3. Rerank (optional)
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
    تبدیل نتایج به فرمت مناسب برای ورودی LLM.

    Parameters
    ----------
    results : List[Dict]
        خروجی legal_rag_retrieve().
    include_metadata : bool, default=True
        شامل کردن اطلاعات metadata در header هر منبع.

    Returns
    -------
    str
        متن فرمت‌شده برای context در LLM.
    """
    formatted: List[str] = []

    for i, result in enumerate(results, 1):
        text = result["text"]

        if include_metadata:
            meta = result["metadata"]
            law_name = meta.get("law_name", "نامشخص")

            principle = meta.get("principle_number")
            article = meta.get("article_number")

            if principle:
                num_str = f"اصل {int(principle)}"
            elif article:
                num_str = f"ماده {article}"
            else:
                num_str = ""

            header = f"[منبع {i}] {law_name}"
            if num_str:
                header += f" - {num_str}"

            formatted.append(f"{header}\n{text}\n")
        else:
            formatted.append(f"[منبع {i}]\n{text}\n")

    return "\n---\n\n".join(formatted)
