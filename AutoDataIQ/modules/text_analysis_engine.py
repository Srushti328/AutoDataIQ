"""Rule-based text analysis, sentiment classification, topic identification, and insight generation.

Phase 17 module for AutoDataIQ Unstructured Document Processing workflow.
All analysis is deterministic, transparent, and completely isolated from structured tabular data.
"""

from collections import Counter
import re
from typing import Any, MutableMapping, Optional, Sequence


# =====================================================================
# Constants and Dictionaries
# =====================================================================

DOCUMENT_ANALYSIS_STATE_KEYS = (
    "doc_analysis_source_id",
    "doc_analysis_text_source",
    "doc_analysis_result",
)

STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "will", "just", "also", "get", "got", "can", "may", "might", "one", "two", "well", "like", "etc",
})

# Transparent positive keyword lexicon (rule-based heuristic)
POSITIVE_WORDS = frozenset({
    "excellent", "good", "happy", "satisfied", "smooth", "helpful", "fast", "comfortable", "reliable", "great",
    "easy", "love", "recommend", "quick", "perfect", "wonderful", "awesome", "friendly", "prompt", "pleased",
    "efficient", "superb", "exceptional", "delighted", "impressed", "fantastic", "positive", "valuable",
    "convenient", "flawless", "responsive", "affordable", "high-quality", "seamless", "courteous", "neat",
    "terrific", "brilliant", "outstanding", "fabulous", "fine", "enjoyed", "liked", "appreciated", "satisfactory",
    "intuitive", "durable", "clean", "premium", "resolved",
})

# Transparent negative keyword lexicon (rule-based heuristic)
NEGATIVE_WORDS = frozenset({
    "bad", "poor", "delayed", "damaged", "failed", "slow", "difficult", "problem", "error", "disappointed",
    "expensive", "terrible", "awful", "worst", "broken", "rude", "useless", "frustrated", "unhelpful",
    "complaint", "issue", "bug", "crash", "crashing", "glitch", "horrible", "annoying", "waste", "defective",
    "defect", "dissatisfied", "disappointing", "subpar", "faulty", "unresponsive", "mess", "overpriced", "hate",
    "hated", "unacceptable", "confusing", "painful", "struggled", "frustrating", "cancellation", "cheap",
})

# Rule-based topic taxonomy with comprehensive keyword triggers
TOPIC_TAXONOMY: dict[str, frozenset[str]] = {
    "Delivery and Shipping": frozenset({
        "delivery", "shipping", "shipment", "package", "courier", "tracking", "dispatch",
        "arrive", "arrived", "arrival", "transit", "delayed", "late", "box", "carrier", "delivered",
        "parcel", "speed", "post", "freight", "transport",
    }),
    "Billing and Payment": frozenset({
        "billing", "payment", "charge", "invoice", "refund", "price", "cost", "expensive",
        "fee", "subscription", "receipt", "credit", "card", "wallet", "charged", "paid",
        "overpriced", "discount", "transaction", "checkout", "money", "pricing", "bill",
    }),
    "Product Quality": frozenset({
        "quality", "product", "item", "material", "materials", "durable", "durability", "damaged",
        "broken", "defect", "defective", "build", "performance", "feature", "hardware", "screen",
        "size", "color", "condition", "faulty", "flimsy", "reliable", "casing",
    }),
    "Customer Support": frozenset({
        "support", "agent", "representative", "help", "helpful", "service", "unhelpful",
        "chat", "call", "ticket", "staff", "response", "email", "assistance", "polite",
        "rude", "contact", "executive", "courteous", "unresponsive",
    }),
    "Mobile Application": frozenset({
        "app", "application", "mobile", "ios", "android", "crash", "crashing", "update",
        "notification", "notifications", "install", "download", "login", "touch", "phone",
        "apk", "playstore", "appstore",
    }),
    "Website Experience": frozenset({
        "website", "web", "site", "page", "pages", "portal", "browser", "link",
        "cart", "button", "load", "loading", "ui", "interface", "navigation", "online",
        "server", "url", "desktop",
    }),
    "General Feedback": frozenset({
        "experience", "feedback", "overall", "service", "company", "recommend", "satisfaction",
        "opinion", "review", "general", "impression", "brand", "standard", "expectations", "satisfactory",
    }),
}


# =====================================================================
# Tokenization & Splitting Helpers
# =====================================================================

def extract_words(text: str) -> list[str]:
    """Extract lowercased alphanumeric words from text, stripping isolated punctuation.

    Preserves standard hyphenated words (e.g. high-quality) and contractions (e.g. don't)
    while treating multiple dashes (e.g. ---) or decorative punctuation as word boundaries.
    """
    if not text:
        return []
    # Replace runs of 2+ hyphens, underscores, or apostrophes with whitespace
    sanitized = re.sub(r"[-_']{2,}", " ", text)
    # Match words: alphanumeric tokens optionally joined by single interior hyphen or apostrophe
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:['-_][a-zA-Z0-9]+)*\b", sanitized)
    return [t.lower() for t in tokens if t]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly based on punctuation boundaries and newlines."""
    if not text:
        return []
    # Split on sentence ending punctuation followed by whitespace, or line breaks
    raw_segments = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [s.strip() for s in raw_segments if s and s.strip()]
    return sentences


def split_paragraphs(text: str) -> list[str]:
    """Split text into distinct non-empty paragraphs."""
    if not text:
        return []
    raw_paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p and p.strip()]
    # If no double newline exists, fall back to non-empty lines
    if not paragraphs:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    return paragraphs


# =====================================================================
# A. Text Statistics
# =====================================================================

def calculate_text_statistics(text: str) -> dict[str, Any]:
    """Compute structural metrics and lexical statistics for document text.

    Metrics:
    - total_characters: Total character count including spaces.
    - non_whitespace_characters: Total non-whitespace characters.
    - total_words: Total word token count.
    - total_sentences: Total non-empty sentence count.
    - total_paragraphs: Total paragraph count.
    - average_word_length: Mean character length per word.
    - average_sentence_length: Mean word count per sentence.
    - most_frequent_words: Top 10 raw words (normalized).
    """
    text = text or ""
    total_characters = len(text)
    non_whitespace_characters = len(re.sub(r"\s+", "", text))
    words = extract_words(text)
    total_words = len(words)
    sentences = split_sentences(text)
    total_sentences = len(sentences)
    paragraphs = split_paragraphs(text)
    total_paragraphs = len(paragraphs)

    avg_word_length = round(sum(len(w) for w in words) / total_words, 2) if total_words > 0 else 0.0
    avg_sentence_length = round(total_words / total_sentences, 2) if total_sentences > 0 else 0.0

    raw_word_counter = Counter(words)
    most_frequent_words = [
        {"word": word, "count": count}
        for word, count in raw_word_counter.most_common(10)
    ]

    return {
        "total_characters": total_characters,
        "non_whitespace_characters": non_whitespace_characters,
        "total_words": total_words,
        "total_sentences": total_sentences,
        "total_paragraphs": total_paragraphs,
        "average_word_length": avg_word_length,
        "average_sentence_length": avg_sentence_length,
        "most_frequent_words": most_frequent_words,
    }


# =====================================================================
# B. Keyword Frequency Analysis
# =====================================================================

def analyze_keyword_frequency(text: str, top_n: int = 15) -> list[dict[str, Any]]:
    """Extract frequently occurring meaningful keywords with stop-word removal.

    - Normalizes case to lowercase.
    - Strips surrounding punctuation.
    - Excludes standard English stopwords.
    - Excludes single-character or purely numeric tokens.
    - Returns top N keywords sorted by descending frequency.
    """
    words = extract_words(text)
    meaningful_words = [
        w for w in words
        if w not in STOP_WORDS and len(w) > 1 and not w.isdigit()
    ]

    total_meaningful = len(meaningful_words)
    if total_meaningful == 0:
        return []

    word_counts = Counter(meaningful_words)
    top_keywords = []
    for word, count in word_counts.most_common(top_n):
        percentage = round((count / total_meaningful) * 100, 2)
        top_keywords.append({
            "keyword": word,
            "frequency": count,
            "percentage": percentage,
        })

    return top_keywords


# =====================================================================
# C. Rule-Based Sentiment Analysis
# =====================================================================

def analyze_sentiment_rules(text: str) -> dict[str, Any]:
    """Perform transparent, rule-based sentiment classification on document sentences.

    Methodology:
    - Splits text into distinct sentences.
    - Matches lowercased word tokens against POSITIVE_WORDS and NEGATIVE_WORDS.
    - Computes sentence net score = positive_matches - negative_matches.
    - Classifies sentence:
        * Positive: positive_matches > negative_matches
        * Negative: negative_matches > positive_matches
        * Neutral: positive_matches == negative_matches (including zero matches)
    - Compiles document-level distribution and overall sentiment rating.

    Limitation Note:
    - This rule-based keyword matching heuristic does not capture grammatical negation
      (e.g., 'not great'), sarcasm, or subtle context.
    """
    sentences = split_sentences(text)
    if not sentences:
        return {
            "overall_sentiment": "Neutral",
            "score": 0,
            "sentence_counts": {"Positive": 0, "Negative": 0, "Neutral": 0},
            "sentence_percentages": {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0},
            "total_sentences": 0,
            "positive_keywords_found": [],
            "negative_keywords_found": [],
            "classified_sentences": [],
            "explanation": "No sentences found to analyze.",
            "limitations": "Rule-based keyword matching does not detect context, sarcasm, or negation.",
        }

    classified_sentences = []
    all_positive_matches: list[str] = []
    all_negative_matches: list[str] = []

    pos_count = 0
    neg_count = 0
    neu_count = 0

    for idx, sentence in enumerate(sentences, 1):
        words = extract_words(sentence)
        pos_words = [w for w in words if w in POSITIVE_WORDS]
        neg_words = [w for w in words if w in NEGATIVE_WORDS]

        all_positive_matches.extend(pos_words)
        all_negative_matches.extend(neg_words)

        pos_score = len(pos_words)
        neg_score = len(neg_words)

        if pos_score > neg_score:
            sentiment = "Positive"
            pos_count += 1
        elif neg_score > pos_score:
            sentiment = "Negative"
            neg_count += 1
        else:
            sentiment = "Neutral"
            neu_count += 1

        classified_sentences.append({
            "index": idx,
            "sentence": sentence,
            "sentiment": sentiment,
            "pos_score": pos_score,
            "neg_score": neg_score,
            "pos_words": pos_words,
            "neg_words": neg_words,
        })

    total_sentences = len(sentences)
    pos_pct = round((pos_count / total_sentences) * 100, 1)
    neg_pct = round((neg_count / total_sentences) * 100, 1)
    neu_pct = round((neu_count / total_sentences) * 100, 1)

    # Document-level overall sentiment determination
    net_overall_score = len(all_positive_matches) - len(all_negative_matches)
    if pos_count > neg_count and (pos_count >= neu_count or net_overall_score > 0):
        overall_sentiment = "Positive"
    elif neg_count > pos_count and (neg_count >= neu_count or net_overall_score < 0):
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    pos_counter = Counter(all_positive_matches)
    neg_counter = Counter(all_negative_matches)

    pos_found = [{"word": w, "count": c} for w, c in pos_counter.most_common(10)]
    neg_found = [{"word": w, "count": c} for w, c in neg_counter.most_common(10)]

    explanation = (
        f"Classified {total_sentences} sentence(s): {pos_count} Positive ({pos_pct}%), "
        f"{neg_count} Negative ({neg_pct}%), and {neu_count} Neutral ({neu_pct}%). "
        f"Total positive keyword triggers: {len(all_positive_matches)}; "
        f"Total negative keyword triggers: {len(all_negative_matches)}. "
        f"Net document score: {net_overall_score:+d}."
    )

    return {
        "overall_sentiment": overall_sentiment,
        "score": net_overall_score,
        "sentence_counts": {
            "Positive": pos_count,
            "Negative": neg_count,
            "Neutral": neu_count,
        },
        "sentence_percentages": {
            "Positive": pos_pct,
            "Negative": neg_pct,
            "Neutral": neu_pct,
        },
        "total_sentences": total_sentences,
        "positive_keywords_found": pos_found,
        "negative_keywords_found": neg_found,
        "classified_sentences": classified_sentences,
        "explanation": explanation,
        "limitations": (
            "Rule-based sentiment analysis relies strictly on predefined keyword triggers. "
            "It does not account for complex syntactic context, double negations (e.g. 'not bad'), "
            "or irony/sarcasm."
        ),
    }


# =====================================================================
# D. Topic Identification
# =====================================================================

def identify_topics(text: str) -> dict[str, Any]:
    """Identify discussion topics using rule-based keyword groups.

    Topics evaluated:
    - Delivery and Shipping
    - Billing and Payment
    - Product Quality
    - Customer Support
    - Mobile Application
    - Website Experience
    - General Feedback

    Sentences may match multiple topics when relevant.
    Returns matched topics ranked by frequency with supporting keywords and sample sentences.
    """
    sentences = split_sentences(text)
    if not sentences:
        return {
            "topics": [],
            "total_topic_matches": 0,
            "summary": "No text available for topic identification.",
        }

    # Tracking per topic
    topic_data: dict[str, dict[str, Any]] = {
        name: {
            "topic": name,
            "match_count": 0,
            "sentence_count": 0,
            "matched_keywords": Counter(),
            "sample_sentences": [],
        }
        for name in TOPIC_TAXONOMY
    }

    total_topic_matches = 0

    for sentence in sentences:
        words = extract_words(sentence)
        sentence_word_set = set(words)

        for topic_name, keywords in TOPIC_TAXONOMY.items():
            matched_in_sentence = [w for w in words if w in keywords]
            if matched_in_sentence:
                count = len(matched_in_sentence)
                topic_data[topic_name]["match_count"] += count
                topic_data[topic_name]["sentence_count"] += 1
                topic_data[topic_name]["matched_keywords"].update(matched_in_sentence)
                if len(topic_data[topic_name]["sample_sentences"]) < 3:
                    topic_data[topic_name]["sample_sentences"].append(sentence)
                total_topic_matches += count

    # Format topic results sorted by match_count descending
    ranked_topics = []
    for name, data in topic_data.items():
        if data["match_count"] > 0:
            kw_list = [
                {"keyword": kw, "count": cnt}
                for kw, cnt in data["matched_keywords"].most_common(8)
            ]
            ranked_topics.append({
                "topic": name,
                "match_count": data["match_count"],
                "sentence_count": data["sentence_count"],
                "matched_keywords": kw_list,
                "sample_sentences": data["sample_sentences"],
                "percentage_of_matches": round(
                    (data["match_count"] / total_topic_matches) * 100, 1
                ) if total_topic_matches > 0 else 0.0,
            })

    ranked_topics.sort(key=lambda t: t["match_count"], reverse=True)

    summary = (
        f"Identified {len(ranked_topics)} active topic(s) across {len(sentences)} sentence(s) "
        f"with a total of {total_topic_matches} keyword matches."
    )

    return {
        "topics": ranked_topics,
        "total_topic_matches": total_topic_matches,
        "summary": summary,
    }


# =====================================================================
# E. Rule-Based Insight Generation
# =====================================================================

def generate_text_insights(
    text: str,
    stats: dict[str, Any],
    keywords: list[dict[str, Any]],
    sentiment: dict[str, Any],
    topics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Synthesize findings from statistics, keywords, sentiment, and topics into actionable insights.

    Each insight adheres strictly to:
    - title: Clear, descriptive headline
    - explanation: Plain language explanation of the finding
    - evidence: Concrete numerical/lexical supporting evidence
    - category: Thematic category or specific topic
    - priority: 'High', 'Medium', or 'Low'
    """
    insights: list[dict[str, Any]] = []

    if not text or not text.strip():
        insights.append({
            "title": "Empty Document Content",
            "explanation": "No extractable textual content was found in the selected text.",
            "evidence": "Character count: 0, Word count: 0.",
            "category": "Data Availability",
            "priority": "High",
        })
        return insights

    total_sentences = stats.get("total_sentences", 0)
    total_words = stats.get("total_words", 0)
    sentiment_counts = sentiment.get("sentence_counts", {})
    pos_count = sentiment_counts.get("Positive", 0)
    neg_count = sentiment_counts.get("Negative", 0)
    neu_count = sentiment_counts.get("Neutral", 0)
    pos_pct = sentiment.get("sentence_percentages", {}).get("Positive", 0.0)
    neg_pct = sentiment.get("sentence_percentages", {}).get("Negative", 0.0)
    overall_sentiment = sentiment.get("overall_sentiment", "Neutral")
    active_topics = topics.get("topics", [])

    # -------------------------------------------------------------
    # 1. Overall Feedback Sentiment Insight
    # -------------------------------------------------------------
    if total_sentences > 0:
        if overall_sentiment == "Positive":
            priority = "High" if pos_pct >= 60.0 else "Medium"
            explanation = (
                f"Overall document tone is predominantly positive, with {pos_pct}% of sentences "
                f"expressing positive sentiment."
            )
            evidence = (
                f"{pos_count} of {total_sentences} sentences classified Positive ({pos_pct}%); "
                f"Top positive words: {', '.join(k['word'] for k in sentiment.get('positive_keywords_found', [])[:4]) or 'N/A'}."
            )
        elif overall_sentiment == "Negative":
            priority = "High"
            explanation = (
                f"Overall document tone reflects customer dissatisfaction, with {neg_pct}% of sentences "
                f"containing negative sentiment indicators."
            )
            evidence = (
                f"{neg_count} of {total_sentences} sentences classified Negative ({neg_pct}%); "
                f"Top negative words: {', '.join(k['word'] for k in sentiment.get('negative_keywords_found', [])[:4]) or 'N/A'}."
            )
        else:
            priority = "Low"
            explanation = (
                "Document tone is balanced or neutral, with no single sentiment direction dominating."
            )
            evidence = (
                f"Neutral: {neu_count} ({sentiment.get('sentence_percentages', {}).get('Neutral', 0.0)}%), "
                f"Positive: {pos_count} ({pos_pct}%), Negative: {neg_count} ({neg_pct}%)."
            )

        insights.append({
            "title": f"Overall Sentiment Tone: {overall_sentiment}",
            "explanation": explanation,
            "evidence": evidence,
            "category": "Sentiment Distribution",
            "priority": priority,
        })

    # -------------------------------------------------------------
    # 2. Dominant Discussion Topic Insight
    # -------------------------------------------------------------
    if active_topics:
        top_topic = active_topics[0]
        top_kws = ", ".join(f"{item['keyword']} ({item['count']})" for item in top_topic["matched_keywords"][:4])
        insights.append({
            "title": f"Dominant Focus Area: {top_topic['topic']}",
            "explanation": (
                f"The topic '{top_topic['topic']}' is the most frequently discussed theme in the document."
            ),
            "evidence": (
                f"Accounted for {top_topic['match_count']} keyword triggers ({top_topic['percentage_of_matches']}% "
                f"of all topic hits) across {top_topic['sentence_count']} sentence(s). Key triggers: {top_kws}."
            ),
            "category": top_topic["topic"],
            "priority": "High" if top_topic["percentage_of_matches"] >= 30.0 else "Medium",
        })

    # -------------------------------------------------------------
    # 3. Common Customer Complaints & Friction Areas
    # -------------------------------------------------------------
    neg_keywords = sentiment.get("negative_keywords_found", [])
    if neg_keywords and neg_count > 0:
        complaint_words_str = ", ".join(f"{item['word']} ({item['count']})" for item in neg_keywords[:5])
        # Find which topics co-occur with negative keywords
        neg_associated_topics = []
        for top in active_topics:
            topic_name = top["topic"]
            kw_names = {k["keyword"] for k in top["matched_keywords"]}
            overlap = kw_names.intersection({k["word"] for k in neg_keywords})
            if overlap or any(kw in TOPIC_TAXONOMY.get(topic_name, set()) for kw in [k["word"] for k in neg_keywords]):
                neg_associated_topics.append(topic_name)

        topic_str = f" Relevant topics: {', '.join(neg_associated_topics)}." if neg_associated_topics else ""
        insights.append({
            "title": "Identified Customer Complaints & Friction Points",
            "explanation": (
                f"Repeated negative sentiment triggers were identified in customer feedback.{topic_str}"
            ),
            "evidence": (
                f"Found {len(neg_keywords)} distinct complaint terms with {sum(k['count'] for k in neg_keywords)} "
                f"total occurrences. Frequent complaint terms: {complaint_words_str}."
            ),
            "category": "Customer Friction",
            "priority": "High",
        })

    # -------------------------------------------------------------
    # 4. Frequently Mentioned Positive Experiences
    # -------------------------------------------------------------
    pos_keywords = sentiment.get("positive_keywords_found", [])
    if pos_keywords and pos_count > 0:
        praise_words_str = ", ".join(f"{item['word']} ({item['count']})" for item in pos_keywords[:5])
        insights.append({
            "title": "Frequently Praised Customer Experiences",
            "explanation": (
                "Customers explicitly praised specific positive aspects of their experience."
            ),
            "evidence": (
                f"Found {len(pos_keywords)} distinct positive terms with {sum(k['count'] for k in pos_keywords)} "
                f"total occurrences. Key praise terms: {praise_words_str}."
            ),
            "category": "Customer Satisfaction",
            "priority": "Medium",
        })

    # -------------------------------------------------------------
    # 5. Repeated Key Terms and Vocabulary Concentration
    # -------------------------------------------------------------
    if keywords:
        top_3_kw = keywords[:3]
        kw_str = ", ".join(f"'{k['keyword']}' ({k['frequency']}x, {k['percentage']}%)" for k in top_3_kw)
        insights.append({
            "title": "Lexical Keyword Concentration",
            "explanation": (
                "A core set of meaningful keywords dominates the document vocabulary."
            ),
            "evidence": (
                f"Top keywords after stopword filtering: {kw_str}."
            ),
            "category": "Keyword Frequency",
            "priority": "Low",
        })

    # -------------------------------------------------------------
    # 6. Secondary / Emerging Discussion Themes
    # -------------------------------------------------------------
    if len(active_topics) >= 2:
        secondary_topics = active_topics[1:3]
        sec_names = ", ".join(
            f"{t['topic']} ({t['match_count']} hits, {t['sentence_count']} sentences)"
            for t in secondary_topics
        )
        insights.append({
            "title": "Secondary Discussion Themes",
            "explanation": (
                "In addition to the primary focus area, customer feedback frequently touched on secondary topics."
            ),
            "evidence": f"Secondary active themes: {sec_names}.",
            "category": "Topic Analysis",
            "priority": "Medium",
        })

    return insights


# =====================================================================
# Main Analysis Pipeline Runner
# =====================================================================

def run_text_analysis(text: str) -> dict[str, Any]:
    """Execute the full Phase 17 text analysis pipeline on the provided text.

    Returns a structured dictionary containing:
    - statistics: dict of structural metrics
    - keywords: list of top keywords with frequencies
    - sentiment: dict of rule-based sentiment results and distributions
    - topics: dict of identified topics and evidence
    - insights: list of prioritized actionable insight dictionaries
    - is_empty: bool flag indicating whether usable text was provided
    """
    text = text or ""
    is_empty = len(text.strip()) == 0

    stats = calculate_text_statistics(text)
    keywords = analyze_keyword_frequency(text, top_n=15)
    sentiment = analyze_sentiment_rules(text)
    topics = identify_topics(text)
    insights = generate_text_insights(text, stats, keywords, sentiment, topics)

    return {
        "text": text,
        "is_empty": is_empty,
        "statistics": stats,
        "keywords": keywords,
        "sentiment": sentiment,
        "topics": topics,
        "insights": insights,
    }


# =====================================================================
# Session State Helpers for Clean Isolation
# =====================================================================

def get_analyzable_text(
    state: MutableMapping[str, Any],
    text_source: str = "auto",
) -> tuple[str, str, str]:
    """Retrieve the text to analyze from session state safely without mutating state.

    Parameters:
    - state: Streamlit session state dictionary
    - text_source: "cleaned", "original", or "auto"

    Returns:
    - (text, source_label, availability_note)
    """
    original_text = state.get("doc_clean_original_text")
    if original_text is None:
        # Fall back to extraction result if present
        ext_res = state.get("doc_extraction_result")
        if isinstance(ext_res, dict):
            original_text = ext_res.get("text", "")

    cleaned_text = state.get("doc_cleaned_text")

    if text_source == "cleaned" and cleaned_text is not None and cleaned_text.strip():
        return (
            cleaned_text,
            "Cleaned Text",
            "Analyzing applied cleaned text from Phase 16 cleaning pipeline.",
        )
    elif text_source == "original" and original_text is not None:
        return (
            original_text,
            "Original Extracted Text",
            "Analyzing raw extracted document text without cleaning transformations.",
        )
    else:
        # Auto mode: prefer cleaned text if available, else original text
        if cleaned_text is not None and cleaned_text.strip():
            return (
                cleaned_text,
                "Cleaned Text",
                "Cleaned text was detected and selected by default.",
            )
        elif original_text is not None:
            return (
                original_text,
                "Original Extracted Text",
                "Cleaned text has not been applied yet; analyzing original extracted text.",
            )
        else:
            return (
                "",
                "No Document Loaded",
                "No document has been uploaded or extracted in session state.",
            )


def clear_document_analysis_state(state: MutableMapping[str, Any]) -> None:
    """Clear document analysis results from session state when document changes."""
    for key in DOCUMENT_ANALYSIS_STATE_KEYS:
        state.pop(key, None)
