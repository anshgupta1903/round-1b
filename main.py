

import os
import json
import time
import spacy
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import fitz
import re
import argparse

# Your Round 1A logic (make sure path matches your project)
from scripts.round1a_main import process_all_pdfs

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_and_rank_keywords(text):
    """
    Extracts keywords from text and ranks them:
    - named entities first (more important)
    - then noun chunks sorted by length
    """
    doc = nlp(text)
    entities = {ent.text.lower() for ent in doc.ents if len(ent.text.strip()) > 1}
    noun_chunks = {chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 1}
    # Remove overlaps
    noun_chunks = noun_chunks - entities
    # Sort noun chunks by length desc
    sorted_noun_chunks = sorted(noun_chunks, key=lambda x: len(x), reverse=True)
    # Combine: entities first, then long noun chunks
    keywords = list(entities) + sorted_noun_chunks
    return keywords



def rank_headings_and_titles(outline_list, persona, job, model, keywords, top_n=5):
    """
    Rank headings & titles: ensure top N cover different keywords first,
    then fill remaining spots by highest score.
    """
    from scipy.spatial.distance import cosine

    query = persona + ". " + job
    query_emb = model.encode(query)

    # Step 1: compute scores & store all candidates
    all_candidates = []

    for doc in outline_list:
        doc_name = doc["document"]

        # Title
        title_text = doc.get("title", "")
        if title_text:
            title_emb = model.encode(title_text.lower())
            base_score = 1 - cosine(query_emb, title_emb)

            # Boost if keywords in title
            boost = 0
            title_lower = title_text.lower()
            for kw in keywords:
                if kw in title_lower:
                    boost += 0.8
                elif any(k in title_lower for k in kw.split()):
                    boost += 0.4

            final_score = (base_score + boost) * 3.0  # title weight

            all_candidates.append({
                "document": doc_name,
                "page_number": 0,
                "section_title": title_text,
                "level": "TITLE",
                "similarity": float(final_score),
                "keywords": [kw for kw in keywords if kw in title_lower or any(k in title_lower for k in kw.split())]
            })

        # Headings
        for h in doc["outline"]:
            heading_text = h["text"].lower()
            heading_emb = model.encode(heading_text)
            base_score = 1 - cosine(query_emb, heading_emb)

            boost = 0
            matched_keywords = []
            for kw in keywords:
                if kw in heading_text:
                    boost += 0.8
                    matched_keywords.append(kw)
                elif any(k in heading_text for k in kw.split()):
                    boost += 0.4
                    matched_keywords.append(kw)

            weight = {"H1": 2.0, "H2": 1.5, "H3": 1.2, "H4": 1.0}.get(h["level"], 1.0)
            final_score = (base_score + boost) * weight

            all_candidates.append({
                "document": doc_name,
                "page_number": h["page"],
                "section_title": h["text"],
                "level": h["level"],
                "similarity": float(final_score),
                "keywords": matched_keywords
            })

    # Step 2: pick top heading for each keyword first
    selected = []
    used_texts = set()

    for kw in keywords:
        kw_candidates = [c for c in all_candidates if kw in c["keywords"] and c["section_title"] not in used_texts]
        if kw_candidates:
            best = max(kw_candidates, key=lambda x: x["similarity"])
            selected.append(best)
            used_texts.add(best["section_title"])

        if len(selected) >= top_n:
            break

    # Step 3: fill remaining spots by highest score, skipping used ones
    if len(selected) < top_n:
        remaining = sorted(
            [c for c in all_candidates if c["section_title"] not in used_texts],
            key=lambda x: x["similarity"],
            reverse=True
        )
        for c in remaining:
            selected.append(c)
            used_texts.add(c["section_title"])
            if len(selected) >= top_n:
                break

    # Step 4: add importance_rank
    for idx, item in enumerate(selected):
        item["importance_rank"] = idx + 1

    return selected



def extract_subsections(top_sections, pdf_dir):
    results = []

    for sec in top_sections[:3]:
        pdf_path = os.path.join(pdf_dir, sec["document"])
        try:
            doc = fitz.open(pdf_path)
            page = doc[sec["page_number"]]
            text = page.get_text()
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            snippet = ""
            # Find first paragraph containing section title, else first long paragraph
            for para in paragraphs:
                if sec["section_title"].lower() in para.lower():
                    snippet = para
                    break
            if not snippet and paragraphs:
                snippet = paragraphs[0]
            results.append({
                "document": sec["document"],
                "page_number": sec["page_number"],
                "refined_text": snippet[:1000]  # limit length
            })
        except Exception as e:
            print(f"❌ Error reading {pdf_path}: {e}")

    return results


from rapidfuzz import fuzz

def compute_boost(text, keywords):
    text_lower = text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in text_lower:
            return 2.0  # strong boost
        else:
            partial_score = fuzz.partial_ratio(text_lower, kw_lower)
            if partial_score > 70:
                return 1.2  # medium boost
    return 1.0  # no boost



# 🔧 Input Handling
# ----------------------------
def get_inputs(input_file_path="input.txt"):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_file_path}")

    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        if len(lines) < 2:
            raise ValueError("❌ input.txt must contain at least 2 lines: persona and job.")
        persona = lines[0].strip()
        job = lines[1].strip()

    return persona, job


def main():
    print("Script Started")
    input_dir = "./pdfs"
    # output_dir_1a = "./Collection/title_and_headings"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Run Round 1A to create outline JSONs
    print("📄 Extracting outlines from PDFs...")
    process_all_pdfs(input_dir, output_dir)

    # Step 2: Define persona & job
    # print("Enter persona")
    persona, job = get_inputs()
    print(f"🧠 Persona: {persona}")
    print(f"🧠 Job: {job}")
    # Step 3: Extract and rank keywords
    stop_words = {"a", "an", "the", "for", "and", "or", "of", "in", "to", "with", "on", "at", "from", "by", "as", "is", "are"}

    # extract words separately
    job_keywords_raw = re.findall(r'\w+', job.lower())
    persona_keywords_raw = re.findall(r'\w+', persona.lower())

    # remove stopwords, keep order
    seen = set()
    job_keywords = []
    for word in job_keywords_raw:
        if word not in stop_words and word not in seen:
            job_keywords.append(word)
            seen.add(word)

    persona_keywords = []
    for word in persona_keywords_raw:
        if word not in stop_words and word not in seen:
            persona_keywords.append(word)
            seen.add(word)

    # combine: job first, persona last
    keywords = job_keywords + persona_keywords

    print(f"🧠 Extracted & ordered keywords: {keywords}")

    # Step 4: Load outlines
    outlines = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".json") and filename != "challenge1b_output.json":
            with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                outlines.append({
                    "document": filename.replace(".json", ".pdf"),
                    "title": data.get("title", ""),
                    "outline": data.get("outline", [])
                })

    # Step 5: load embedding model
    model = SentenceTransformer('./model')

    # Step 6: rank sections
    ranked = rank_headings_and_titles(outlines, persona, job, model, keywords)
    topN = ranked[:20]  # get top 20

    # Step 7: extract subsections
    subsections = extract_subsections(topN, input_dir)

    # Step 8: build final output
    final_output = {
        "metadata": {
            "input_documents": [o["document"] for o in outlines],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "extracted_sections": [
            {
                "document": item["document"],
                "page_number": item["page_number"],
                "section_title": item["section_title"],
                "level": item["level"],
                "importance_rank": item["importance_rank"]
            } for item in topN
        ],
        "subsection_analysis": subsections
    }

    # Step 9: save output
    with open(os.path.join(output_dir, "challenge1b_output.json"), "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)

    print("✅ Done! Output saved to Collection/output/challenge1b_output.json")


if __name__ == "__main__":
    main()







