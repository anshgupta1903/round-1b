# # import os
# # import json
# # import time
# # import spacy
# # from sentence_transformers import SentenceTransformer
# # from scipy.spatial.distance import cosine
# # import fitz  # PyMuPDF

# # # Import your Round 1A logic
# # from scripts.round1a_main import process_all_pdfs

# # def extract_keywords_with_spacy(text):
# #     """Extracts noun chunks & named entities as keywords from text"""
# #     nlp = spacy.load("en_core_web_sm")
# #     doc = nlp(text)
# #     keywords = set(chunk.text.lower() for chunk in doc.noun_chunks)
# #     keywords.update(ent.text.lower() for ent in doc.ents)
# #     return list(keywords)

# # def rank_headings_and_title(outline_list, persona, job, model, keywords):
# #     query = persona + ". " + job
# #     query_emb = model.encode(query)

# #     ranked = []

# #     for doc in outline_list:
# #         doc_name = doc["document"]

# #         # Title
# #         if doc.get("title"):
# #             text = doc["title"]
# #             title_emb = model.encode(text)
# #             score = 1 - cosine(query_emb, title_emb)
# #             keyword_hits = sum(1 for kw in keywords if kw in text.lower())
# #             boost = 0.05 * keyword_hits
# #             ranked.append({
# #                 "document": doc_name,
# #                 "page_number": 0,
# #                 "section_title": text,
# #                 "similarity": float(score) * 2.0 + boost,  # Title weight
# #                 "source": "title"
# #             })

# #         # Headings
# #         for h in doc["outline"]:
# #             text = h["text"]
# #             heading_emb = model.encode(text)
# #             base_score = 1 - cosine(query_emb, heading_emb)
# #             weight = {"H1": 1.5, "H2": 1.2, "H3": 1.0}.get(h["level"], 1.0)
# #             keyword_hits = sum(1 for kw in keywords if kw in text.lower())
# #             boost = 0.05 * keyword_hits
# #             ranked.append({
# #                 "document": doc_name,
# #                 "page_number": h["page"],
# #                 "section_title": text,
# #                 "similarity": float(base_score) * weight + boost,
# #                 "source": h["level"]
# #             })

# #     ranked.sort(key=lambda x: x["similarity"], reverse=True)
# #     for idx, item in enumerate(ranked):
# #         item["importance_rank"] = idx + 1
# #     return ranked

# # def extract_subsections(top_sections, pdf_dir, keywords):
# #     results = []
# #     for sec in top_sections:
# #         pdf_path = os.path.join(pdf_dir, sec["document"])
# #         try:
# #             doc = fitz.open(pdf_path)
# #             page = doc[sec["page_number"]]
# #             text = page.get_text()
# #             paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

# #             best_para = ""
# #             best_score = 0

# #             for para in paragraphs:
# #                 para_lower = para.lower()
# #                 keyword_hits = sum(1 for kw in keywords if kw in para_lower)
# #                 score = keyword_hits
# #                 if len(para) > 50:
# #                     score += 0.5
# #                 if score > best_score:
# #                     best_score = score
# #                     best_para = para

# #             snippet = best_para[:500] if best_para else (paragraphs[0][:500] if paragraphs else "")

# #             results.append({
# #                 "document": sec["document"],
# #                 "page_number": sec["page_number"],
# #                 "refined_text": snippet
# #             })

# #         except Exception as e:
# #             print(f"ERROR reading {pdf_path}: {e}")
# #     return results

# # def main():
# #     input_dir = "./input"
# #     output_dir = "./output"
# #     os.makedirs(output_dir, exist_ok=True)

# #     # Step 1: process PDFs
# #     print("INFO: Running Round 1A to extract outlines...")
# #     process_all_pdfs(input_dir, output_dir, ml_output_dir=None)

# #     # Step 2: persona & job
# #     persona = "HR professional"
# #     job = "Create and manage fillable forms for onboarding and compliance."

# #     # Step 3: extract keywords with spaCy
# #     keywords = extract_keywords_with_spacy(persona + " " + job)
# #     print(f"INFO: Extracted keywords for boosting: {keywords}")

# #     # Step 4: load outlines
# #     outlines = []
# #     for filename in os.listdir(output_dir):
# #         if filename.endswith(".json") and filename != "final_output.json":
# #             with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
# #                 data = json.load(f)
# #                 outlines.append({
# #                     "document": filename.replace(".json", ".pdf"),
# #                     "title": data.get("title", ""),
# #                     "outline": data.get("outline", [])
# #                 })

# #     # Step 5: load embedding model
# #     model = SentenceTransformer('./local_model')  # offline MiniLM

# #     # Step 6: rank headings & titles
# #     ranked = rank_headings_and_title(outlines, persona, job, model, keywords)

# #     # keep top 5
# #     topN = ranked[:5]

# #     # Step 7: extract subsections
# #     subsections = extract_subsections(topN, input_dir, keywords)

# #     # Step 8: build final output
# #     final_output = {
# #         "metadata": {
# #             "input_documents": [o["document"] for o in outlines],
# #             "persona": persona,
# #             "job_to_be_done": job,
# #             "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
# #         },
# #         "extracted_sections": [
# #             {
# #                 "document": item["document"],
# #                 "page_number": item["page_number"],
# #                 "section_title": item["section_title"],
# #                 "importance_rank": item["importance_rank"]
# #             } for item in topN
# #         ],
# #         "subsection_analysis": [
# #             {
# #                 "document": sub["document"],
# #                 "page_number": sub["page_number"],
# #                 "refined_text": sub["refined_text"]
# #             } for sub in subsections
# #         ]
# #     }

# #     with open(os.path.join(output_dir, "final_output.json"), "w", encoding='utf-8') as f:
# #         json.dump(final_output, f, indent=2)
# #     print("✅ Done! Output saved to output/final_output.json")

# # if __name__ == "__main__":
# #     main()




# # import os
# # import json
# # import time
# # import re
# # import spacy
# # import difflib
# # from sentence_transformers import SentenceTransformer
# # from scipy.spatial.distance import cosine
# # import fitz  # PyMuPDF

# # # Import your Round 1A logic
# # try:
# #     from scripts.round1a_main import process_all_pdfs
# # except ImportError:
# #     print("FATAL ERROR: Could not import 'process_all_pdfs'. Check your 'scripts' folder setup.")
# #     exit()

# # def extract_keywords_with_spacy(text):
# #     """Extracts noun chunks as keywords from text"""
# #     try:
# #         nlp = spacy.load("en_core_web_sm")
# #     except OSError:
# #         print("Downloading 'en_core_web_sm' model...")
# #         spacy.cli.download("en_core_web_sm")
# #         nlp = spacy.load("en_core_web_sm")
        
# #     doc = nlp(text)
# #     keywords = set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1)
# #     # Add single important nouns as well
# #     keywords.update(token.text.lower() for token in doc if token.pos_ == "NOUN" and len(token.text) > 4)
# #     return list(keywords)

# # def rank_headings_and_title(outline_list, persona, job, model, keywords):
# #     """Ranks headings based on a combination of semantic similarity and a strong keyword boost."""
# #     query = persona + ". " + job
# #     query_emb = model.encode(query)
    
# #     ranked = []
# #     KEYWORD_BOOST = 10.0  # Massively increased boost

# #     for doc in outline_list:
# #         doc_name = doc["document"]
# #         all_content = []
# #         if doc.get("title"):
# #             all_content.append({"text": doc["title"], "page": 0, "level": "H1"}) # Treat title as H1
# #         if doc.get("outline"):
# #             all_content.extend(doc["outline"])

# #         for item in all_content:
# #             text = item["text"]
# #             text_lower = text.lower()
            
# #             # Semantic Score
# #             heading_emb = model.encode(text)
# #             semantic_score = 1 - cosine(query_emb, heading_emb)
            
# #             # Keyword Score
# #             keyword_hits = 0
# #             for kw in keywords:
# #                 # Use regex for whole-word matching
# #                 if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
# #                     keyword_hits += 1
            
# #             # Final score is dominated by the keyword boost
# #             final_score = (keyword_hits * KEYWORD_BOOST) + semantic_score

# #             if final_score > 0: # Only include items with some relevance
# #                 ranked.append({
# #                     "document": doc_name,
# #                     "page_number": item["page"],
# #                     "section_title": text,
# #                     "score": final_score
# #                 })

# #     ranked.sort(key=lambda x: x["score"], reverse=True)
# #     for idx, item in enumerate(ranked):
# #         item["importance_rank"] = idx + 1
# #     return ranked

# # def extract_subsections(top_sections, pdf_dir):
# #     """Extracts content under a heading using fuzzy matching to locate it."""
# #     results = []
# #     MATCH_THRESHOLD = 0.85

# #     for sec in top_sections:
# #         pdf_path = os.path.join(pdf_dir, sec["document"])
# #         snippet = ""
# #         try:
# #             doc = fitz.open(pdf_path)
# #             page = doc[sec["page_number"]]
# #             blocks = page.get_text("blocks")
            
# #             for i, block in enumerate(blocks):
# #                 block_text = block[4].strip().replace("\n", " ")
# #                 similarity = difflib.SequenceMatcher(None, sec["section_title"].lower(), block_text.lower()).ratio()

# #                 if similarity > MATCH_THRESHOLD:
# #                     # Found the heading, now find the next content block
# #                     for next_block in blocks[i+1:]:
# #                         next_text = next_block[4].strip()
# #                         if next_text:
# #                             snippet = next_text.replace("\n", " ")
# #                             break
# #                     break 
# #         except Exception as e:
# #             print(f"ERROR reading {pdf_path}: {e}")
        
# #         results.append({
# #             "document": sec["document"],
# #             "page_number": sec["page_number"],
# #             "refined_text": snippet[:700]
# #         })
# #     return results

# # def main():
# #     input_dir = "./input"
# #     output_dir = "./output"
# #     os.makedirs(output_dir, exist_ok=True)

# #     print("INFO: Running Round 1A to extract outlines...")
# #     # This function should create the JSON outline files in the output directory
# #     # process_all_pdfs(input_dir, output_dir, ml_output_dir=None)

# #     persona = "HR professional"
# #     job = "Create and manage fillable forms for onboarding and compliance."

# #     print("INFO: Extracting keywords from persona and job description...")
# #     keywords = extract_keywords_with_spacy(persona + " " + job)
# #     print(f"INFO: Found keywords: {keywords}")

# #     outlines = []
# #     json_files = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != "final_output.json"]
# #     for filename in json_files:
# #         with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
# #             data = json.load(f)
# #             # Standardize the data structure for processing
# #             outlines.append({
# #                 "document": filename.replace(".json", ".pdf"),
# #                 "title": data.get("title", ""),
# #                 "outline": data.get("outline", [])
# #             })
    
# #     if not outlines:
# #         print("FATAL ERROR: No outline files found in the 'output' directory. Please ensure Round 1A ran successfully.")
# #         return

# #     print("INFO: Loading sentence transformer model...")
# #     model = SentenceTransformer('./local_model')

# #     print("INFO: Ranking sections...")
# #     ranked = rank_headings_and_title(outlines, persona, job, model, keywords)
# #     topN = ranked[:5]

# #     print("INFO: Extracting subsections for top 5 sections...")
# #     subsections = extract_subsections(topN, input_dir)

# #     final_output = {
# #         "metadata": {
# #             "input_documents": [os.path.basename(o["document"]) for o in outlines],
# #             "persona": persona,
# #             "job_to_be_done": job,
# #             "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
# #         },
# #         "extracted_sections": [
# #             {"document": item["document"], "page_number": item["page_number"], "section_title": item["section_title"], "importance_rank": item["importance_rank"]} 
# #             for item in topN
# #         ],
# #         "subsection_analysis": subsections
# #     }

# #     output_path = os.path.join(output_dir, "final_output.json")
# #     with open(output_path, "w", encoding='utf-8') as f:
# #         json.dump(final_output, f, indent=2, ensure_ascii=False)
        
# #     print(f"\n✅ Done! Output saved to {output_path}")

# # if __name__ == "__main__":
# #     # Note: I've commented out the call to process_all_pdfs since you are providing the JSON files directly.
# #     # If you need the script to run the PDF processing from scratch, uncomment the line below.
# #     main()



# # import os
# # import json
# # import time
# # import re
# # import spacy
# # import difflib
# # from sentence_transformers import SentenceTransformer
# # from scipy.spatial.distance import cosine
# # import fitz  # PyMuPDF

# # # Import your Round 1A logic
# # try:
# #     from scripts.round1a_main import process_all_pdfs
# # except ImportError:
# #     print("FATAL ERROR: Could not import 'process_all_pdfs'. Check your 'scripts' folder setup.")
# #     exit()

# # def extract_keywords_with_spacy(text):
# #     """Extracts noun chunks as keywords from text"""
# #     try:
# #         nlp = spacy.load("en_core_web_sm")
# #     except OSError:
# #         print("Downloading 'en_core_web_sm' model...")
# #         spacy.cli.download("en_core_web_sm")
# #         nlp = spacy.load("en_core_web_sm")
        
# #     doc = nlp(text)
# #     keywords = set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1)
# #     keywords.update(token.text.lower() for token in doc if token.pos_ == "NOUN" and len(token.text) > 4)
# #     return list(keywords)

# # def rank_headings_and_title(outline_list, persona, job, model, keywords):
# #     """Ranks headings based on a combination of semantic similarity and a strong keyword boost."""
# #     query = persona + ". " + job
# #     query_emb = model.encode(query)
    
# #     ranked = []
# #     KEYWORD_BOOST = 10.0

# #     for doc in outline_list:
# #         doc_name = doc["document"]
# #         all_content = []
# #         if doc.get("title"):
# #             all_content.append({"text": doc["title"], "page": 0, "level": "H1"})
# #         if doc.get("outline"):
# #             all_content.extend(doc["outline"])

# #         for item in all_content:
# #             text, text_lower = item["text"], item["text"].lower()
# #             heading_emb = model.encode(text)
# #             semantic_score = 1 - cosine(query_emb, heading_emb)
            
# #             keyword_hits = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
# #             final_score = (keyword_hits * KEYWORD_BOOST) + semantic_score

# #             if final_score > 0:
# #                 ranked.append({"document": doc_name, "page_number": item["page"], "section_title": text, "score": final_score})

# #     ranked.sort(key=lambda x: x["score"], reverse=True)
# #     for idx, item in enumerate(ranked):
# #         item["importance_rank"] = idx + 1
# #     return ranked

# # def extract_subsections(top_sections, pdf_dir, all_headings_map):
# #     """
# #     Extracts multiple paragraphs of content under a heading until the next heading is found.
# #     """
# #     results = []
# #     MATCH_THRESHOLD = 0.85

# #     for sec in top_sections:
# #         pdf_path = os.path.join(pdf_dir, sec["document"])
# #         snippet = ""
# #         try:
# #             doc = fitz.open(pdf_path)
# #             page = doc[sec["page_number"]]
# #             blocks = page.get_text("blocks")
            
# #             # Get all headings on the current page to know where to stop
# #             headings_on_this_page = all_headings_map.get(sec["document"], {}).get(sec["page_number"], [])

# #             for i, block in enumerate(blocks):
# #                 block_text = block[4].strip().replace("\n", " ")
# #                 similarity = difflib.SequenceMatcher(None, sec["section_title"].lower(), block_text.lower()).ratio()

# #                 if similarity > MATCH_THRESHOLD:
# #                     # --- REVISED LOGIC: Collect multiple blocks ---
# #                     content_parts = []
# #                     # Loop through subsequent blocks
# #                     for next_block in blocks[i+1:]:
# #                         next_text = next_block[4].strip().replace("\n", " ")
                        
# #                         # Check if this block is another heading
# #                         is_next_heading = False
# #                         for h in headings_on_this_page:
# #                             if difflib.SequenceMatcher(None, h.lower(), next_text.lower()).ratio() > 0.9:
# #                                 is_next_heading = True
# #                                 break
                        
# #                         if is_next_heading:
# #                             break # Stop if we hit the next heading

# #                         if next_text:
# #                             content_parts.append(next_text)
                    
# #                     snippet = " ".join(content_parts)
# #                     break 
# #         except Exception as e:
# #             print(f"ERROR reading {pdf_path}: {e}")
        
# #         results.append({
# #             "document": sec["document"],
# #             "page_number": sec["page_number"],
# #             # Increased character limit for a larger subsection
# #             "refined_text": snippet[:1500]
# #         })
# #     return results

# # def main():
# #     input_dir = "./input"
# #     output_dir = "./output"
# #     os.makedirs(output_dir, exist_ok=True)

# #     # process_all_pdfs(input_dir, output_dir, ml_output_dir=None)

# #     persona = "HR professional"
# #     job = "Create and manage fillable forms for onboarding and compliance."

# #     keywords = extract_keywords_with_spacy(persona + " " + job)
    
# #     outlines = []
# #     json_files = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != "final_output.json"]
# #     for filename in json_files:
# #         with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
# #             data = json.load(f)
# #             outlines.append({
# #                 "document": filename.replace(".json", ".pdf"),
# #                 "title": data.get("title", ""),
# #                 "outline": data.get("outline", [])
# #             })
    
# #     if not outlines:
# #         print("FATAL ERROR: No outline files found.")
# #         return

# #     # Create a map of all headings for the extraction function
# #     all_headings_map = {}
# #     for doc in outlines:
# #         doc_name = doc["document"]
# #         all_headings_map[doc_name] = {}
# #         for heading in doc.get("outline", []):
# #             page_num = heading["page"]
# #             if page_num not in all_headings_map[doc_name]:
# #                 all_headings_map[doc_name][page_num] = []
# #             all_headings_map[doc_name][page_num].append(heading["text"])

# #     model = SentenceTransformer('./local_model')

# #     ranked = rank_headings_and_title(outlines, persona, job, model, keywords)
# #     topN = ranked[:5]

# #     subsections = extract_subsections(topN, input_dir, all_headings_map)

# #     final_output = {
# #         "metadata": {
# #             "input_documents": [os.path.basename(o["document"]) for o in outlines],
# #             "persona": persona,
# #             "job_to_be_done": job,
# #             "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
# #         },
# #         "extracted_sections": [
# #             {"document": item["document"], "page_number": item["page_number"], "section_title": item["section_title"], "importance_rank": item["importance_rank"]} 
# #             for item in topN
# #         ],
# #         "subsection_analysis": subsections
# #     }

# #     output_path = os.path.join(output_dir, "final_output.json")
# #     with open(output_path, "w", encoding='utf-8') as f:
# #         json.dump(final_output, f, indent=2, ensure_ascii=False)
        
# #     print(f"\n✅ Done! Output saved to {output_path}")

# # if __name__ == "__main__":
# #     main()



# # import os
# # import json
# # import time
# # import re
# # import spacy
# # import difflib
# # from sentence_transformers import SentenceTransformer
# # from scipy.spatial.distance import cosine
# # import fitz  # PyMuPDF

# # # Import your Round 1A logic
# # try:
    
# #     from scripts.round1a_main import process_all_pdfs
# #     print("INFO: Successfully imported 'process_all_pdfs'.")
# # except ImportError:
# #     print("FATAL ERROR: Could not import 'process_all_pdfs'. Check your 'scripts' folder setup.")
# #     exit()

# # def extract_keywords_with_spacy(text):
# #     """Extracts noun chunks as keywords from text"""
# #     try:
# #         nlp = spacy.load("en_core_web_sm")
# #     except OSError:
# #         print("Downloading 'en_core_web_sm' model...")
# #         spacy.cli.download("en_core_web_sm")
# #         nlp = spacy.load("en_core_web_sm")
        
# #     doc = nlp(text)
# #     keywords = set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1)
# #     keywords.update(token.text.lower() for token in doc if token.pos_ == "NOUN" and len(token.text) > 4)
# #     return list(keywords)

# # def rank_headings_and_title(outline_list, persona, job, model, keywords):
# #     """Ranks headings based on a combination of semantic similarity and a strong keyword boost."""
# #     query = persona + ". " + job
# #     query_emb = model.encode(query)
    
# #     ranked = []
# #     KEYWORD_BOOST = 1.0

# #     for doc in outline_list:
# #         doc_name = doc["document"]
# #         all_content = []
# #         if doc.get("title"):
# #             all_content.append({"text": doc["title"], "page": 0, "level": "H1"})
# #         if doc.get("outline"):
# #             all_content.extend(doc["outline"])

# #         for item in all_content:
# #             text, text_lower = item["text"], item["text"].lower()
# #             heading_emb = model.encode(text)
# #             semantic_score = 2 - cosine(query_emb, heading_emb)
            
# #             keyword_hits = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
# #             final_score = (keyword_hits * KEYWORD_BOOST) + semantic_score

# #             if final_score > 0:
# #                 ranked.append({
# #                     "document": doc_name,
# #                     "page_number": item["page"],
# #                     "section_title": text,
# #                     "level": item["level"], # Keep the level for the new logic
# #                     "score": final_score
# #                 })

# #     ranked.sort(key=lambda x: x["score"], reverse=True)
# #     for idx, item in enumerate(ranked):
# #         item["importance_rank"] = idx + 1
# #     return ranked

# # def extract_subsections(top_sections, pdf_dir, all_headings_details_map):
# #     """
# #     Extracts content under a heading until it finds another heading of the same or higher level.
# #     """
# #     results = []
# #     MATCH_THRESHOLD = 0.85

# #     for sec in top_sections:
# #         pdf_path = os.path.join(pdf_dir, sec["document"])
# #         snippet = ""
# #         try:
# #             doc = fitz.open(pdf_path)
# #             page = doc[sec["page_number"]]
# #             blocks = page.get_text("blocks")
            
# #             current_level_num = int(sec['level'][1:])
# #             headings_on_this_page = all_headings_details_map.get(sec["document"], {}).get(sec["page_number"], [])

# #             for i, block in enumerate(blocks):
# #                 block_text = block[4].strip().replace("\n", " ")
# #                 similarity = difflib.SequenceMatcher(None, sec["section_title"].lower(), block_text.lower()).ratio()

# #                 if similarity > MATCH_THRESHOLD:
# #                     content_parts = []
# #                     # Loop through all subsequent blocks on the page
# #                     for next_block in blocks[i+1:]:
# #                         next_text = next_block[4].strip().replace("\n", " ")
                        
# #                         is_stopping_heading = False
# #                         # Check if this block is another heading
# #                         for h_details in headings_on_this_page:
# #                             if difflib.SequenceMatcher(None, h_details['text'].lower(), next_text.lower()).ratio() > 0.9:
# #                                 # It's a heading. Now check its level.
# #                                 next_heading_level_num = int(h_details['level'][1:])
# #                                 # Stop if the next heading is same or higher level (e.g., H2 <= H2)
# #                                 if next_heading_level_num <= current_level_num:
# #                                     is_stopping_heading = True
# #                                     break
                        
# #                         if is_stopping_heading:
# #                             break

# #                         if next_text:
# #                             content_parts.append(next_text)
                    
# #                     snippet = " ".join(content_parts)
# #                     break 
# #         except Exception as e:
# #             print(f"ERROR reading {pdf_path}: {e}")
        
# #         results.append({
# #             "document": sec["document"],
# #             "page_number": sec["page_number"],
# #             "refined_text": snippet[:2000] # Increased character limit
# #         })
# #     return results

# # def main():
# #     input_dir = "./input"
# #     output_dir = "./output"
# #     os.makedirs(output_dir, exist_ok=True)

# #     process_all_pdfs(input_dir, output_dir, ml_output_dir=None)

# #     persona = "HR professional"
# #     job = "Create and manage fillable forms for onboarding and compliance."

# #     keywords = extract_keywords_with_spacy(persona + " " + job)
# #     print(f"INFO: Extracted keywords for boosting: {keywords}")
    
# #     outlines = []
# #     json_files = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != "final_output.json"]
# #     for filename in json_files:
# #         with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
# #             data = json.load(f)
# #             outlines.append({
# #                 "document": filename.replace(".json", ".pdf"),
# #                 "title": data.get("title", ""),
# #                 "outline": data.get("outline", [])
# #             })
    
# #     if not outlines:
# #         print("FATAL ERROR: No outline files found.")
# #         return

# #     # Create a map of all headings WITH their levels for the extraction function
# #     all_headings_details_map = {}
# #     for doc in outlines:
# #         doc_name = doc["document"]
# #         all_headings_details_map[doc_name] = {}
# #         for heading in doc.get("outline", []):
# #             page_num = heading["page"]
# #             if page_num not in all_headings_details_map[doc_name]:
# #                 all_headings_details_map[doc_name][page_num] = []
# #             all_headings_details_map[doc_name][page_num].append({
# #                 "text": heading["text"],
# #                 "level": heading["level"]
# #             })

# #     model = SentenceTransformer('./local_model')

# #     ranked = rank_headings_and_title(outlines, persona, job, model, keywords)
# #     topN = ranked[:5]

# #     subsections = extract_subsections(topN, input_dir, all_headings_details_map)

# #     final_output = {
# #         "metadata": {
# #             "input_documents": [os.path.basename(o["document"]) for o in outlines],
# #             "persona": persona,
# #             "job_to_be_done": job,
# #             "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
# #         },
# #         "extracted_sections": [
# #             {"document": item["document"], "page_number": item["page_number"], "section_title": item["section_title"], "importance_rank": item["importance_rank"]} 
# #             for item in topN
# #         ],
# #         "subsection_analysis": subsections
# #     }

# #     output_path = os.path.join(output_dir, "final_output.json")
# #     with open(output_path, "w", encoding='utf-8') as f:
# #         json.dump(final_output, f, indent=2, ensure_ascii=False)
        
# #     print(f"\n✅ Done! Output saved to {output_path}")

# # if __name__ == "__main__":
# #     main()




# import os
# import json
# import time
# import re
# import spacy
# import difflib
# from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine
# import fitz  # PyMuPDF

# # Import your Round 1A logic
# try:
    
#     from scripts.round1a_main import process_all_pdfs
#     print("INFO: Successfully imported 'process_all_pdfs'.")
# except ImportError:
#     print("FATAL ERROR: Could not import 'process_all_pdfs'. Check your 'scripts' folder setup.")
#     exit()

# def extract_keywords_with_spacy(text):
#     """Extracts noun chunks as keywords from text"""
#     try:
#         nlp = spacy.load("en_core_web_sm")
#     except OSError:
#         print("Downloading 'en_core_web_sm' model...")
#         spacy.cli.download("en_core_web_sm")
#         nlp = spacy.load("en_core_web_sm")
        
#     doc = nlp(text)
#     keywords = set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1)
#     keywords.update(token.text.lower() for token in doc if token.pos_ == "NOUN" and len(token.text) > 4)
#     return list(keywords)

# def rank_headings_and_title(outline_list, persona, job, model, keywords):
#     """Ranks headings with a primary focus on semantic similarity, boosted by keywords."""
#     query = persona + ". " + job
#     query_emb = model.encode(query)
    
#     ranked = []
#     # --- REVISED LOGIC: New weights to prioritize semantic score ---
#     SEMANTIC_WEIGHT = 1.0
#     KEYWORD_BOOST = 1.0

#     for doc in outline_list:
#         doc_name = doc["document"]
#         all_content = []
#         if doc.get("title"):
#             all_content.append({"text": doc["title"], "page": 0, "level": "H1"})
#         if doc.get("outline"):
#             all_content.extend(doc["outline"])

#         for item in all_content:
#             text, text_lower = item["text"], item["text"].lower()
#             heading_emb = model.encode(text)
#             semantic_score = 1 - cosine(query_emb, heading_emb)
            
#             keyword_hits = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
            
#             # --- REVISED LOGIC: New scoring formula ---
#             final_score = (semantic_score * SEMANTIC_WEIGHT) + (keyword_hits * KEYWORD_BOOST)

#             if final_score > 0:
#                 ranked.append({
#                     "document": doc_name,
#                     "page_number": item["page"],
#                     "section_title": text,
#                     "level": item["level"],
#                     "score": final_score
#                 })

#     ranked.sort(key=lambda x: x["score"], reverse=True)
#     for idx, item in enumerate(ranked):
#         item["importance_rank"] = idx + 1
#     return ranked

# def extract_subsections(top_sections, pdf_dir, all_headings_details_map):
#     """
#     Extracts content under a heading until it finds another heading of the same or higher level.
#     """
#     results = []
#     MATCH_THRESHOLD = 0.85

#     for sec in top_sections:
#         pdf_path = os.path.join(pdf_dir, sec["document"])
#         snippet = ""
#         try:
#             doc = fitz.open(pdf_path)
#             page = doc[sec["page_number"]]
#             blocks = page.get_text("blocks")
            
#             current_level_num = int(sec['level'][1:])
#             headings_on_this_page = all_headings_details_map.get(sec["document"], {}).get(sec["page_number"], [])

#             for i, block in enumerate(blocks):
#                 block_text = block[4].strip().replace("\n", " ")
#                 similarity = difflib.SequenceMatcher(None, sec["section_title"].lower(), block_text.lower()).ratio()

#                 if similarity > MATCH_THRESHOLD:
#                     content_parts = []
#                     for next_block in blocks[i+1:]:
#                         next_text = next_block[4].strip().replace("\n", " ")
#                         is_stopping_heading = False
#                         for h_details in headings_on_this_page:
#                             if difflib.SequenceMatcher(None, h_details['text'].lower(), next_text.lower()).ratio() > 0.9:
#                                 next_heading_level_num = int(h_details['level'][1:])
#                                 if next_heading_level_num <= current_level_num:
#                                     is_stopping_heading = True
#                                     break
#                         if is_stopping_heading:
#                             break
#                         if next_text:
#                             content_parts.append(next_text)
#                     snippet = " ".join(content_parts)
#                     break 
#         except Exception as e:
#             print(f"ERROR reading {pdf_path}: {e}")
        
#         results.append({
#             "document": sec["document"],
#             "page_number": sec["page_number"],
#             "refined_text": snippet[:2000]
#         })
#     return results

# def main():
#     input_dir = "./input"
#     output_dir = "./output"
#     os.makedirs(output_dir, exist_ok=True)

#     process_all_pdfs(input_dir, output_dir, ml_output_dir=None)

#     persona = "HR professional"
#     job = "Create and manage fillable forms for onboarding and compliance."

#     keywords = extract_keywords_with_spacy(persona + " " + job)
#     print(f"INFO: Extracted keywords for boosting: {keywords}")
    
#     outlines = []
#     json_files = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != "final_output.json"]
#     for filename in json_files:
#         with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             outlines.append({
#                 "document": filename.replace(".json", ".pdf"),
#                 "title": data.get("title", ""),
#                 "outline": data.get("outline", [])
#             })
    
#     if not outlines:
#         print("FATAL ERROR: No outline files found.")
#         return

#     all_headings_details_map = {}
#     for doc in outlines:
#         doc_name = doc["document"]
#         all_headings_details_map[doc_name] = {}
#         for heading in doc.get("outline", []):
#             page_num = heading["page"]
#             if page_num not in all_headings_details_map[doc_name]:
#                 all_headings_details_map[doc_name][page_num] = []
#             all_headings_details_map[doc_name][page_num].append({
#                 "text": heading["text"],
#                 "level": heading["level"]
#             })

#     model = SentenceTransformer('./local_model')

#     ranked = rank_headings_and_title(outlines, persona, job, model, keywords)
#     topN = ranked[:5]

#     subsections = extract_subsections(topN, input_dir, all_headings_details_map)

#     final_output = {
#         "metadata": {
#             "input_documents": [os.path.basename(o["document"]) for o in outlines],
#             "persona": persona,
#             "job_to_be_done": job,
#             "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
#         },
#         "extracted_sections": [
#             {"document": item["document"], "page_number": item["page_number"], "section_title": item["section_title"], "importance_rank": item["importance_rank"]} 
#             for item in topN
#         ],
#         "subsection_analysis": subsections
#     }

#     output_path = os.path.join(output_dir, "final_output.json")
#     with open(output_path, "w", encoding='utf-8') as f:
#         json.dump(final_output, f, indent=2, ensure_ascii=False)
        
#     print(f"\n✅ Done! Output saved to {output_path}")

# if __name__ == "__main__":
#     main()





# import os
# import json
# import time
# import re
# import spacy
# import difflib
# from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine
# import fitz  # PyMuPDF

# # Import your Round 1A logic
# try:
#     from scripts.round1a_main import process_all_pdfs
#     print("INFO: Successfully imported 'process_all_pdfs'.")
# except ImportError:
#     print("FATAL ERROR: Could not import 'process_all_pdfs'. Check your 'scripts' folder setup.")
#     exit()

# def extract_keywords_with_spacy(text):
#     """Extracts noun chunks as keywords from text"""
#     try:
#         nlp = spacy.load("en_core_web_sm")
#     except OSError:
#         print("Downloading 'en_core_web_sm' model...")
#         spacy.cli.download("en_core_web_sm")
#         nlp = spacy.load("en_core_web_sm")
#     doc = nlp(text)
#     keywords = set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1)
#     keywords.update(token.text.lower() for token in doc if token.pos_ == "NOUN" and len(token.text) > 4)
#     return list(keywords)

# def rank_headings_and_title(outline_list, persona, job, model, keywords):
#     """Ranks headings based on a combination of semantic similarity and a strong keyword boost."""
#     query = persona + ". " + job
#     query_emb = model.encode(query)
#     ranked = []
#     KEYWORD_BOOST = 10.0
#     for doc in outline_list:
#         doc_name = doc["document"]
#         all_content = []
#         if doc.get("title"):
#             all_content.append({"text": doc["title"], "page": 0, "level": "H1"})
#         if doc.get("outline"):
#             all_content.extend(doc["outline"])
#         for item in all_content:
#             text, text_lower = item["text"], item["text"].lower()
#             heading_emb = model.encode(text)
#             semantic_score = 1 - cosine(query_emb, heading_emb)
#             keyword_hits = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
#             final_score = (keyword_hits * KEYWORD_BOOST) + semantic_score
#             if final_score > 0:
#                 ranked.append({"document": doc_name, "page_number": item["page"], "section_title": text, "level": item["level"], "score": final_score, "embedding": heading_emb})
#     ranked.sort(key=lambda x: x["score"], reverse=True)
#     return ranked

# def deduplicate_ranked_list(ranked_list, model, threshold=0.92):
#     """Filters the ranked list to remove semantically similar headings."""
#     final_list = []
#     for item in ranked_list:
#         if not final_list:
#             final_list.append(item)
#             continue
        
#         is_duplicate = False
#         for final_item in final_list:
#             similarity = 1 - cosine(item["embedding"], final_item["embedding"])
#             if similarity > threshold:
#                 is_duplicate = True
#                 break
        
#         if not is_duplicate:
#             final_list.append(item)
            
#         if len(final_list) >= 5:
#             break
            
#     # Re-assign importance rank after de-duplication
#     for idx, item in enumerate(final_list):
#         item["importance_rank"] = idx + 1
        
#     return final_list

# def extract_subsections(top_sections, pdf_dir, all_headings_details_map):
#     """Extracts content under a heading with hierarchical logic and a cover page fallback."""
#     results = []
#     MATCH_THRESHOLD = 0.85
#     for sec in top_sections:
#         pdf_path = os.path.join(pdf_dir, sec["document"])
#         snippet = ""
#         try:
#             doc = fitz.open(pdf_path)
#             page = doc[sec["page_number"]]
#             blocks = page.get_text("blocks")
#             current_level_num = int(sec['level'][1:])
#             headings_on_this_page = all_headings_details_map.get(sec["document"], {}).get(sec["page_number"], [])
            
#             for i, block in enumerate(blocks):
#                 block_text = block[4].strip().replace("\n", " ")
#                 if difflib.SequenceMatcher(None, sec["section_title"].lower(), block_text.lower()).ratio() > MATCH_THRESHOLD:
#                     content_parts = []
#                     for next_block in blocks[i+1:]:
#                         next_text = next_block[4].strip().replace("\n", " ")
#                         is_stopping_heading = False
#                         is_sub_heading = False
#                         for h_details in headings_on_this_page:
#                             if difflib.SequenceMatcher(None, h_details['text'].lower(), next_text.lower()).ratio() > 0.9:
#                                 next_heading_level_num = int(h_details['level'][1:])
#                                 if next_heading_level_num <= current_level_num:
#                                     is_stopping_heading = True
#                                 else:
#                                     is_sub_heading = True
#                                 break
#                         if is_stopping_heading:
#                             break
#                         if next_text:
#                             content_parts.append(next_text)
#                     snippet = " ".join(content_parts)
#                     break
            
#             # --- FALLBACK LOGIC FOR COVER PAGES ---
#             if not snippet and sec["page_number"] + 1 < len(doc):
#                 print(f"INFO: No content found for '{sec['section_title']}' on page {sec['page_number']}. Checking next page...")
#                 next_page = doc[sec["page_number"] + 1]
#                 next_page_blocks = next_page.get_text("blocks")
#                 for block in next_page_blocks:
#                     block_text = block[4].strip()
#                     if block_text and len(block_text.split()) > 5: # Find first meaningful block
#                         snippet = block_text.replace("\n", " ")
#                         break

#         except Exception as e:
#             print(f"ERROR reading {pdf_path}: {e}")
        
#         results.append({"document": sec["document"], "page_number": sec["page_number"], "refined_text": snippet[:2000]})
#     return results

# def main():
#     input_dir, output_dir = "./input", "./output"
#     os.makedirs(output_dir, exist_ok=True)
#     process_all_pdfs(input_dir, output_dir, ml_output_dir=None)

#     persona = "HR professional"
#     job = "Create and manage fillable forms for onboarding and compliance."
#     keywords = extract_keywords_with_spacy(persona + " " + job)
    
#     outlines = []
#     json_files = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != "final_output.json"]
#     for filename in json_files:
#         with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             outlines.append({"document": filename.replace(".json", ".pdf"), "title": data.get("title", ""), "outline": data.get("outline", [])})
    
#     if not outlines:
#         print("FATAL ERROR: No outline files found.")
#         return

#     all_headings_details_map = {}
#     for doc in outlines:
#         doc_name = doc["document"]
#         all_headings_details_map[doc_name] = {}
#         for heading in doc.get("outline", []):
#             page_num = heading["page"]
#             if page_num not in all_headings_details_map[doc_name]:
#                 all_headings_details_map[doc_name][page_num] = []
#             all_headings_details_map[doc_name][page_num].append({"text": heading["text"], "level": heading["level"]})

#     model = SentenceTransformer('./local_model')
    
#     # Rank all headings
#     ranked = rank_headings_and_title(outlines, persona, job, model, keywords)
    
#     # Deduplicate the ranked list to get diverse top results
#     topN = deduplicate_ranked_list(ranked, model)

#     # Extract subsections for the final de-duplicated list
#     subsections = extract_subsections(topN, input_dir, all_headings_details_map)

#     final_output = {
#         "metadata": {"input_documents": [os.path.basename(o["document"]) for o in outlines], "persona": persona, "job_to_be_done": job, "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
#         "extracted_sections": [{"document": item["document"], "page_number": item["page_number"], "section_title": item["section_title"], "importance_rank": item["importance_rank"]} for item in topN],
#         "subsection_analysis": subsections
#     }

#     output_path = os.path.join(output_dir, "final_output.json")
#     with open(output_path, "w", encoding='utf-8') as f:
#         json.dump(final_output, f, indent=2, ensure_ascii=False)
        
#     print(f"\n✅ Done! Output saved to {output_path}")

# if __name__ == "__main__":
#     main()












# import os
# import json
# import time
# import re
# import spacy
# import difflib
# from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine
# import fitz  # PyMuPDF

# # Import your Round 1A logic
# try:
#     from scripts.round1a_main import process_all_pdfs
#     print("INFO: Successfully imported 'process_all_pdfs'.")
# except ImportError:
#     print("FATAL ERROR: Could not import 'process_all_pdfs'. Check your 'scripts' folder setup.")
#     exit()

# def extract_keywords_with_spacy(text):
#     """Extracts noun chunks as keywords from text"""
#     try:
#         nlp = spacy.load("en_core_web_sm")
#     except OSError:
#         print("Downloading 'en_core_web_sm' model...")
#         spacy.cli.download("en_core_web_sm")
#         nlp = spacy.load("en_core_web_sm")
        
#     doc = nlp(text)
#     keywords = set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1)
#     keywords.update(token.text.lower() for token in doc if token.pos_ == "NOUN" and len(token.text) > 4)
#     return list(keywords)

# def rank_headings_and_title(outline_list, persona, job, model, keywords):
#     """Ranks headings based on a combination of semantic similarity and a strong keyword boost."""
#     query = persona + ". " + job
#     query_emb = model.encode(query)
    
#     ranked = []
#     KEYWORD_BOOST = 10.0

#     for doc in outline_list:
#         doc_name = doc["document"]
#         all_content = []
#         if doc.get("title"):
#             all_content.append({"text": doc["title"], "page": 0, "level": "H1"})
#         if doc.get("outline"):
#             all_content.extend(doc["outline"])

#         for item in all_content:
#             text, text_lower = item["text"], item["text"].lower()
#             heading_emb = model.encode(text)
#             semantic_score = 1 - cosine(query_emb, heading_emb)
            
#             # --- REVISED KEYWORD LOGIC: Use regex for precise, whole-word matching ---
#             keyword_hits = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
            
#             final_score = (keyword_hits * KEYWORD_BOOST) + semantic_score

#             if final_score > 0:
#                 ranked.append({
#                     "document": doc_name,
#                     "page_number": item["page"],
#                     "section_title": text,
#                     "level": item["level"],
#                     "score": final_score,
#                     "embedding": heading_emb
#                 })

#     ranked.sort(key=lambda x: x["score"], reverse=True)
#     return ranked

# def deduplicate_ranked_list(ranked_list, threshold=0.92):
#     """Filters the ranked list to remove semantically similar headings."""
#     final_list = []
#     for item in ranked_list:
#         if not final_list:
#             final_list.append(item)
#             continue
        
#         is_duplicate = False
#         for final_item in final_list:
#             if 1 - cosine(item["embedding"], final_item["embedding"]) > threshold:
#                 is_duplicate = True
#                 break
        
#         if not is_duplicate:
#             final_list.append(item)
            
#         if len(final_list) >= 5:
#             break
            
#     for idx, item in enumerate(final_list):
#         item["importance_rank"] = idx + 1
        
#     return final_list

# def extract_subsections(top_sections, pdf_dir, all_headings_details_map):
#     """Extracts content under a heading with hierarchical logic and a cover page fallback."""
#     results = []
#     MATCH_THRESHOLD = 0.85
#     for sec in top_sections:
#         pdf_path, snippet = os.path.join(pdf_dir, sec["document"]), ""
#         try:
#             doc = fitz.open(pdf_path)
#             page = doc[sec["page_number"]]
#             blocks = page.get_text("blocks")
#             current_level_num = int(sec['level'][1:])
#             headings_on_this_page = all_headings_details_map.get(sec["document"], {}).get(sec["page_number"], [])
            
#             for i, block in enumerate(blocks):
#                 block_text = block[4].strip().replace("\n", " ")
#                 if difflib.SequenceMatcher(None, sec["section_title"].lower(), block_text.lower()).ratio() > MATCH_THRESHOLD:
#                     content_parts = []
#                     for next_block in blocks[i+1:]:
#                         next_text = next_block[4].strip().replace("\n", " ")
#                         is_stopping_heading, is_sub_heading = False, False
#                         for h_details in headings_on_this_page:
#                             if difflib.SequenceMatcher(None, h_details['text'].lower(), next_text.lower()).ratio() > 0.9:
#                                 next_heading_level_num = int(h_details['level'][1:])
#                                 is_stopping_heading = next_heading_level_num <= current_level_num
#                                 is_sub_heading = not is_stopping_heading
#                                 break
#                         if is_stopping_heading: break
#                         if next_text:
#                             content_parts.append(f"**{next_text}**" if is_sub_heading else next_text)
#                     snippet = " ".join(content_parts)
#                     break
            
#             if not snippet and sec["page_number"] + 1 < len(doc):
#                 print(f"INFO: No content on page {sec['page_number']}. Checking next page...")
#                 next_page_blocks = doc[sec["page_number"] + 1].get_text("blocks")
#                 for block in next_page_blocks:
#                     block_text = block[4].strip()
#                     if block_text and len(block_text.split()) > 5:
#                         snippet = block_text.replace("\n", " ")
#                         break

#         except Exception as e:
#             print(f"ERROR reading {pdf_path}: {e}")
        
#         results.append({"document": sec["document"], "page_number": sec["page_number"], "refined_text": snippet[:2000]})
#     return results

# def main():
#     input_dir, output_dir = "./input", "./output"
#     os.makedirs(output_dir, exist_ok=True)
#     process_all_pdfs(input_dir, output_dir, ml_output_dir=None)

#     persona = "HR professional"
#     job = "Create and manage fillable forms for onboarding and compliance."
#     keywords = extract_keywords_with_spacy(persona + " " + job)
#     print(f"INFO: Extracted keywords for boosting: {keywords}")
#     outlines = []
#     json_files = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != "final_output.json"]
#     for filename in json_files:
#         with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
#             outlines.append({"document": filename.replace(".json", ".pdf"), **json.load(f)})
    
#     if not outlines:
#         print("FATAL ERROR: No outline files found.")
#         return

#     all_headings_details_map = {doc["document"]: {h["page"]: [] for h in doc.get("outline", [])} for doc in outlines}
#     for doc in outlines:
#         for heading in doc.get("outline", []):
#             all_headings_details_map[doc["document"]][heading["page"]].append({"text": heading["text"], "level": heading["level"]})

#     model = SentenceTransformer('./local_model')
#     ranked = rank_headings_and_title(outlines, persona, job, model, keywords)
#     topN = deduplicate_ranked_list(ranked)
#     subsections = extract_subsections(topN, input_dir, all_headings_details_map)

#     final_output = {
#         "metadata": {"input_documents": [os.path.basename(o["document"]) for o in outlines], "persona": persona, "job_to_be_done": job, "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
#         "extracted_sections": [{"document": item["document"], "page_number": item["page_number"], "section_title": item["section_title"], "importance_rank": item["importance_rank"]} for item in topN],
#         "subsection_analysis": subsections
#     }

#     output_path = os.path.join(output_dir, "final_output.json")
#     with open(output_path, "w", encoding='utf-8') as f:
#         json.dump(final_output, f, indent=2, ensure_ascii=False)
        
#     print(f"\n✅ Done! Output saved to {output_path}")

# if __name__ == "__main__":
#     main()

import os
import json
import time
import spacy
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import fitz
import re

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
    print("Enter persona")
    persona = "Travel Planner"
    print("Enter job to be done: ")
    job = "Plan a trip for 4 days for a friends group of 10."

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
    model = SentenceTransformer('all-MiniLM-L6-v2')

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









# import os
# import json
# import time
# import spacy
# from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine
# import fitz
# import re
# from rapidfuzz import fuzz

# # Your Round 1A logic (make sure path matches your project)
# # Ensure this import works correctly based on your project structure.
# # If main.py is in the root and scripts is a folder, you might need to adjust the Python path
# # or use a different import method depending on how you run the script.
# from scripts.round1a_main import process_all_pdfs

# # Load spaCy model
# print("Loading spaCy model...")
# try:
#     nlp = spacy.load("en_core_web_sm")
#     print("✅ spaCy model loaded.")
# except OSError:
#     print("❌ spaCy model 'en_core_web_sm' not found.")
#     print("Please run 'python -m spacy download en_core_web_sm' to install it.")
#     exit()


# def extract_and_rank_keywords(text):
#     """
#     Extracts keywords from text and ranks them:
#     - named entities first (more important)
#     - then noun chunks sorted by length
#     """
#     doc = nlp(text)
#     entities = {ent.text.lower() for ent in doc.ents if len(ent.text.strip()) > 1}
#     noun_chunks = {chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 1}
#     # Remove overlaps
#     noun_chunks = noun_chunks - entities
#     # Sort noun chunks by length desc
#     sorted_noun_chunks = sorted(list(noun_chunks), key=lambda x: len(x), reverse=True)
#     # Combine: entities first, then long noun chunks
#     keywords = list(entities) + sorted_noun_chunks
#     return keywords


# def rank_headings_and_titles(outline_list, persona, job, model, keywords, top_n=20):
#     """
#     Rank headings & titles: ensure top N cover different keywords first,
#     then fill remaining spots by highest score.
#     """
#     query = persona + ". " + job
#     print(f"Encoding query: '{query}'")
#     query_emb = model.encode(query)

#     all_candidates = []

#     for doc in outline_list:
#         doc_name = doc["document"]

#         # Title
#         title_text = doc.get("title", "")
#         if title_text:
#             title_emb = model.encode(title_text.lower())
#             base_score = 1 - cosine(query_emb, title_emb)
#             boost = compute_boost(title_text, keywords)
#             final_score = (base_score * boost) * 3.0  # title weight

#             all_candidates.append({
#                 "document": doc_name,
#                 "page_number": 0,
#                 "section_title": title_text,
#                 "level": "TITLE",
#                 "similarity": float(final_score),
#                 "keywords": [kw for kw in keywords if kw.lower() in title_text.lower()]
#             })

#         # Headings
#         for h in doc["outline"]:
#             heading_text = h["text"]
#             heading_emb = model.encode(heading_text.lower())
#             base_score = 1 - cosine(query_emb, heading_emb)
#             boost = compute_boost(heading_text, keywords)
#             weight = {"H1": 2.0, "H2": 1.5, "H3": 1.2, "H4": 1.0}.get(h["level"], 1.0)
#             final_score = (base_score * boost) * weight

#             all_candidates.append({
#                 "document": doc_name,
#                 "page_number": h["page"],
#                 "section_title": heading_text,
#                 "level": h["level"],
#                 "similarity": float(final_score),
#                 "keywords": [kw for kw in keywords if kw.lower() in heading_text.lower()]
#             })

#     if not all_candidates:
#         print("⚠️ No candidates were generated for ranking. Check if outlines were loaded correctly.")
#         return []

#     # Step 2: Pick top heading for each keyword first
#     selected = []
#     used_texts = set()

#     for kw in keywords:
#         kw_candidates = [c for c in all_candidates if kw in c["keywords"] and c["section_title"] not in used_texts]
#         if kw_candidates:
#             best = max(kw_candidates, key=lambda x: x["similarity"])
#             selected.append(best)
#             used_texts.add(best["section_title"])
#         if len(selected) >= top_n:
#             break

#     # Step 3: Fill remaining spots by highest score, skipping used ones
#     if len(selected) < top_n:
#         remaining_candidates = sorted(
#             [c for c in all_candidates if c["section_title"] not in used_texts],
#             key=lambda x: x["similarity"],
#             reverse=True
#         )
#         for c in remaining_candidates:
#             if len(selected) >= top_n:
#                 break
#             selected.append(c)
#             used_texts.add(c["section_title"])

#     # Step 4: Add importance_rank
#     for idx, item in enumerate(selected):
#         item["importance_rank"] = idx + 1

#     return selected


# def extract_subsections(top_sections, pdf_dir):
#     results = []
#     print(f"\n extracting subsections from top {len(top_sections[:3])} sections...")
#     for sec in top_sections[:3]:  # Limit to top 3 for snippet extraction
#         pdf_path = os.path.join(pdf_dir, sec["document"])
#         snippet = ""
#         try:
#             doc = fitz.open(pdf_path)
#             page = doc[sec["page_number"]]
#             text = page.get_text()
            
#             # Find a paragraph containing the section title for a relevant snippet
#             paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
#             for para in paragraphs:
#                 # Use fuzzy matching for robustness
#                 if fuzz.partial_ratio(sec["section_title"].lower(), para.lower()) > 80:
#                     snippet = para
#                     break
            
#             if not snippet and paragraphs:
#                 snippet = paragraphs[0] # Fallback to the first paragraph
            
#             results.append({
#                 "document": sec["document"],
#                 "page_number": sec["page_number"],
#                 "refined_text": snippet[:1000]  # limit length
#             })
#         except Exception as e:
#             print(f"❌ Error reading PDF {pdf_path} on page {sec['page_number']}: {e}")

#     return results


# def compute_boost(text, keywords):
#     """
#     Computes a score boost if keywords are found in the text.
#     Uses fuzzy matching for more robust detection.
#     """
#     text_lower = text.lower()
#     max_boost = 1.0
#     for kw in keywords:
#         kw_lower = kw.lower()
#         # Use partial ratio to find keyword within text
#         partial_score = fuzz.partial_ratio(kw_lower, text_lower)
#         if partial_score > 95: # Very high confidence match
#             max_boost = max(max_boost, 2.0) # Strong boost
#         elif partial_score > 80: # Good partial match
#             max_boost = max(max_boost, 1.5) # Medium boost
#     return max_boost


# def main():
#     print("🚀 Script Started")
    
#     # --- Configuration ---
#     input_dir = "./pdfs"
#     # This is where the final output of this script will be saved.
#     output_dir = "./output"
#     # *** LIKELY FIX: This is where the intermediate outlines from Round1A are located. ***
#     # Your folder structure suggests they are in 'title_and_headings'.
#     # outline_source_dir = "./title_and_headings" 
    
#     os.makedirs(output_dir, exist_ok=True)

#     # Step 1: Run Round 1A to create outline JSONs
#     print(f"📄 Calling process_all_pdfs to extract outlines from '{input_dir}' and save to '{output_dir}'...")
#     # We pass the correct source directory to the function.
#     process_all_pdfs(input_dir, output_dir)
#     print("✅ Outline extraction process finished.")

#     # Step 2: Define persona & job
#     persona = "Travel Planner"
#     job = "Plan a trip for 4 days for a friends group of 10."
#     print(f"\n👤 Persona: {persona}")
#     print(f"📝 Job: {job}")

#     # Step 3: Extract and rank keywords
#     stop_words = {"a", "an", "the", "for", "and", "or", "of", "in", "to", "with", "on", "at", "from", "by", "as", "is", "are"}
#     job_keywords_raw = re.findall(r'\b\w+\b', job.lower())
#     persona_keywords_raw = re.findall(r'\b\w+\b', persona.lower())
    
#     seen = set()
#     job_keywords = [w for w in job_keywords_raw if w not in stop_words and not (w in seen or seen.add(w))]
#     persona_keywords = [w for w in persona_keywords_raw if w not in stop_words and not (w in seen or seen.add(w))]
    
#     keywords = job_keywords + persona_keywords
#     print(f"🧠 Extracted & ordered keywords: {keywords}")

#     # Step 4: Load outlines from the correct source directory
#     print(f"\n📂 Loading outlines from: '{output_dir}'")
#     outlines = []
#     if not os.path.exists(output_dir) or not os.listdir(output_dir):
#         print(f"❌ Error: No outline files found in '{output_dir}'.")
#         print("Please ensure the 'process_all_pdfs' script is correctly generating JSON files in that directory.")
#         exit()
        
#     for filename in os.listdir(output_dir):
#         if filename.endswith(".json"):
#             try:
#                 with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#                     outlines.append({
#                         "document": filename.replace(".json", ".pdf"),
#                         "title": data.get("title", ""),
#                         "outline": data.get("outline", [])
#                     })
#             except (json.JSONDecodeError, KeyError) as e:
#                 print(f"⚠️ Warning: Could not process file {filename}. Error: {e}")
#     print(f"✅ Loaded {len(outlines)} outline files.")

#     if not outlines:
#         print("❌ No valid outlines were loaded. Exiting.")
#         exit()

#     # Step 5: Load embedding model
#     print("\n🔄 Loading sentence transformer model...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     print("✅ Model loaded.")

#     # Step 6: Rank sections
#     print("\n⚖️ Ranking sections based on persona and job...")
#     ranked = rank_headings_and_titles(outlines, persona, job, model, keywords)
#     topN = ranked[:20]
#     print(f"✅ Found {len(topN)} top sections.")

#     # Step 7: Extract subsections
#     subsections = extract_subsections(topN, input_dir)
#     print(f"✅ Extracted {len(subsections)} subsection snippets.")

#     # Step 8: Build final output
#     final_output = {
#         "metadata": {
#             "input_documents": [o["document"] for o in outlines],
#             "persona": persona,
#             "job_to_be_done": job,
#             "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
#         },
#         "extracted_sections": [
#             {
#                 "document": item["document"],
#                 "page_number": item["page_number"],
#                 "section_title": item["section_title"],
#                 "level": item["level"],
#                 "importance_rank": item["importance_rank"]
#             } for item in topN
#         ],
#         "subsection_analysis": subsections
#     }

#     # Step 9: Save output
#     output_path = os.path.join(output_dir, "challenge1b_output.json")
#     print(f"\n💾 Saving final output to {output_path}")
#     with open(output_path, "w", encoding='utf-8') as f:
#         json.dump(final_output, f, indent=4)

#     print("\n🎉 Done! Script finished successfully.")


# if __name__ == "__main__":
#     main()
