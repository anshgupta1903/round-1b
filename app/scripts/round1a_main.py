# import fitz  # PyMuPDF
# import json
# import os
# import re
# import time
# from collections import Counter
# import pandas as pd
# import joblib

# # --- 1. Load the Pre-trained Model and Label Mapping ---
# try:
#     model = joblib.load('model/heading_model.pkl')
#     label_mapping = joblib.load('model/label_mapping.pkl')
#     print("INFO: Successfully loaded ML model.")
# except FileNotFoundError:
#     print("WARNING: ML model not found. Running in rule-based-only mode.")
#     model = None
#     label_mapping = None


# def detect_headers_and_footers(doc, line_threshold=0.4):
#     """
#     Detects repeating text that is likely a header or footer based on
#     frequency and position on the page. This is a critical first step.
#     """
#     page_count = len(doc)
#     if page_count < 4: return set()

#     text_counts = Counter()
#     # Scan the middle half of the document to avoid title and reference pages
#     start_page = page_count // 4
#     end_page = page_count - start_page

#     for page_num in range(start_page, end_page):
#         page = doc[page_num]
#         page_height = page.rect.height
#         blocks = page.get_text("blocks")
#         for b in blocks:
#             # Check a wider vertical area: top 20% and bottom 15% of the page
#             if b[1] < page_height * 0.20 or b[3] > page_height * 0.85:
#                 line_text = b[4].strip().replace('\n', ' ')
#                 if 5 < len(line_text) < 100 and not line_text.endswith('.'):
#                     text_counts[line_text] += 1
    
#     ignore_set = set()
#     # Lower the threshold to catch text that appears on 40% of scanned pages
#     min_occurrences = (end_page - start_page) * line_threshold
#     for text, count in text_counts.items():
#         if count >= min_occurrences:
#             ignore_set.add(text)
            
#     print(f"INFO: Detected {len(ignore_set)} repeating lines to ignore as headers/footers.")
#     return ignore_set

# def is_line_in_table(line_bbox, page_table_areas):
#     """Checks if a line's bounding box is inside any of a page's table areas."""
#     if not page_table_areas:
#         return False
    
#     l_x0, l_y0, l_x1, l_y1 = line_bbox
#     for t_bbox in page_table_areas:
#         t_x0, t_y0, t_x1, t_y1 = t_bbox
#         # Check for containment. A line is in a table if its bbox is inside the table's bbox.
#         if l_x0 >= t_x0 and l_y0 >= t_y0 and l_x1 <= t_x1 and l_y1 <= t_y1:
#             return True
#     return False

# def get_dominant_style(line):
#     """
#     Determines the most common (dominant) style in a line of text.
#     This is more robust than just checking the first span.
#     """
#     if not line["spans"]:
#         return (10, False) # Default style

#     style_counts = Counter()
#     for span in line["spans"]:
#         # More robust check for bold fonts
#         is_bold = bool(re.search(r'bold|black|heavy', span["font"], re.IGNORECASE))
#         style = (round(span["size"]), is_bold)
#         # We weigh the style by the length of the text in the span
#         style_counts[style] += len(span["text"].strip())
    
#     # Return the most common style
#     return style_counts.most_common(1)[0][0]

# def is_mostly_uppercase(s):
#     """
#     Checks if a string is predominantly uppercase. More robust than isupper().
#     """
#     letters = [char for char in s if char.isalpha()]
#     if not letters:
#         return False
#     uppercase_letters = [char for char in letters if char.isupper()]
#     return (len(uppercase_letters) / len(letters)) > 0.8

# def get_page_layout(page, threshold=0.3):
#     """
#     Analyzes the layout of a page to determine if it is single or multi-column.
#     Returns the number of detected columns (1 or 2).
#     """
#     page_width = page.rect.width
#     midpoint = page_width / 2
    
#     blocks = page.get_text("blocks")
#     if not blocks:
#         return 1 # Default to 1 column if no text

#     left_blocks = 0
#     right_blocks = 0
    
#     for b in blocks:
#         if b[2] < midpoint: # Block ends before midpoint
#             left_blocks += 1
#         elif b[0] > midpoint: # Block starts after midpoint
#             right_blocks += 1

#     total_sided_blocks = left_blocks + right_blocks
#     if total_sided_blocks == 0:
#         return 1

#     # Heuristic: If there are a significant number of blocks on both sides, it's a 2-column layout.
#     if (left_blocks > 0 and right_blocks > 0):
#         if (left_blocks / total_sided_blocks > threshold) and (right_blocks / total_sided_blocks > threshold):
#             return 2
    
#     return 1

# def process_pdf(pdf_path, ml_output_path=None):
#     """
#     Processes a PDF using a hybrid ML and rule-based filtering approach.
#     """
#     doc = fitz.open(pdf_path)
#     ignored_texts = detect_headers_and_footers(doc)
    
#     table_areas = {}
#     for page_num, page in enumerate(doc):
#         tables = page.find_tables()
#         if tables.tables:
#             table_areas[page_num] = [t.bbox for t in tables]
    
#     if table_areas:
#         print(f"INFO: Detected tables on pages: {list(table_areas.keys())}")
        
#     all_lines = []
#     style_counts = Counter()
#     page_heights = {}
    
#     for page_num, page in enumerate(doc):
#         page_width = page.rect.width
#         page_heights[page_num] = page.rect.height
#         page_table_bboxes = table_areas.get(page_num, [])
        
#         # --- NEW: Determine page layout for correct reading order ---
#         num_columns = get_page_layout(page)
#         page_midpoint = page_width / 2

#         blocks = page.get_text("dict")["blocks"]
#         for block in blocks:
#             if "lines" in block:
#                 for line in block["lines"]:
#                     if not line["spans"]: continue

#                     if is_line_in_table(line["bbox"], page_table_bboxes):
#                         continue

#                     line_text = "".join(span["text"] for span in line["spans"]).strip()
#                     if not line_text or line_text in ignored_texts: continue

#                     is_date = False
#                     text_lower = line_text.lower()
#                     if re.search(r'\b\d{4}\b', text_lower):
#                         if any(month in text_lower for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
#                             if len(line_text.split()) <= 4:
#                                 is_date = True
#                     if is_date:
#                         continue

#                     if re.match(r'^Page \d+\s*of\s*\d+$', line_text, re.I): 
#                         continue

#                     style = get_dominant_style(line)
#                     style_counts[style] += 1

#             # --- NEW: Add column index to each line ---
#                     line_center_x = (line["bbox"][0] + line["bbox"][2]) / 2
#                     column_index = 0
#                     if num_columns == 2 and line_center_x > page_midpoint:

#                         column_index = 1

#                     for idx, span in enumerate(line["spans"]):
#                         all_lines.append({
#                     "page": page_num,
#                     "page_number": page_num,
#                     "block_number": block.get("number", 0),
#                     "line_number": line.get("number", 0),
#                     "span_number": idx,  # index of span in line
#                     "text": span["text"],
#                     "font": span["font"],
#                     "size": span["size"],
#                     "color": span["color"],
#                     "is_bold": bool(re.search(r'bold|black|heavy', span["font"], re.IGNORECASE)),
#                     "is_italic": "italic" in span["font"].lower(),
#                     "x0": line["bbox"][0],
#                     "y0": line["bbox"][1],
#                     "x1": line["bbox"][2],
#                     "y1": line["bbox"][3],
#                     "column": column_index,
#                     "id": f"{page_num}-{line['bbox'][1]}-{idx}",
#                     "style": style,
#                 })


#     if not all_lines:
#         return {"title": "", "outline": []}
        
#     df = pd.DataFrame(all_lines)
#     if df.empty:
#         return {"title": "", "outline": []}
#     # --- NEW: Pass dataframe to pipeline for preprocessing ---
#     from scripts.pipeline import preprocess_for_model
#     df_preprocessed = preprocess_for_model(df)

    

#     if model:
#         # Use only the columns your ML model expects
#         predictions_encoded = model.predict(
#     df_preprocessed[['size', 'is_bold', 'is_italic', 'x0','y0','x1','y1', 'font_encoded', 'color_encoded']])

#         df['ml_level'] = [label_mapping.get(pred, 'Other') for pred in predictions_encoded]
#         print("INFO: ML model predictions: ")
#         initial_candidates = df[df['ml_level'] != 'Other'].to_dict('records')
#     else:
#         df['ml_level'] = 'Other'
#         initial_candidates = []


#     page_0_height = page_heights.get(0, 1000)
#     page_0_lines_top_half = sorted(
#         [line for line in all_lines if line["page"] == 0 and line["y0"] < page_0_height / 2],
#         key=lambda x: (x["column"], x["y0"])
#     )

#     doc_title = ""
#     title_line_ids = set()
    
#     title_breaker_keywords = ["summary", "background", "introduction", "table of contents", "abstract", "keywords"]
#     author_affiliation_keywords = ["department", "university", "college", "institute", "@"]

#     if page_0_lines_top_half:
#         try:
#             max_size_page_0 = max(line["size"] for line in page_0_lines_top_half)
#             first_potential_title_lines = [line for line in page_0_lines_top_half if line["size"] == max_size_page_0]
#             if not first_potential_title_lines:
#                  raise ValueError("No lines found to be part of the title.")

#             first_title_line = first_potential_title_lines[0]
#             start_index = page_0_lines_top_half.index(first_title_line)
            
#             title_lines_text = []
#             last_line = None

#             for i in range(start_index, len(page_0_lines_top_half)):
#                 current_line = page_0_lines_top_half[i]
#                 current_text_lower = current_line["text"].lower()

#                 if last_line:
#                     if abs(current_line["y0"] - last_line["y0"]) > last_line["size"] * 2.5: break
#                     if current_line["size"] < first_title_line["size"] * 0.7: break
#                     if re.match(r"^(chapter|section|part|appendix|\d+(\.\d+)*\.?)\s", current_text_lower, re.I): break
#                     if any(keyword in current_text_lower for keyword in title_breaker_keywords): break
#                     if any(keyword in current_text_lower for keyword in author_affiliation_keywords): break
#                     # --- NEW: Stop title if it looks like a bullet point ---
#                     if re.match(r"^[•●*+-]\s*", current_line["text"]): break

#                 title_lines_text.append(current_line["text"])
#                 title_line_ids.add(current_line["id"])
#                 last_line = current_line
            
#             doc_title = " ".join(title_lines_text)

#         except (ValueError, IndexError):
#             doc_title = ""


#     if title_line_ids:
#         print(f"INFO: Identified title: '{doc_title}'. Excluding {len(title_line_ids)} lines from heading analysis.")
#         all_lines = [line for line in all_lines if line['id'] not in title_line_ids]

#     non_bold_styles = [s for s, c in style_counts.items() if not s[1]]
#     if not non_bold_styles:
#         body_style = style_counts.most_common(1)[0][0] if style_counts else (10, False)
#     else:
#         body_style = Counter({s: style_counts[s] for s in non_bold_styles}).most_common(1)[0][0]

#     df = pd.DataFrame(all_lines)
#     if df.empty:
#         return {"title": doc_title, "outline": []}
#     # --- NEW: pass dataframe to pipeline for preprocessing ---
#     from scripts.pipeline import preprocess_for_model
#     df_preprocessed = preprocess_for_model(df)


#     if model:
#         # Use df_preprocessed and correct columns (should match features used in train.py)
#         predictions_encoded = model.predict(
#     df_preprocessed[['size', 'is_bold', 'is_italic', 'x0','y0','x1','y1', 'font_encoded', 'color_encoded']])

#         df['ml_level'] = [label_mapping.get(pred, 'Other') for pred in predictions_encoded]
#         initial_candidates = df[df['ml_level'] != 'Other'].to_dict('records')
#         print("INFO: ML model predictions: ")
#         print(predictions_encoded)
#     else:
#         df['ml_level'] = 'Other'
#         initial_candidates = []

#     all_candidates = {cand['id']: cand for cand in initial_candidates}

#     for line in all_lines:
#         if line['id'] not in all_candidates:
#             text = line['text']
#             is_stylistically_distinct = (line['size'] > body_style[0]) or (line['is_bold'] and not body_style[1])
#             is_short_enough = len(text.split()) < 25
#             is_numbered = re.match(r"^\d+(\.\d+)*\.?\s", text) or re.match(r"^Appendix [A-Z]:", text, re.I)

#             if is_short_enough and (is_stylistically_distinct or is_numbered):
#                 line['ml_level'] = 'H2'
#                 all_candidates[line['id']] = line

#     comprehensive_candidates = list(all_candidates.values())

#     candidate_texts = [c['text'] for c in comprehensive_candidates]
#     candidate_counts = Counter(candidate_texts)
#     repeating_texts_to_ignore = {text for text, count in candidate_counts.items() if count > 2}

#     if repeating_texts_to_ignore:
#         print(f"INFO: Pre-emptively ignoring potential headings that repeat frequently: {list(repeating_texts_to_ignore)}")

#     refined_headings = []
    
#     high_confidence_keywords = [
#         "abstract", "keywords", "introduction", "background", "related work",
#         "methodology", "methods", "materials and methods", "experimental setup",
#         "results", "discussion", "conclusion", "summary",
#         "acknowledgements", "references", "bibliography", "appendix", "author",
#         "table of contents", "list of figures", "list of tables"
#     ]

#     for cand in comprehensive_candidates:
#         text = cand["text"]
#         style = cand["style"]
        
#         is_centered = abs((cand["x0"] + cand["x1"]) / 2 - (page_width / 2)) < 20
#         is_numbered = re.match(r"^(\d+(\.\d+)*)\.?\s", text) or re.match(r"^Appendix [A-Z]:", text, re.I)
#         is_short = len(text.split()) <= 12
#         is_upper = is_mostly_uppercase(text)
#         if style[0] == body_style[0] and not style[1]:
#             continue 
#         if text in repeating_texts_to_ignore:
#             continue
            
#         if any(keyword == text.lower().strip().rstrip(':') for keyword in high_confidence_keywords):
#             cand["level"] = "H1"
#             refined_headings.append(cand)
#             continue

#         elif re.match(r"^(Appendix [A-Z]:|\d+(\.\d+)*\.?\s)", text):
#             # --- MODIFIED: More robust check for numbered headings using uppercase ratio ---
#             numeric_match_obj = re.match(r"^(\d+(\.\d+)*)\.?\s*(.*)", text)
#             if numeric_match_obj:
#                 text_part = numeric_match_obj.group(3)
#                 if is_mostly_uppercase(text_part) and len(text_part) > 3:
#                     pass
#                 elif style == body_style:
#                     continue
            
#             numeric_match = re.match(r"^(\d+(\.\d+)*)\.?\s", text)
#             if numeric_match:
#                 number_part = numeric_match.group(1)
#                 depth = number_part.count('.') + 1
#                 cand["level"] = f"H{min(depth, 6)}"
#             else:
#                 cand["level"] = "H1"
#             refined_headings.append(cand)
#             continue

#         elif style[0] == body_style[0] and style[1] and not body_style[1]:
#     # Only consider heading if text ends with ':' OR is centered
#             text = cand["text"].strip()
#             is_centered = abs((cand["x0"] + cand["x1"]) / 2 - (page_width / 2)) < 20
#             if (text.endswith(':') or is_centered) :
#                 cand["level"] = "H3"
#                 refined_headings.append(cand)

#             print(f"DEBUG: text='{text}' style={style} ml_level={cand.get('ml_level')} body_style={body_style}")


#         elif style[0] > body_style[0]:
#     # only add if bold or text is short (like heading)
#             if style[1] or len(text.split()) < 8:
#                 cand["level"] = cand.get('ml_level', 'H2')
#                 refined_headings.append(cand)
        
        
#         elif style[0] == body_style[0] and style[1] and not body_style[1]:
#     # only consider heading if ends with ':' or is centered
#             is_centered = abs((cand["x0"] + cand["x1"]) / 2 - (page_width / 2)) < 20
#             if text.endswith(':') or is_centered:
#                 cand["level"] = "H3"
#                 refined_headings.append(cand)
#         else:
#     # skip plain body-sized non-bold text
#             continue
        

#     # --- MODIFIED: Sort by page, then column, then vertical position ---
#     sorted_headings = sorted(refined_headings, key=lambda x: (x['page'], x['column'], x['y0']))
#     outline = []
#     i = 0
#     min_heading_size = body_style[0]
#     in_references_section = False

#     while i < len(sorted_headings):
#         current_heading = sorted_headings[i]
#         j = i + 1
#         while j < len(sorted_headings):
#             prev_line = sorted_headings[j-1]
#             next_line = sorted_headings[j]

#             if next_line["page"] == current_heading["page"] and \
#                next_line["column"] == current_heading["column"] and \
#                abs(next_line["y0"] - prev_line["y0"]) < prev_line["size"] * 1.8 and \
#                next_line['size'] >= current_heading['size'] * 0.85:
                
#                 if not re.match(r"^\d+(\.\d+)*\.?\s", next_line["text"]):
#                     current_heading["text"] += " " + next_line["text"]
#                     j += 1
#                 else:
#                     break 
#             else:
#                 break
        
#         text = current_heading["text"].strip()
        
#         is_rejected = False
#         text_lower = text.lower()

#         if any(keyword in text_lower for keyword in ["references", "bibliography"]):
#             in_references_section = True
        
#         if in_references_section and not any(keyword in text_lower for keyword in ["references", "bibliography"]):
#              if not re.match(r"^(Appendix [A-Z]|\d+(\.\d+)*)\s", text):
#                 is_rejected = True

#         if text.endswith(('.', ',', ';')) and len(text.split()) > 20: is_rejected = True
#         if len(text.split()) > 25: is_rejected = True
#         if re.match(r"^[•●*+-]\s*", text): is_rejected = True
#         if any(keyword in text_lower for keyword in author_affiliation_keywords): is_rejected = True
        
#         # --- NEW: More specific rejection rules for metadata and list-like items ---
#         if re.search(r'original research article|section:|doi:|inclusion criteria|exclusion criteria', text_lower):
#             is_rejected = True
            
#         # Reject likely author lists (contain numbers, commas, and are title-cased)
#         if bool(re.search(r'\d', text)) and bool(re.search(r',', text)) and text.istitle():
#             is_rejected = True

#         if text.isupper() and len(text.split()) < 5 and not current_heading.get('is_centered', 0):

#             cleaned_text_lower = text_lower.strip().rstrip(':')
#             if not any(keyword == cleaned_text_lower for keyword in high_confidence_keywords):
#                 is_rejected = True

#         if current_heading["size"] < min_heading_size and not current_heading['is_bold']: is_rejected = True
#         if re.fullmatch(r"[\d\W_]+", text) or len(text) < 4: is_rejected = True
        
#         if not is_rejected:
#             if 3 < len(text) < 250:
#                 outline.append({"level": current_heading["level"], "text": text, "page": current_heading["page"]})
        
#         i = j
        
#     if len(all_lines) < 30 and not outline and all_lines:
#         largest_line = max(all_lines, key=lambda x: x['size'])
#         outline.append({"level": "H1", "text": largest_line['text'], "page": largest_line['page']})

#     return {"title": doc_title, "outline": outline}



# def process_all_pdfs(input_dir, output_dir, ml_output_dir):
#     """
#     Processes all PDF files in a given directory.
#     """
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith(".pdf"):
#             pdf_path = os.path.join(input_dir, filename)
#             print(f"--- Processing {filename} ---")
#             start_time = time.time()
            
#             base_filename = os.path.splitext(filename)[0]
#             json_output_path = os.path.join(output_dir, base_filename + ".json")
            
#             csv_output_path = None
#             if ml_output_dir:
#                 if not os.path.exists(ml_output_dir):
#                     os.makedirs(ml_output_dir)
#                 csv_output_path = os.path.join(ml_output_dir, base_filename + "_ml_predictions.csv")

#             output_data = process_pdf(pdf_path, ml_output_path=csv_output_path)
            
#             with open(json_output_path, 'w', encoding='utf-8') as f:
#                 json.dump(output_data, f, indent=4)
#             end_time = time.time()
#             print(f"--- Finished {filename} in {end_time - start_time:.2f} seconds. ---")

# # if __name__ == "__main__":
# #     INPUT_DIR = "../input"
# #     OUTPUT_DIR = "../output"
# #     ML_OUTPUT_DIR = "ml_output"
# #     if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
# #     if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
# #     if not os.path.exists(ML_OUTPUT_DIR): os.makedirs(ML_OUTPUT_DIR)
# #     process_all_pdfs(INPUT_DIR, OUTPUT_DIR, ML_OUTPUT_DIR)







# import fitz  # PyMuPDF
# import json
# import os
# import re
# import time
# from collections import Counter
# import pandas as pd
# import joblib

# # --- 1. Load the Pre-trained Model and Label Mapping ---
# try:
#     model = joblib.load('./model/heading_model.pkl')
#     label_mapping = joblib.load('./model/label_mapping.pkl')
#     print("INFO: Successfully loaded ML model.")
# except FileNotFoundError:
#     print("WARNING: ML model not found. Running in rule-based-only mode.")
#     model = None
#     label_mapping = None


# def detect_headers_and_footers(doc, line_threshold=0.4):
#     page_count = len(doc)
#     if page_count < 4: return set()
#     text_counts = Counter()
#     start_page, end_page = page_count // 4, page_count - (page_count // 4)
#     for page_num in range(start_page, end_page):
#         page = doc[page_num]
#         blocks = page.get_text("blocks")
#         for b in blocks:
#             if b[1] < page.rect.height * 0.20 or b[3] > page.rect.height * 0.85:
#                 line_text = b[4].strip().replace('\n', ' ')
#                 if 5 < len(line_text) < 100 and not line_text.endswith('.'):
#                     text_counts[line_text] += 1
#     ignore_set = set()
#     min_occurrences = (end_page - start_page) * line_threshold
#     for text, count in text_counts.items():
#         if count >= min_occurrences:
#             ignore_set.add(text)
#     print(f"INFO: Detected {len(ignore_set)} repeating lines to ignore as headers/footers.")
#     return ignore_set

# def get_dominant_style(line):
#     if not line["spans"]: return (10, False)
#     style_counts = Counter()
#     for span in line["spans"]:
#         is_bold = bool(re.search(r'bold|black|heavy', span["font"], re.IGNORECASE))
#         style = (round(span["size"]), is_bold)
#         style_counts[style] += len(span["text"].strip())
#     return style_counts.most_common(1)[0][0]

# def get_page_layout(page):
#     page_width = page.rect.width
#     midpoint = page_width / 2
#     blocks = page.get_text("blocks")
#     if not blocks: return 1
#     left_blocks = sum(1 for b in blocks if b[2] < midpoint)
#     right_blocks = sum(1 for b in blocks if b[0] > midpoint)
#     total_sided = left_blocks + right_blocks
#     if total_sided > 0 and left_blocks > 0 and right_blocks > 0:
#         if (left_blocks / total_sided > 0.3) and (right_blocks / total_sided > 0.3):
#             return 2
#     return 1

# def process_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     ignored_texts = detect_headers_and_footers(doc)
#     all_lines, style_counts, page_heights = [], Counter(), {}

#     for page_num, page in enumerate(doc):
#         page_width, page_heights[page_num] = page.rect.width, page.rect.height
#         num_columns = get_page_layout(page)
#         page_midpoint = page_width / 2
#         blocks = page.get_text("dict")["blocks"]
#         for block in blocks:
#             if "lines" in block:
#                 for line in block["lines"]:
#                     if not line["spans"]: continue
#                     line_text = "".join(span["text"] for span in line["spans"]).strip()
#                     if not line_text or line_text in ignored_texts: continue
#                     if re.match(r'^Page \d+\s*of\s*\d+$', line_text, re.I): continue
                    
#                     style = get_dominant_style(line)
#                     style_counts[style] += 1
#                     line_center_x = (line["bbox"][0] + line["bbox"][2]) / 2
#                     column_index = 1 if num_columns == 2 and line_center_x > page_midpoint else 0

#                     all_lines.append({
#                         "page": page_num, "text": line_text, "font": line["spans"][0]["font"],
#                         "size": style[0], "is_bold": style[1], "y0": line["bbox"][1],
#                         "column": column_index, "id": f'{page_num}-{line["bbox"][1]}'
#                     })

#     if not all_lines: return {"title": "", "outline": []}
    
#     # Title Extraction Logic (remains the same)
#     page_0_height = page_heights.get(0, 1000)
#     page_0_lines = sorted([line for line in all_lines if line["page"] == 0 and line["y0"] < page_0_height / 2], key=lambda x: (x["column"], x["y0"]))
#     doc_title, title_line_ids = "", set()
#     if page_0_lines:
#         try:
#             max_size = max(line["size"] for line in page_0_lines)
#             potential_title_lines = [line for line in page_0_lines if line["size"] == max_size]
#             if potential_title_lines:
#                 title_lines_text = []
#                 start_index = page_0_lines.index(potential_title_lines[0])
#                 for i in range(start_index, len(page_0_lines)):
#                     line = page_0_lines[i]
#                     if i > start_index and (line["size"] < max_size * 0.8 or (line["y0"] - page_0_lines[i-1]["y0"]) > line["size"] * 2):
#                         break
#                     title_lines_text.append(line["text"])
#                     title_line_ids.add(line["id"])
#                 doc_title = " ".join(title_lines_text)
#         except (ValueError, IndexError):
#             doc_title = ""
    
#     # Filter out title lines from heading consideration
#     all_lines = [line for line in all_lines if line['id'] not in title_line_ids]
#     if not all_lines: return {"title": doc_title, "outline": []}

#     non_bold_styles = [s for s, c in style_counts.items() if not s[1]]
#     body_style = Counter({s: style_counts[s] for s in non_bold_styles}).most_common(1)[0][0] if non_bold_styles else style_counts.most_common(1)[0][0]

#     # --- REFINED HEADING LOGIC STARTS HERE ---
#     # 1. Gather all potential heading candidates
#     all_candidates = []
#     for line in all_lines:
#         text = line['text']
#         is_distinct = (line['size'] > body_style[0]) or (line['is_bold'] and not body_style[1])
#         is_numbered = re.match(r"^\d+(\.\d+)*\.?\s", text)
#         if (len(text.split()) < 25) and (is_distinct or is_numbered):
#             all_candidates.append(line)

#     # 2. First Pass: Extract high-confidence keywords as H1
#     high_confidence_keywords = ["abstract", "introduction", "conclusion", "references", "appendix", "table of contents"]
#     keyword_headings = []
#     other_headings = []
#     for cand in all_candidates:
#         text_lower = cand["text"].lower().strip().rstrip(':')
#         if text_lower in high_confidence_keywords:
#             cand["level"] = "H1"
#             keyword_headings.append(cand)
#         else:
#             other_headings.append(cand)
            
#     # 3. Second Pass: Rank remaining headings by font size
#     if other_headings:
#         unique_sizes = sorted(list(set(h['size'] for h in other_headings)), reverse=True)
#         # Create a map from size to H-level (H1, H2, H3...)
#         size_to_level_map = {size: f"H{i + 1}" for i, size in enumerate(unique_sizes)}
        
#         for cand in other_headings:
#             cand["level"] = size_to_level_map.get(cand['size'], 'H4') # Default to H4 for smaller sizes
    
#     # 4. Combine and process final headings
#     final_headings = keyword_headings + other_headings
#     sorted_headings = sorted(final_headings, key=lambda x: (x['page'], x['column'], x['y0']))

#     # 5. Merge multi-line headings and perform final filtering
#     outline, i = [], 0
#     while i < len(sorted_headings):
#         current_heading = sorted_headings[i]
#         j = i + 1
#         # Merge consecutive lines that form a single heading
#         while j < len(sorted_headings):
#             next_heading = sorted_headings[j]
#             if next_heading["page"] == current_heading["page"] and \
#                (next_heading["y0"] - current_heading["y0"]) < current_heading["size"] * 1.8 and \
#                next_heading["level"] == current_heading["level"]:
#                 current_heading["text"] += " " + next_heading["text"]
#                 current_heading["y0"] = next_heading["y0"] # Update position
#                 j += 1
#             else:
#                 break
        
#         text = current_heading["text"].strip()
#         # Final filtering for quality
#         if 3 < len(text) < 250 and not text.endswith(('.', ',')):
#              outline.append({"level": current_heading["level"], "text": text, "page": current_heading["page"]})
#         i = j

#     return {"title": doc_title, "outline": outline}


# def process_all_pdfs(input_dir, output_dir):
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith(".pdf"):
#             pdf_path = os.path.join(input_dir, filename)
#             print(f"--- Processing {filename} ---")
#             start_time = time.time()
#             output_data = process_pdf(pdf_path)
#             json_output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")
#             with open(json_output_path, 'w', encoding='utf-8') as f:
#                 json.dump(output_data, f, indent=4)
#             print(f"--- Finished {filename} in {time.time() - start_time:.2f} seconds. ---")






# import fitz  # PyMuPDF
# import json
# import os
# import re
# import time
# from collections import Counter
# import pandas as pd
# import joblib





# def detect_headers_and_footers(doc, line_threshold=0.4):
#     """
#     Detects repeating text that is likely a header or footer based on
#     frequency and position on the page. This is a critical first step.
#     """
#     page_count = len(doc)
#     if page_count < 4: return set()

#     text_counts = Counter()
#     # Scan the middle half of the document to avoid title and reference pages
#     start_page = page_count // 4
#     end_page = page_count - start_page

#     for page_num in range(start_page, end_page):
#         page = doc[page_num]
#         page_height = page.rect.height
#         blocks = page.get_text("blocks")
#         for b in blocks:
#             # Check a wider vertical area: top 20% and bottom 15% of the page
#             if b[1] < page_height * 0.20 or b[3] > page_height * 0.85:
#                 line_text = b[4].strip().replace('\n', ' ')
#                 if 5 < len(line_text) < 100 and not line_text.endswith('.'):
#                     text_counts[line_text] += 1
    
#     ignore_set = set()
#     # Lower the threshold to catch text that appears on 40% of scanned pages
#     min_occurrences = (end_page - start_page) * line_threshold
#     for text, count in text_counts.items():
#         if count >= min_occurrences:
#             ignore_set.add(text)
            
#     print(f"INFO: Detected {len(ignore_set)} repeating lines to ignore as headers/footers.")
#     return ignore_set

# def is_line_in_table(line_bbox, page_table_areas):
#     """Checks if a line's bounding box is inside any of a page's table areas."""
#     if not page_table_areas:
#         return False
    
#     l_x0, l_y0, l_x1, l_y1 = line_bbox
#     for t_bbox in page_table_areas:
#         t_x0, t_y0, t_x1, t_y1 = t_bbox
#         # Check for containment. A line is in a table if its bbox is inside the table's bbox.
#         if l_x0 >= t_x0 and l_y0 >= t_y0 and l_x1 <= t_x1 and l_y1 <= t_y1:
#             return True
#     return False

# def get_dominant_style(line):
#     """
#     Determines the most common (dominant) style in a line of text.
#     This is more robust than just checking the first span.
#     """
#     if not line["spans"]:
#         return (10, False) # Default style

#     style_counts = Counter()
#     for span in line["spans"]:
#         # More robust check for bold fonts
#         is_bold = bool(re.search(r'bold|black|heavy', span["font"], re.IGNORECASE))
#         style = (round(span["size"]), is_bold)
#         # We weigh the style by the length of the text in the span
#         style_counts[style] += len(span["text"].strip())
    
#     # Return the most common style
#     return style_counts.most_common(1)[0][0]

# def is_mostly_uppercase(s):
#     """
#     Checks if a string is predominantly uppercase. More robust than isupper().
#     """
#     letters = [char for char in s if char.isalpha()]
#     if not letters:
#         return False
#     uppercase_letters = [char for char in letters if char.isupper()]
#     return (len(uppercase_letters) / len(letters)) > 0.8

# def get_page_layout(page, threshold=0.3):
#     """
#     Analyzes the layout of a page to determine if it is single or multi-column.
#     Returns the number of detected columns (1 or 2).
#     """
#     page_width = page.rect.width
#     midpoint = page_width / 2
    
#     blocks = page.get_text("blocks")
#     if not blocks:
#         return 1 # Default to 1 column if no text

#     left_blocks = 0
#     right_blocks = 0
    
#     for b in blocks:
#         if b[2] < midpoint: # Block ends before midpoint
#             left_blocks += 1
#         elif b[0] > midpoint: # Block starts after midpoint
#             right_blocks += 1

#     total_sided_blocks = left_blocks + right_blocks
#     if total_sided_blocks == 0:
#         return 1

#     # Heuristic: If there are a significant number of blocks on both sides, it's a 2-column layout.
#     if (left_blocks > 0 and right_blocks > 0):
#         if (left_blocks / total_sided_blocks > threshold) and (right_blocks / total_sided_blocks > threshold):
#             return 2
    
#     return 1

# def process_pdf(pdf_path, ml_output_path=None):
#     """
#     Processes a PDF using a hybrid ML and rule-based filtering approach.
#     """
#     doc = fitz.open(pdf_path)
#     ignored_texts = detect_headers_and_footers(doc)
    
#     table_areas = {}
#     for page_num, page in enumerate(doc):
#         tables = page.find_tables()
#         if tables.tables:
#             table_areas[page_num] = [t.bbox for t in tables]
    
#     if table_areas:
#         print(f"INFO: Detected tables on pages: {list(table_areas.keys())}")
        
#     all_lines = []
#     style_counts = Counter()
#     page_heights = {}
    
#     for page_num, page in enumerate(doc):
#         page_width = page.rect.width
#         page_heights[page_num] = page.rect.height
#         page_table_bboxes = table_areas.get(page_num, [])
        
#         # --- NEW: Determine page layout for correct reading order ---
#         num_columns = get_page_layout(page)
#         page_midpoint = page_width / 2

#         blocks = page.get_text("dict")["blocks"]
#         for block in blocks:
#             if "lines" in block:
#                 for line in block["lines"]:
#                     if not line["spans"]: continue

#                     if is_line_in_table(line["bbox"], page_table_bboxes):
#                         continue

#                     line_text = "".join(span["text"] for span in line["spans"]).strip()
#                     if not line_text or line_text in ignored_texts: continue

#                     is_date = False
#                     text_lower = line_text.lower()
#                     if re.search(r'\b\d{4}\b', text_lower):
#                         if any(month in text_lower for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
#                             if len(line_text.split()) <= 4:
#                                 is_date = True
#                     if is_date:
#                         continue

#                     if re.match(r'^Page \d+\s*of\s*\d+$', line_text, re.I): 
#                         continue

#                     style = get_dominant_style(line)
#                     style_counts[style] += 1

#             # --- NEW: Add column index to each line ---
#                     line_center_x = (line["bbox"][0] + line["bbox"][2]) / 2
#                     column_index = 0
#                     if num_columns == 2 and line_center_x > page_midpoint:

#                         column_index = 1

#                     for idx, span in enumerate(line["spans"]):
#                         all_lines.append({
#                     "page": page_num,
#                     "page_number": page_num,
#                     "block_number": block.get("number", 0),
#                     "line_number": line.get("number", 0),
#                     "span_number": idx,  # index of span in line
#                     "text": span["text"],
#                     "font": span["font"],
#                     "size": span["size"],
#                     "color": span["color"],
#                     "is_bold": bool(re.search(r'bold|black|heavy', span["font"], re.IGNORECASE)),
#                     "is_italic": "italic" in span["font"].lower(),
#                     "x0": line["bbox"][0],
#                     "y0": line["bbox"][1],
#                     "x1": line["bbox"][2],
#                     "y1": line["bbox"][3],
#                     "column": column_index,
#                     "id": f"{page_num}-{line['bbox'][1]}-{idx}",
#                     "style": style,
#                 })


#     if not all_lines:
#         return {"title": "", "outline": []}
        
#     df = pd.DataFrame(all_lines)
#     if df.empty:
#         return {"title": "", "outline": []}
#     # --- NEW: Pass dataframe to pipeline for preprocessing ---
    
    

    

    
    
#     initial_candidates = []


#     page_0_height = page_heights.get(0, 1000)
#     page_0_lines_top_half = sorted(
#         [line for line in all_lines if line["page"] == 0 and line["y0"] < page_0_height / 2],
#         key=lambda x: (x["column"], x["y0"])
#     )

#     doc_title = ""
#     title_line_ids = set()
    
#     title_breaker_keywords = ["summary", "background", "introduction", "table of contents", "abstract", "keywords"]
#     author_affiliation_keywords = ["department", "university", "college", "institute", "@"]

#     if page_0_lines_top_half:
#         try:
#             max_size_page_0 = max(line["size"] for line in page_0_lines_top_half)
#             first_potential_title_lines = [line for line in page_0_lines_top_half if line["size"] == max_size_page_0]
#             if not first_potential_title_lines:
#                  raise ValueError("No lines found to be part of the title.")

#             first_title_line = first_potential_title_lines[0]
#             start_index = page_0_lines_top_half.index(first_title_line)
            
#             title_lines_text = []
#             last_line = None

#             for i in range(start_index, len(page_0_lines_top_half)):
#                 current_line = page_0_lines_top_half[i]
#                 current_text_lower = current_line["text"].lower()

#                 if last_line:
#                     if abs(current_line["y0"] - last_line["y0"]) > last_line["size"] * 2.5: break
#                     if current_line["size"] < first_title_line["size"] * 0.7: break
#                     if re.match(r"^(chapter|section|part|appendix|\d+(\.\d+)*\.?)\s", current_text_lower, re.I): break
#                     if any(keyword in current_text_lower for keyword in title_breaker_keywords): break
#                     if any(keyword in current_text_lower for keyword in author_affiliation_keywords): break
#                     # --- NEW: Stop title if it looks like a bullet point ---
#                     if re.match(r"^[•●*+-]\s*", current_line["text"]): break

#                 title_lines_text.append(current_line["text"])
#                 title_line_ids.add(current_line["id"])
#                 last_line = current_line
            
#             doc_title = " ".join(title_lines_text)

#         except (ValueError, IndexError):
#             doc_title = ""


#     if title_line_ids:
#         print(f"INFO: Identified title: '{doc_title}'. Excluding {len(title_line_ids)} lines from heading analysis.")
#         all_lines = [line for line in all_lines if line['id'] not in title_line_ids]

#     non_bold_styles = [s for s, c in style_counts.items() if not s[1]]
#     if not non_bold_styles:
#         body_style = style_counts.most_common(1)[0][0] if style_counts else (10, False)
#     else:
#         body_style = Counter({s: style_counts[s] for s in non_bold_styles}).most_common(1)[0][0]

#     df = pd.DataFrame(all_lines)
#     if df.empty:
#         return {"title": doc_title, "outline": []}
    


    
#     initial_candidates = []

#     all_candidates = {cand['id']: cand for cand in initial_candidates}

#     for line in all_lines:
#         if line['id'] not in all_candidates:
#             text = line['text']
#             is_stylistically_distinct = (line['size'] > body_style[0]) or (line['is_bold'] and not body_style[1])
#             is_short_enough = len(text.split()) < 25
#             is_numbered = re.match(r"^\d+(\.\d+)*\.?\s", text) or re.match(r"^Appendix [A-Z]:", text, re.I)

#             if is_short_enough and (is_stylistically_distinct or is_numbered):
#                 line['ml_level'] = 'H2'
#                 all_candidates[line['id']] = line

#     comprehensive_candidates = list(all_candidates.values())

#     candidate_texts = [c['text'] for c in comprehensive_candidates]
#     candidate_counts = Counter(candidate_texts)
#     repeating_texts_to_ignore = {text for text, count in candidate_counts.items() if count > 2}

#     if repeating_texts_to_ignore:
#         print(f"INFO: Pre-emptively ignoring potential headings that repeat frequently: {list(repeating_texts_to_ignore)}")

#     refined_headings = []
    
#     high_confidence_keywords = [
#         "abstract", "keywords", "introduction", "background", "related work",
#         "methodology", "methods", "materials and methods", "experimental setup",
#         "results", "discussion", "conclusion", "summary",
#         "acknowledgements", "references", "bibliography", "appendix", "author",
#         "table of contents", "list of figures", "list of tables"
#     ]

#     for cand in comprehensive_candidates:
#         text = cand["text"]
#         style = cand["style"]
        
#         is_centered = abs((cand["x0"] + cand["x1"]) / 2 - (page_width / 2)) < 20
#         is_numbered = re.match(r"^(\d+(\.\d+)*)\.?\s", text) or re.match(r"^Appendix [A-Z]:", text, re.I)
#         is_short = len(text.split()) <= 12
#         is_upper = is_mostly_uppercase(text)
#         if style[0] == body_style[0] and not style[1]:
#             continue 
#         if text in repeating_texts_to_ignore:
#             continue
            
#         if any(keyword == text.lower().strip().rstrip(':') for keyword in high_confidence_keywords):
#             cand["level"] = "H1"
#             refined_headings.append(cand)
#             continue

#         elif re.match(r"^(Appendix [A-Z]:|\d+(\.\d+)*\.?\s)", text):
#             # --- MODIFIED: More robust check for numbered headings using uppercase ratio ---
#             numeric_match_obj = re.match(r"^(\d+(\.\d+)*)\.?\s*(.*)", text)
#             if numeric_match_obj:
#                 text_part = numeric_match_obj.group(3)
#                 if is_mostly_uppercase(text_part) and len(text_part) > 3:
#                     pass
#                 elif style == body_style:
#                     continue
            
#             numeric_match = re.match(r"^(\d+(\.\d+)*)\.?\s", text)
#             if numeric_match:
#                 number_part = numeric_match.group(1)
#                 depth = number_part.count('.') + 1
#                 cand["level"] = f"H{min(depth, 6)}"
#             else:
#                 cand["level"] = "H1"
#             refined_headings.append(cand)
#             continue

#         elif style[0] == body_style[0] and style[1] and not body_style[1]:
#     # Only consider heading if text ends with ':' OR is centered
#             text = cand["text"].strip()
#             is_centered = abs((cand["x0"] + cand["x1"]) / 2 - (page_width / 2)) < 20
#             if (text.endswith(':') or is_centered) :
#                 cand["level"] = "H3"
#                 refined_headings.append(cand)

#             print(f"DEBUG: text='{text}' style={style} ml_level={cand.get('ml_level')} body_style={body_style}")


#         elif style[0] > body_style[0]:
#     # only add if bold or text is short (like heading)
#             if style[1] or len(text.split()) < 8:
#                 cand["level"] = cand.get('ml_level', 'H2')
#                 refined_headings.append(cand)
        
        
#         elif style[0] == body_style[0] and style[1] and not body_style[1]:
#     # only consider heading if ends with ':' or is centered
#             is_centered = abs((cand["x0"] + cand["x1"]) / 2 - (page_width / 2)) < 20
#             if text.endswith(':') or is_centered:
#                 cand["level"] = "H3"
#                 refined_headings.append(cand)
#         else:
#     # skip plain body-sized non-bold text
#             continue
        

#     # --- MODIFIED: Sort by page, then column, then vertical position ---
#     sorted_headings = sorted(refined_headings, key=lambda x: (x['page'], x['column'], x['y0']))
#     outline = []
#     i = 0
#     min_heading_size = body_style[0]
#     in_references_section = False

#     while i < len(sorted_headings):
#         current_heading = sorted_headings[i]
#         j = i + 1
#         while j < len(sorted_headings):
#             prev_line = sorted_headings[j-1]
#             next_line = sorted_headings[j]

#             if next_line["page"] == current_heading["page"] and \
#                next_line["column"] == current_heading["column"] and \
#                abs(next_line["y0"] - prev_line["y0"]) < prev_line["size"] * 1.8 and \
#                next_line['size'] >= current_heading['size'] * 0.85:
                
#                 if not re.match(r"^\d+(\.\d+)*\.?\s", next_line["text"]):
#                     current_heading["text"] += " " + next_line["text"]
#                     j += 1
#                 else:
#                     break 
#             else:
#                 break
        
#         text = current_heading["text"].strip()
        
#         is_rejected = False
#         text_lower = text.lower()

#         if any(keyword in text_lower for keyword in ["references", "bibliography"]):
#             in_references_section = True
        
#         if in_references_section and not any(keyword in text_lower for keyword in ["references", "bibliography"]):
#              if not re.match(r"^(Appendix [A-Z]|\d+(\.\d+)*)\s", text):
#                 is_rejected = True

#         if text.endswith(('.', ',', ';')) and len(text.split()) > 20: is_rejected = True
#         if len(text.split()) > 25: is_rejected = True
#         if re.match(r"^[•●*+-]\s*", text): is_rejected = True
#         if any(keyword in text_lower for keyword in author_affiliation_keywords): is_rejected = True
        
#         # --- NEW: More specific rejection rules for metadata and list-like items ---
#         if re.search(r'original research article|section:|doi:|inclusion criteria|exclusion criteria', text_lower):
#             is_rejected = True
            
#         # Reject likely author lists (contain numbers, commas, and are title-cased)
#         if bool(re.search(r'\d', text)) and bool(re.search(r',', text)) and text.istitle():
#             is_rejected = True

#         if text.isupper() and len(text.split()) < 5 and not current_heading.get('is_centered', 0):

#             cleaned_text_lower = text_lower.strip().rstrip(':')
#             if not any(keyword == cleaned_text_lower for keyword in high_confidence_keywords):
#                 is_rejected = True

#         if current_heading["size"] < min_heading_size and not current_heading['is_bold']: is_rejected = True
#         if re.fullmatch(r"[\d\W_]+", text) or len(text) < 4: is_rejected = True
        
#         if not is_rejected:
#             if 3 < len(text) < 250:
#                 outline.append({"level": current_heading["level"], "text": text, "page": current_heading["page"]})
        
#         i = j
        
#     if len(all_lines) < 30 and not outline and all_lines:
#         largest_line = max(all_lines, key=lambda x: x['size'])
#         outline.append({"level": "H1", "text": largest_line['text'], "page": largest_line['page']})

#     return {"title": doc_title, "outline": outline}



# def process_all_pdfs(input_dir, output_dir):
#     """
#     Processes all PDF files in a given directory.
#     """
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith(".pdf"):
#             pdf_path = os.path.join(input_dir, filename)
#             print(f"--- Processing {filename} ---")
#             start_time = time.time()
            
#             base_filename = os.path.splitext(filename)[0]
#             json_output_path = os.path.join(output_dir, base_filename + ".json")
            
            

#             output_data = process_pdf(pdf_path)
            
#             with open(json_output_path, 'w', encoding='utf-8') as f:
#                 json.dump(output_data, f, indent=4)
#             end_time = time.time()
#             print(f"--- Finished {filename} in {end_time - start_time:.2f} seconds. ---")

# if __name__ == "__main__":
#     INPUT_DIR = "../input"
#     OUTPUT_DIR = "../output"
#     ML_OUTPUT_DIR = "ml_output"
#     if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
#     if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
#     if not os.path.exists(ML_OUTPUT_DIR): os.makedirs(ML_OUTPUT_DIR)
#     process_all_pdfs(INPUT_DIR, OUTPUT_DIR, ML_OUTPUT_DIR)






import fitz  # PyMuPDF
import json
import os
import re
import time
from collections import Counter
import pandas as pd
import joblib

def detect_headers_and_footers(doc, line_threshold=0.4):
    """
    Detects repeating text that is likely a header or footer based on
    frequency and position on the page. This is a critical first step.
    """
    page_count = len(doc)
    if page_count < 4: return set()

    text_counts = Counter()
    # Scan the middle half of the document to avoid title and reference pages
    start_page = page_count // 4
    end_page = page_count - start_page

    for page_num in range(start_page, end_page):
        page = doc[page_num]
        page_height = page.rect.height
        blocks = page.get_text("blocks")
        for b in blocks:
            # Check a wider vertical area: top 20% and bottom 15% of the page
            if b[1] < page_height * 0.20 or b[3] > page_height * 0.85:
                line_text = b[4].strip().replace('\n', ' ')
                if 5 < len(line_text) < 100 and not line_text.endswith('.'):
                    text_counts[line_text] += 1
    
    ignore_set = set()
    # Lower the threshold to catch text that appears on 40% of scanned pages
    min_occurrences = (end_page - start_page) * line_threshold
    for text, count in text_counts.items():
        if count >= min_occurrences:
            ignore_set.add(text)
            
    print(f"INFO: Detected {len(ignore_set)} repeating lines to ignore as headers/footers.")
    return ignore_set

def is_line_in_table(line_bbox, page_table_areas):
    """Checks if a line's bounding box is inside any of a page's table areas."""
    if not page_table_areas:
        return False
    
    l_x0, l_y0, l_x1, l_y1 = line_bbox
    for t_bbox in page_table_areas:
        t_x0, t_y0, t_x1, t_y1 = t_bbox
        # Check for containment. A line is in a table if its bbox is inside the table's bbox.
        if l_x0 >= t_x0 and l_y0 >= t_y0 and l_x1 <= t_x1 and l_y1 <= t_y1:
            return True
    return False

def get_dominant_style(line):
    """
    Determines the most common (dominant) style in a line of text.
    This is more robust than just checking the first span.
    """
    if not line["spans"]:
        return (10, False) # Default style

    style_counts = Counter()
    for span in line["spans"]:
        # More robust check for bold fonts
        is_bold = bool(re.search(r'bold|black|heavy', span["font"], re.IGNORECASE))
        style = (round(span["size"]), is_bold)
        # We weigh the style by the length of the text in the span
        style_counts[style] += len(span["text"].strip())
    
    # Return the most common style
    return style_counts.most_common(1)[0][0]

def is_mostly_uppercase(s):
    """
    Checks if a string is predominantly uppercase. More robust than isupper().
    """
    letters = [char for char in s if char.isalpha()]
    if not letters:
        return False
    uppercase_letters = [char for char in letters if char.isupper()]
    return (len(uppercase_letters) / len(letters)) > 0.8

def get_page_layout(page, threshold=0.3):
    """
    Analyzes the layout of a page to determine if it is single or multi-column.
    Returns the number of detected columns (1 or 2).
    """
    page_width = page.rect.width
    midpoint = page_width / 2
    
    blocks = page.get_text("blocks")
    if not blocks:
        return 1 # Default to 1 column if no text

    left_blocks = 0
    right_blocks = 0
    
    for b in blocks:
        if b[2] < midpoint: # Block ends before midpoint
            left_blocks += 1
        elif b[0] > midpoint: # Block starts after midpoint
            right_blocks += 1

    total_sided_blocks = left_blocks + right_blocks
    if total_sided_blocks == 0:
        return 1

    # Heuristic: If there are a significant number of blocks on both sides, it's a 2-column layout.
    if (left_blocks > 0 and right_blocks > 0):
        if (left_blocks / total_sided_blocks > threshold) and (right_blocks / total_sided_blocks > threshold):
            return 2
    
    return 1

def process_pdf(pdf_path, ml_output_path=None):
    """
    Processes a PDF using a hybrid ML and rule-based filtering approach.
    """
    doc = fitz.open(pdf_path)
    ignored_texts = detect_headers_and_footers(doc)
    
    table_areas = {}
    for page_num, page in enumerate(doc):
        tables = page.find_tables()
        if tables.tables:
            table_areas[page_num] = [t.bbox for t in tables]
    
    if table_areas:
        print(f"INFO: Detected tables on pages: {list(table_areas.keys())}")
        
    all_lines = []
    style_counts = Counter()
    page_heights = {}
    
    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        page_heights[page_num] = page.rect.height
        page_table_bboxes = table_areas.get(page_num, [])
        
        num_columns = get_page_layout(page)
        page_midpoint = page_width / 2

        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if not line["spans"]: continue

                    if is_line_in_table(line["bbox"], page_table_bboxes):
                        continue

                    line_text = "".join(span["text"] for span in line["spans"]).strip()
                    if not line_text or line_text in ignored_texts: continue

                    is_date = False
                    text_lower = line_text.lower()
                    if re.search(r'\b\d{4}\b', text_lower):
                        if any(month in text_lower for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                            if len(line_text.split()) <= 4:
                                is_date = True
                    if is_date:
                        continue

                    if re.match(r'^Page \d+\s*of\s*\d+$', line_text, re.I): 
                        continue

                    style = get_dominant_style(line)
                    style_counts[style] += 1

                    line_center_x = (line["bbox"][0] + line["bbox"][2]) / 2
                    column_index = 0
                    if num_columns == 2 and line_center_x > page_midpoint:
                        column_index = 1

                    # Create a single entry for the entire line
                    all_lines.append({
                        "page": page_num,
                        "text": line_text,
                        "style": style,
                        "is_bold": style[1],
                        "size": style[0],
                        "x0": line["bbox"][0],
                        "y0": line["bbox"][1],
                        "x1": line["bbox"][2],
                        "y1": line["bbox"][3],
                        "column": column_index,
                        "id": f"{page_num}-{line['bbox'][1]}",
                    })

    if not all_lines:
        return {"title": "", "outline": []}
    
    # --- TITLE IDENTIFICATION (remains mostly the same) ---
    page_0_height = page_heights.get(0, 1000)
    page_0_lines_top_half = sorted(
        [line for line in all_lines if line["page"] == 0 and line["y0"] < page_0_height / 2],
        key=lambda x: (x["column"], x["y0"])
    )

    doc_title = ""
    title_line_ids = set()
    title_breaker_keywords = ["summary", "background", "introduction", "table of contents", "abstract", "keywords"]
    author_affiliation_keywords = ["department", "university", "college", "institute", "@"]

    # if page_0_lines_top_half:
    #     try:
    #         max_size_page_0 = max(line["size"] for line in page_0_lines_top_half)
    #         first_potential_title_lines = [line for line in page_0_lines_top_half if line["size"] == max_size_page_0]
    #         if not first_potential_title_lines:
    #             raise ValueError("No lines found to be part of the title.")

    #         first_title_line = first_potential_title_lines[0]
    #         start_index = page_0_lines_top_half.index(first_title_line)
            
    #         title_lines_text = []
    #         last_line = None

    #         for i in range(start_index, len(page_0_lines_top_half)):
    #             current_line = page_0_lines_top_half[i]
    #             current_text_lower = current_line["text"].lower()

    #             if last_line:
    #                 if abs(current_line["y0"] - last_line["y0"]) > last_line["size"] * 2.5: break
    #                 if current_line["size"] < first_title_line["size"] * 0.7: break
    #                 if re.match(r"^(chapter|section|part|appendix|\d+(\.\d+)*\.?)\s", current_text_lower, re.I): break
    #                 if any(keyword in current_text_lower for keyword in title_breaker_keywords): break
    #                 if any(keyword in current_text_lower for keyword in author_affiliation_keywords): break
    #                 if re.match(r"^[•●*+-]\s*", current_line["text"]): break

    #             title_lines_text.append(current_line["text"])
    #             title_line_ids.add(current_line["id"])
    #             last_line = current_line
            
    #         doc_title = " ".join(title_lines_text)

    #     except (ValueError, IndexError):
    #         doc_title = ""
    if page_0_lines_top_half:
        try:
            sizes = [line["size"] for line in page_0_lines_top_half]
            max_size_page_0 = max(sizes)
            min_size_page_0 = min(sizes)

            # Check if all fonts are basically same size (difference <1pt)
            if max_size_page_0 - min_size_page_0 < 1:
                # fallback: find first bold & centered line (centered = x0 + x1 ≈ center of page)
                page_width = doc[0].rect.width
                center_x = page_width / 2
                centered_bold_lines = [
                    line for line in page_0_lines_top_half
                    if line["is_bold"] and abs((line["x0"] + line["x1"]) / 2 - center_x) < page_width * 0.1
                ]
                if centered_bold_lines:
                    # pick first, limit to max 2 lines
                    title_lines_text = [centered_bold_lines[0]["text"]]
                    title_line_ids.add(centered_bold_lines[0]["id"])
                    if len(centered_bold_lines) > 1:
                        title_lines_text.append(centered_bold_lines[1]["text"])
                        title_line_ids.add(centered_bold_lines[1]["id"])
                    doc_title = " ".join(title_lines_text)
                else:
                    doc_title = ""
            else:
                # normal path: use biggest font lines
                first_potential_title_lines = [line for line in page_0_lines_top_half if line["size"] == max_size_page_0]
                if not first_potential_title_lines:
                    raise ValueError("No lines found to be part of the title.")

                first_title_line = first_potential_title_lines[0]
                start_index = page_0_lines_top_half.index(first_title_line)

                title_lines_text = []
                last_line = None

                for i in range(start_index, len(page_0_lines_top_half)):
                    current_line = page_0_lines_top_half[i]
                    current_text_lower = current_line["text"].lower()

                    # break if too long title (more than 2 lines)
                    if len(title_lines_text) >= 2:
                        break
                    if last_line:
                        if abs(current_line["y0"] - last_line["y0"]) > last_line["size"] * 2.5: break
                        if current_line["size"] < first_title_line["size"] * 0.7: break
                        if re.match(r"^(chapter|section|part|appendix|\d+(\.\d+)*\.?)\s", current_text_lower, re.I): break
                        if any(keyword in current_text_lower for keyword in title_breaker_keywords): break
                        if any(keyword in current_text_lower for keyword in author_affiliation_keywords): break
                        if re.match(r"^[•●*+-]\s*", current_line["text"]): break

                    title_lines_text.append(current_line["text"])
                    title_line_ids.add(current_line["id"])
                    last_line = current_line

                doc_title = " ".join(title_lines_text)

        except (ValueError, IndexError):
            doc_title = ""


    if title_line_ids:
        print(f"INFO: Identified title: '{doc_title}'. Excluding {len(title_line_ids)} lines from heading analysis.")
        all_lines = [line for line in all_lines if line['id'] not in title_line_ids]

    # --- HEADING IDENTIFICATION (NEW LOGIC) ---

    # 1. Determine the main body style of the document
    non_bold_styles = [s for s, c in style_counts.items() if not s[1]]
    if not non_bold_styles:
        body_style = style_counts.most_common(1)[0][0] if style_counts else (10, False)
    else:
        body_style = Counter({s: style_counts[s] for s in non_bold_styles}).most_common(1)[0][0]
    print(f"INFO: Deduced body text style: {body_style} (size, is_bold)")

    # 2. **NEW**: Check if page 0 contains any paragraph text
    page_0_has_paragraphs = any(line['page'] == 0 and line['style'] == body_style for line in all_lines)
    if not page_0_has_paragraphs:
        print("INFO: Page 0 has no paragraph text. It will be ignored for heading extraction.")

    # 3. Filter for initial heading candidates based on style and simple heuristics
    initial_candidates = []
    for line in all_lines:
        # **NEW**: Skip page 0 if it's determined to be a cover page
        if not page_0_has_paragraphs and line['page'] == 0:
            continue
            
        # A heading must be stylistically distinct from the body text
        is_stylistically_distinct = (line['size'] > body_style[0]) or (line['is_bold'] and not body_style[1])
        if not is_stylistically_distinct:
            continue

        # Basic text filters
        text = line['text']
        if not (3 < len(text) < 250): continue
        if len(text.split()) > 25: continue # Exclude long lines
        if re.fullmatch(r"[\d\W_]+", text): continue # Exclude lines with only numbers/symbols
        if text.endswith(('.', ',', ';')) and len(text.split()) > 15: continue
        
        initial_candidates.append(line)

    # 4. **NEW**: Dynamically determine heading levels based on sorted styles
    if not initial_candidates:
        return {"title": doc_title, "outline": []}

    # Get all unique styles from our candidates
    heading_styles = sorted(
        list(set(c['style'] for c in initial_candidates)),
        key=lambda s: (-s[0], -s[1])  # Sort by size (desc), then by bold status (True first)
    )

    # Create a map from a style to its hierarchical level (H1, H2, etc.)
    style_to_level_map = {style: f"H{i+1}" for i, style in enumerate(heading_styles)}
    
    print("INFO: Detected heading style hierarchy:")
    for style, level in style_to_level_map.items():
        print(f"  - {level}: {style}")

    # 5. Assign levels to all candidates
    refined_headings = []
    for cand in initial_candidates:
        cand['level'] = style_to_level_map.get(cand['style'])
        if cand['level']:
             refined_headings.append(cand)
             
    # --- POST-PROCESSING (Merging and Final Filtering) ---
    sorted_headings = sorted(refined_headings, key=lambda x: (x['page'], x['column'], x['y0']))
    
    outline = []
    i = 0
    in_references_section = False
    
    while i < len(sorted_headings):
        current_heading = sorted_headings[i]
        j = i + 1
        # Merge consecutive lines that are part of the same heading
        while j < len(sorted_headings):
            prev_line = sorted_headings[j-1]
            next_line = sorted_headings[j]

            # Merge if lines are close, on the same page/column, and have the same style
            if (next_line["page"] == current_heading["page"] and
                next_line["column"] == current_heading["column"] and
                next_line["style"] == current_heading["style"] and
                abs(next_line["y0"] - prev_line["y1"]) < current_heading["size"] * 0.5):
                
                # Don't merge if the next line looks like a new numbered item
                if not re.match(r"^\d+(\.\d+)*\.?\s", next_line["text"]):
                    current_heading["text"] += " " + next_line["text"]
                    current_heading["y1"] = next_line["y1"] # Update bbox
                    j += 1
                else:
                    break
            else:
                break
        
        text = current_heading["text"].strip()
        text_lower = text.lower()
        is_rejected = False

        if any(keyword in text_lower for keyword in ["references", "bibliography"]):
            in_references_section = True
        
        # In reference section, only allow Appendix or new numbered sections
        if in_references_section and not any(keyword in text_lower for keyword in ["references", "bibliography"]):
            if not re.match(r"^(Appendix [A-Z]|\d+(\.\d+)*)\s", text):
                is_rejected = True

        if re.search(r'original research article|section:|doi:|inclusion criteria|exclusion criteria', text_lower):
            is_rejected = True

        if not is_rejected:
            outline.append({"level": current_heading["level"], "text": text, "page": current_heading["page"]})
        
        i = j

    return {"title": doc_title, "outline": outline}


def process_all_pdfs(input_dir, output_dir):
    """
    Processes all PDF files in a given directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"--- Processing {filename} ---")
            start_time = time.time()
            
            base_filename = os.path.splitext(filename)[0]
            json_output_path = os.path.join(output_dir, base_filename + ".json")
            
            output_data = process_pdf(pdf_path)
            
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            end_time = time.time()
            print(f"--- Finished {filename} in {end_time - start_time:.2f} seconds. ---")


# Example usage:
# process_all_pdfs('path/to/your/pdfs', 'path/to/your/output')