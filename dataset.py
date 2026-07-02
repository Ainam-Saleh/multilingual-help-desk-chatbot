import json
import os


# ============================================================
# DATASET CONVERTER
# This script reads the raw FAQ data from a plain text file (data.txt)
# and converts it into a structured JSON file (nile_dataset.json)
# that the chatbot can load and search through.
#
# WHY THIS IS NEEDED:
# The raw data collected from Nile University's website comes as
# unstructured text. The chatbot needs it in a clean
# {"question": "...", "answer": "..."} format to work properly.
#
# HOW TO RUN:
# Simply run: python dataset.py
# It will read data.txt and produce nile_dataset.json automatically.
# ============================================================


def clean_and_convert(input_file="data.txt", output_file="nile_dataset.json"):
    # --------------------------------------------------------
    # CHECK IF INPUT FILE EXISTS
    # If data.txt is missing, stop and show an error message
    # instead of crashing with a confusing Python error
    # --------------------------------------------------------
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # --------------------------------------------------------
    # READ ALL LINES FROM THE TEXT FILE
    # Opens data.txt and loads every line into a list.
    # utf-8 encoding is specified to handle special characters
    # in Nigerian language content (e.g. Yoruba tonal marks)
    # --------------------------------------------------------
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # --------------------------------------------------------
    # INITIALIZE TRACKING VARIABLES
    # qa_pairs        → the final list of Q&A pairs we are building
    # current_q       → holds the current question being processed
    # current_a_lines → collects answer lines for the current question
    #                   (answers can span multiple lines, so we collect
    #                    them and join them together at the end)
    # --------------------------------------------------------
    qa_pairs = []
    current_q = None
    current_a_lines = []

    # --------------------------------------------------------
    # SECTION HEADERS TO IGNORE
    # The raw text file contains section headings like
    # "Undergraduate", "Hostel/Accommodation", etc.
    # These are NOT questions or answers — they are just
    # category labels used to organize the FAQ content.
    # We store them in a set so we can skip them during processing.
    # --------------------------------------------------------
    ignore_headers = {
        "Undergraduate",
        "Postgraduate",
        "School Of Preliminary Studies (SPS)",
        "Applications",
        "Online Payment",
        "Scholarships",
        "Online Programs",
        "Transfers",
        "Students Portal",
        "Hostel/Accommodation",
        "Research And Innovation",
        "Others"
    }

    # --------------------------------------------------------
    # MAIN PROCESSING LOOP
    # Goes through every line in the text file one by one
    # and decides whether it is:
    # (a) A section header  → skip it
    # (b) A question        → save previous Q&A, start a new one
    # (c) An answer line    → append to the current answer
    # --------------------------------------------------------
    for line in lines:
        # Remove leading/trailing whitespace from each line
        line = line.strip()

        # Skip completely empty lines
        if not line:
            continue

        # --- HANDLE SECTION HEADERS ---
        # If the line matches one of our known headers,
        # save any pending Q&A pair we were building,
        # then reset so we start fresh for the next section.
        # This prevents section titles from accidentally being
        # included inside an answer from the previous section.
        if line in ignore_headers:
            if current_q and current_a_lines:
                full_answer = " ".join(current_a_lines).strip()
                qa_pairs.append({"question": current_q, "answer": full_answer})

            # Reset both trackers for the new section
            current_q = None
            current_a_lines = []
            continue

        # --- IDENTIFY WHETHER LINE IS A QUESTION OR ANSWER ---
        # A line is treated as a QUESTION if it meets either condition:
        # 1. It ends with a "?" character, OR
        # 2. It is written in ALL CAPS (common in the raw FAQ data)
        #    AND is longer than 5 characters (to avoid short labels)
        # URLs are excluded because they sometimes appear in caps
        # but are always part of an answer, never a question.
        is_question = (
            line.endswith('?') or
            (line.isupper() and len(line) > 5)
        ) and "http" not in line

        if is_question:
            # --- NEW QUESTION DETECTED ---
            # Before starting the new question, save the previous
            # Q&A pair if we have one waiting to be saved.
            if current_q and current_a_lines:
                full_answer = " ".join(current_a_lines).strip()
                qa_pairs.append({"question": current_q, "answer": full_answer})

            # Start tracking the new question
            current_q = line
            current_a_lines = []  # Reset answer collector for new question

        else:
            # --- ANSWER LINE DETECTED ---
            # Add this line to the answer for the current question.
            # We only do this if we already have an active question —
            # otherwise we'd be collecting orphaned lines with no question.
            if current_q:
                current_a_lines.append(line)

    # --------------------------------------------------------
    # SAVE THE VERY LAST Q&A PAIR
    # The loop above saves a pair only when it detects a NEW question.
    # This means the final question in the file never gets saved
    # inside the loop. We save it here manually after the loop ends.
    # --------------------------------------------------------
    if current_q and current_a_lines:
        qa_pairs.append({"question": current_q, "answer": " ".join(current_a_lines).strip()})

    # --------------------------------------------------------
    # WRITE OUTPUT TO JSON FILE
    # Saves all collected Q&A pairs to nile_dataset.json
    # indent=4 makes the JSON file human-readable with proper formatting
    # --------------------------------------------------------
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4)

    print(f"Success! Converted {len(qa_pairs)} Q&A pairs to {output_file}")


# ============================================================
# SCRIPT ENTRY POINT
# This block runs only when you execute the file directly
# (e.g. python dataset.py).
# It won't run if this file is imported by another script.
# ============================================================
if __name__ == "__main__":
    clean_and_convert()
