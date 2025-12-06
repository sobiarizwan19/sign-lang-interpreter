import google.generativeai as genai
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

GEMINI_API_KEY = "AIzaSyCv2XlAHLKQBCp6TzGk1GDiGLJ-EJ0mJ_g"
GEMINI_MODEL = "gemini-2.5-flash"


class GeminiInterpreter:
    def __init__(self):
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        else:
            self.gemini_model = None
    
    def extract_interpretation(self, response_text):
        """Extract interpretation from Gemini response by looking for INTERPRETATION: pattern"""
        patterns = [
            r'INTERPRETATION:\s*(.+?)(?:\n|$)',
            r'Interpretation:\s*(.+?)(?:\n|$)',
            r'FINAL INTERPRETATION:\s*(.+?)(?:\n|$)',
            r'Final Interpretation:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, return the entire response
        return response_text.strip()
    
    def ask_gemini(self, filtered_format):
        if self.gemini_model is None:
            return "Gemini API key not configured"
        
        prompt = f"""
        I have ASL (American Sign Language) detection results in format (letter, count).
        Each (letter, count) pair represents a single letter held for "count" consecutive frames.
        The sequence of pairs represents the order of letters in the ASL message.
        When no hand is detected, it is represented as "SPACE", indicating separation between words.

        IMPORTANT NOTE ABOUT DATA QUALITY:
        The detection data comes from a computer vision model and may contain ERRORS.
        Misdetections may occur due to:
        1. Model confusion between visually similar signs
        2. False positives (detecting a letter instead of SPACE)
        3. False negatives (missing letters)
        4. Hand movements causing temporary misclassification
        5. Lighting, angle, or occlusion issues

        Your task:
        ➡ You MUST return your final interpretation in this exact format:
        INTERPRETATION: [your interpretation here]

        ➡ The interpretation must be a valid and meaningful English word, phrase, or sentence.
        ➡ It must be something a real person would logically say.
        ➡ Prefer interpretations using ONLY the detected letters.
        ➡ "SPACE" in the data indicates word separation.
        ➡ Do NOT invent unnecessary extra letters.

        IMPORTANT RULES:
        1. Use primarily the letters that appear in the filtered data.
        2. If a letter does NOT appear at all, do NOT assume it unless absolutely necessary.
        3. Respect the sequence and grouping of letters shown.
        4. Account for possible misdetections when interpreting.
        5. "SPACE" represents a word boundary.

        PRIORITIZING REPEATED LETTERS:
        - Letters with larger counts or multiple repeated detections are HIGH CONFIDENCE.
        - Interpretation MUST prioritize words formed using strongly repeated letters.
        - A confusion-based replacement is allowed ONLY IF:
            ✔ The detected letter is low count, weak, or isolated
            ✔ The replacement results in a more logical real-world interpretation
            ✔ The change does NOT modify or contradict strongly repeated letters
        - DO NOT replace a repeated or high-count letter merely to form a different word.
        Example:
            [('C', 15), ('O', 22), ('F', 23), ('E', 27)]
            → "coffee" is correct because F and E repeat strongly.
            → NOT "code", which wrongly replaces a reliable F with D.

        CRITICAL - NO ABBREVIATIONS:
        - The input contains only A-Z and SPACE — NO abbreviations.
        - Your interpretation MUST NOT contain abbreviations or acronyms.
        - Do NOT output things like: IDK, LOL, BRB, ASAP, FYI, etc.
        - Output only complete properly spelled English words or sentences.

        ALPHABET CONFUSIONS (USE ONLY WHEN NECESSARY):
        - Common ASL confusions include:
            M ↔ N ↔ T
            A ↔ S ↔ Y ↔ E
            O ↔ C
            D ↔ F
        - Space may also be confused with a letter.
        - Use these ONLY to clarify a meaningful interpretation, not to create random words.

        Handling Misinterpretations:
        - If the resulting sequence looks odd or incomplete:
            * Consider similar-letter confusion ONLY when logical
            * Remove accidental SPACE only if clearly noise
            * Correct repeated fragments caused by movement
        - Always choose the most plausible real-world meaningful interpretation.

        FILTERED DATA: {filtered_format}

        Your output MUST:
        - Start with: INTERPRETATION:
        - Contain ONLY the interpretation (no notes, NO explanation)
        - Use complete words (NO abbreviations)

        Remember: The data may contain errors — find the MOST logical and meaningful interpretation.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract interpretation using pattern matching
            interpretation = self.extract_interpretation(response_text)
            return interpretation
            
        except Exception as e:
            return f"Error: {e}"