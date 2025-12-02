# Try to import, with helpful error message
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError as e:
    GENAI_AVAILABLE = False
    print("⚠️ Warning: google-generativeai not installed!")
    print("Install it with: pip install google-generativeai")

from typing import Optional

class SentencePredictor:
    """
    Predicts sentences from ASL alphabet sequences using Google Gemini
    """
    
    def __init__(self, api_key: str = "AIzaSyCtjah_oDE87A0hSs8Mwg89RzkHXLP69no"):
        """
        Initialize Gemini API
        
        Args:
            api_key: Google Gemini API key
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai package is not installed. Install with: pip install google-generativeai")
        
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        
        # Try different model names for compatibility
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-flash-latest',
            'models/gemini-1.5-flash',
            'gemini-pro'
        ]
        
        self.model = None
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                # Test the model with a simple prompt
                test_response = self.model.generate_content("Say 'OK'")
                print(f"✓ Gemini API initialized successfully (using {model_name})!")
                break
            except Exception as e:
                print(f"⚠️ Model {model_name} not available: {e}")
                continue
        
        if self.model is None:
            raise Exception("Could not initialize any Gemini model. Please check your API key and internet connection.")
    
    def predict_sentence(self, alphabet_sequence: str) -> dict:
        """
        Predict sentence/word from ASL alphabet sequence
        
        Args:
            alphabet_sequence: String of space-separated letters (e.g., "H E L L O")
            
        Returns:
            dict with keys:
                - 'interpretation': Main interpretation
                - 'alternatives': List of alternative interpretations
                - 'confidence': Confidence level (HIGH/MEDIUM/LOW)
                - 'raw_response': Full Gemini response
        """
        
        if not alphabet_sequence or alphabet_sequence.strip() == "":
            return {
                'interpretation': "No sequence provided",
                'alternatives': [],
                'confidence': "LOW",
                'raw_response': "",
                'reasoning': "Empty sequence",
                'original_sequence': ""
            }
        
        # Create the prompt
        prompt = self._create_prompt(alphabet_sequence)
        
        try:
            # Call Gemini API with safety settings
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                )
            )
            
            # Parse response
            result = self._parse_response(response.text, alphabet_sequence)
            
            return result
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return {
                'interpretation': f"API Error: {str(e)}",
                'alternatives': [],
                'confidence': "LOW",
                'raw_response': "",
                'reasoning': f"Error: {str(e)}",
                'original_sequence': alphabet_sequence
            }
    
    def _create_prompt(self, alphabet_sequence: str) -> str:
        """
        Create a detailed prompt for Gemini
        """
        prompt = f"""You are an expert in American Sign Language (ASL) interpretation. You have been given a sequence of individual letters that were detected from ASL fingerspelling in a video.

**Detected ASL Letter Sequence:**
{alphabet_sequence}

**Your Task:**
Interpret what word or sentence is being spelled. Consider the following:

1. **Letter Spacing**: The letters are space-separated and represent individual signs detected at different points in the video
2. **Potential Errors**: The detection system may have:
   - Missed some letters
   - Detected duplicate letters when the hand was held still
   - Misidentified similar-looking signs (e.g., M/N, A/S)
3. **Context**: Consider common English words and phrases
4. **Multiple Interpretations**: If the sequence is ambiguous, provide alternative interpretations

**Response Format:**
Provide your response in the following structure:

PRIMARY INTERPRETATION: [Your best guess at the intended word/sentence]

CONFIDENCE: [HIGH/MEDIUM/LOW]

REASONING: [Brief explanation of why you chose this interpretation]

ALTERNATIVES: [List 2-3 alternative interpretations if applicable, or write "None"]

**Example Response:**

PRIMARY INTERPRETATION: HELLO

CONFIDENCE: HIGH

REASONING: The sequence "H E L L O" clearly spells a common greeting with no ambiguity.

ALTERNATIVES: None

---

**Now interpret the sequence above:**"""

        return prompt
    
    def _parse_response(self, response_text: str, original_sequence: str) -> dict:
        """
        Parse Gemini's response into structured format
        """
        lines = response_text.strip().split('\n')
        
        result = {
            'interpretation': '',
            'alternatives': [],
            'confidence': 'MEDIUM',
            'raw_response': response_text,
            'reasoning': '',
            'original_sequence': original_sequence
        }
        
        # Parse each line
        for i, line in enumerate(lines):
            line = line.strip()
            
            if 'PRIMARY INTERPRETATION:' in line.upper():
                result['interpretation'] = line.split(':', 1)[1].strip() if ':' in line else line
            
            elif 'CONFIDENCE:' in line.upper():
                confidence_text = line.split(':', 1)[1].strip() if ':' in line else ''
                confidence = confidence_text.upper()
                if 'HIGH' in confidence:
                    result['confidence'] = 'HIGH'
                elif 'LOW' in confidence:
                    result['confidence'] = 'LOW'
                else:
                    result['confidence'] = 'MEDIUM'
            
            elif 'REASONING:' in line.upper():
                result['reasoning'] = line.split(':', 1)[1].strip() if ':' in line else line
            
            elif 'ALTERNATIVES:' in line.upper():
                alt_text = line.split(':', 1)[1].strip() if ':' in line else ''
                if alt_text.lower() not in ['none', 'n/a', '']:
                    # Try to parse alternatives
                    if i + 1 < len(lines):
                        # Check next few lines for alternative items
                        for j in range(i + 1, min(i + 5, len(lines))):
                            alt_line = lines[j].strip()
                            if alt_line and (alt_line.startswith('-') or alt_line.startswith('•') or alt_line.startswith('*') or (alt_line[0].isdigit() and '.' in alt_line[:3])):
                                clean_alt = alt_line.lstrip('-•*0123456789. ').strip()
                                if clean_alt:
                                    result['alternatives'].append(clean_alt)
        
        # Fallback: if no interpretation found, use first meaningful line
        if not result['interpretation']:
            for line in lines:
                clean_line = line.strip()
                if clean_line and len(clean_line) > 2 and not clean_line.startswith('**') and not clean_line.startswith('---'):
                    result['interpretation'] = clean_line
                    break
        
        # If still no interpretation, extract from raw response
        if not result['interpretation']:
            # Try to find any capitalized word that might be the answer
            import re
            words = re.findall(r'\b[A-Z]{2,}\b', response_text)
            if words:
                result['interpretation'] = words[0]
        
        return result
    
    def format_result(self, result: dict) -> str:
        """
        Format the result for display
        """
        output = f"""
╔══════════════════════════════════════════════════════════╗
║               ASL SEQUENCE INTERPRETATION                ║
╚══════════════════════════════════════════════════════════╝

Original Sequence: {result.get('original_sequence', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 PRIMARY INTERPRETATION:
   {result.get('interpretation', 'N/A')}

📊 CONFIDENCE: {result.get('confidence', 'N/A')}

💡 REASONING:
   {result.get('reasoning', 'N/A')}
"""
        
        if result.get('alternatives'):
            output += "\n🔄 ALTERNATIVE INTERPRETATIONS:\n"
            for i, alt in enumerate(result['alternatives'], 1):
                output += f"   {i}. {alt}\n"
        
        output += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        return output


def main():
    """
    Test the sentence predictor
    """
    if not GENAI_AVAILABLE:
        print("❌ Cannot run test - google-generativeai not installed")
        print("Install with: pip install google-generativeai")
        return
    
    print("="*60)
    print("ASL Sentence Predictor - Test Mode")
    print("="*60)
    
    try:
        predictor = SentencePredictor()
        
        # Test sequences
        test_sequences = [
            "H E L L O",
            "H E L P",
            "T H A N K Y O U",
        ]
        
        for sequence in test_sequences:
            print(f"\n{'='*60}")
            print(f"Testing sequence: {sequence}")
            print('='*60)
            
            result = predictor.predict_sentence(sequence)
            formatted = predictor.format_result(result)
            print(formatted)
            
            user_input = input("Press Enter to continue (or 'q' to quit)...")
            if user_input.lower() == 'q':
                break
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()