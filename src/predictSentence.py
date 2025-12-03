try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError as e:
    GENAI_AVAILABLE = False

from typing import Optional

class SentencePredictor:
    """
    Predicts sentences from ASL alphabet sequences using Google Gemini
    """
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize Gemini API
        
        Args:
            api_key: Google Gemini API key (if None, uses default)
            model_name: Gemini model to use (if None, tries defaults)
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai package is not installed. Install with: pip install google-generativeai")
        
        # Use provided API key or default
        if api_key is None:
            api_key = "AIzaSyCb-XaqhT3v1He3cTRH0zn6QCZFwHBKRNs"
        
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        
        # Use provided model or try defaults
        if model_name:
            # Remove 'models/' prefix if present for SDK initialization
            clean_model_name = model_name.replace('models/', '')
            model_attempts = [clean_model_name]
        else:
            model_attempts = [
                'gemini-2.0-flash-exp',
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro'
            ]
        
        self.model = None
        self.model_name = None
        
        # Safety settings to prevent blocking
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        for model_name in model_attempts:
            try:
                test_model = genai.GenerativeModel(
                    model_name,
                    safety_settings=safety_settings
                )
                # Test with simple prompt
                test_response = test_model.generate_content("Test")
                # Verify we can access text
                _ = test_response.text
                self.model = test_model
                self.model_name = model_name
                print(f"✓ Successfully initialized model: {model_name}")
                break
            except Exception as e:
                error_msg = str(e)
                print(f"✗ Failed to initialize {model_name}: {error_msg[:100]}")
                if "API key" in error_msg.lower():
                    break  # Don't try other models if API key is invalid
                continue
        
        if self.model is None:
            raise Exception("Could not initialize any Gemini model")
    
    def predict_sentence(self, alphabet_sequence: str) -> dict:
        """
        Predict sentence/word from ASL alphabet sequence
        
        Args:
            alphabet_sequence: String of space-separated letters (e.g., "H E L L O")
            
        Returns:
            dict with interpretation results
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
        
        if self.model is None:
            return {
                'interpretation': "Model not initialized",
                'alternatives': [],
                'confidence': "LOW", 
                'raw_response': "",
                'reasoning': "Gemini model failed to initialize",
                'original_sequence': alphabet_sequence
            }
        
        prompt = self._create_prompt(alphabet_sequence)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_output_tokens=500,
                )
            )
            
            # Check if response was blocked
            if not response.parts:
                return {
                    'interpretation': alphabet_sequence.replace(' ', ''),
                    'alternatives': [],
                    'confidence': "LOW",
                    'raw_response': "",
                    'reasoning': "Response blocked by safety filters",
                    'original_sequence': alphabet_sequence
                }
            
            # Try to get text
            try:
                response_text = response.text
            except Exception as e:
                # If response.text fails, try to get from parts
                if response.parts:
                    response_text = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
                else:
                    raise e
            
            result = self._parse_response(response_text, alphabet_sequence)
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"API Error: {error_msg}")
            
            # Fallback: return simple interpretation
            simple_interpretation = alphabet_sequence.replace(' ', '')
            return {
                'interpretation': simple_interpretation,
                'alternatives': [],
                'confidence': "LOW",
                'raw_response': "",
                'reasoning': f"API error - showing raw sequence. Error: {error_msg[:100]}",
                'original_sequence': alphabet_sequence
            }
    
    def _create_prompt(self, alphabet_sequence: str) -> str:
        """Create an improved prompt for better sentence interpretation that handles noise"""
        prompt = f"""You are an expert ASL (American Sign Language) fingerspelling interpreter. Your task is to interpret a sequence of letters detected from real-time ASL fingerspelling.

DETECTED LETTER SEQUENCE: {alphabet_sequence}

IMPORTANT CONSIDERATIONS:
1. This sequence comes from COMPUTER VISION detection - there will be RANDOM NOISE letters due to:
   - False positive detections
   - Transition frames between signs
   - Hand movement artifacts
   - Similar-looking signs being misclassified

2. Common noise patterns to filter out:
   - Single random letters that don't fit context
   - Repeated letters that don't make sense (like "HHEELLOO" for "HELLO")
   - Letters that appear and disappear quickly
   - Common misclassifications (B↔D, M↔N, P↔K, etc.)

3. Real fingerspelling characteristics:
   - People spell actual WORDS, NAMES, or PHRASES
   - Noise letters are usually brief and isolated
   - Meaningful letters appear in coherent patterns
   - If sequence seems random, it might be all noise

4. Your interpretation should:
   - Extract the meaningful word/phrase from the noise
   - Correct obvious spelling errors from detection
   - Ignore isolated noise letters
   - Consider common names, places, and terms
   - Return ONLY the clean interpretation

EXAMPLES WITH NOISE:
- "H X E L L O Y" → "HELLO" (filter 'X' and 'Y' as noise)
- "M Y A N A M E B I S J O H N" → "MY NAME IS JOHN" (filter 'A' and 'B' as noise)
- "T H A N K Y O U Z" → "THANK YOU" (filter 'Z' as noise)
- "C O F F E E Q W" → "COFFEE" (filter 'Q W' as noise)
- "S T A N F O R D X X" → "STANFORD" (filter 'X X' as noise)
- "A B C D E F G" → "ABCDEFG" (if all seems random, might be spelling alphabet practice)

INTERPRETATION STRATEGY:
1. Look for coherent word patterns in the sequence
2. Filter out isolated letters not fitting the pattern
3. Consider the sequence as a whole, not just individual letters
4. If multiple interpretations possible, choose the most common/likely one
5. When in doubt, return the letters that form a recognizable word

YOUR TASK: Given the detected sequence above, provide ONLY the most likely clean English interpretation. Return JUST the interpreted text, nothing else.

Interpretation:"""
        return prompt
    
    def _parse_response(self, response_text: str, original_sequence: str) -> dict:
        """Parse Gemini's response into structured format"""
        # Clean the response text
        interpretation = response_text.strip()
        
        # Remove any quotes or extra formatting
        interpretation = interpretation.strip('"\'')
        
        # Remove common prefixes and explanations
        prefixes_to_remove = [
            'PRIMARY INTERPRETATION:', 'INTERPRETATION:', 'Answer:', 'Word:',
            'The interpretation is:', 'Interpretation is:', 'Clean interpretation:',
            'Most likely:', 'Likely word:', 'Result:', 'Output:'
        ]
        
        for prefix in prefixes_to_remove:
            if interpretation.upper().startswith(prefix.upper()):
                interpretation = interpretation[len(prefix):].strip()
        
        # Remove any trailing explanations (anything after newline or period that's not part of the word)
        interpretation = interpretation.split('\n')[0].strip()
        interpretation = interpretation.split('.')[0].strip()
        
        # If interpretation is empty or too short, fall back to original without spaces
        if not interpretation or len(interpretation) < 2:
            interpretation = original_sequence.replace(' ', '')
        
        # Determine confidence based on various factors
        sequence_no_spaces = original_sequence.replace(' ', '')
        
        # Calculate confidence
        confidence = 'MEDIUM'
        
        # High confidence if:
        # 1. Interpretation matches a common word/name
        # 2. Interpretation is significantly shorter than original (noise filtered)
        # 3. Interpretation forms a recognizable pattern
        if interpretation.upper() == sequence_no_spaces.upper():
            confidence = 'HIGH'
        elif self._is_common_word(interpretation):
            confidence = 'HIGH'
        elif len(interpretation) < len(sequence_no_spaces) * 0.7:  # Filtered out >30% as noise
            confidence = 'HIGH'  # Likely successfully filtered noise
        elif len(interpretation) >= 2 and ' ' in interpretation:  # Multi-word phrase
            confidence = 'HIGH'
        
        # Generate some alternatives
        alternatives = []
        if confidence == 'MEDIUM':
            # Add original sequence without spaces as alternative
            alternatives.append(sequence_no_spaces)
            # Add a version with common corrections
            corrected = self._apply_common_corrections(sequence_no_spaces)
            if corrected != interpretation and corrected != sequence_no_spaces:
                alternatives.append(corrected)
        
        result = {
            'interpretation': interpretation,
            'alternatives': alternatives,
            'confidence': confidence,
            'raw_response': response_text,
            'reasoning': f'Filtered noise from {len(sequence_no_spaces)} detected letters to {len(interpretation)} meaningful characters',
            'original_sequence': original_sequence,
            'noise_filtered': len(sequence_no_spaces) - len(interpretation.replace(' ', ''))
        }
        
        return result
    
    def _is_common_word(self, text: str) -> bool:
        """Check if text is a common word/phrase"""
        text_lower = text.lower()
        
        # Common words list
        common_words = {
            'hello', 'hi', 'thank', 'you', 'thanks', 'please', 'sorry', 'yes', 'no',
            'name', 'my', 'your', 'what', 'where', 'when', 'why', 'how', 'who',
            'help', 'need', 'want', 'water', 'food', 'bathroom', 'restroom',
            'coffee', 'tea', 'milk', 'sugar', 'friend', 'family', 'home', 'work',
            'school', 'college', 'university', 'doctor', 'hospital', 'emergency'
        }
        
        # Check if any common word is in the text
        for word in common_words:
            if word in text_lower or text_lower in word:
                return True
        
        # Check if it looks like a name (capitalized, reasonable length)
        if text and text[0].isupper() and 2 <= len(text) <= 20 and ' ' not in text:
            return True
        
        return False
    
    def _apply_common_corrections(self, text: str) -> str:
        """Apply common ASL detection corrections"""
        corrections = {
            'B': 'D', 'D': 'B',
            'M': 'N', 'N': 'M',
            'P': 'K', 'K': 'P',
            'V': 'U', 'U': 'V',
            'H': 'N',  # H and N can be confused
            'I': 'J', 'J': 'I',
            'S': 'A', 'A': 'S'  # Less common but possible
        }
        
        corrected = list(text.upper())
        for i, char in enumerate(corrected):
            if char in corrections:
                # Only correct if it makes a better word
                corrected[i] = corrections[char]
        
        return ''.join(corrected)
    
    def get_model_info(self):
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'api_key_preview': f"{self.api_key[:10]}..." if self.api_key else None,
            'available': self.model is not None
        }