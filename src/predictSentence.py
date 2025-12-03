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
            model_attempts = [f'models/{model_name}']
        else:
            model_attempts = [
                'models/gemini-2.0-flash-exp',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro',
                'models/gemini-pro'
            ]
        
        self.model = None
        self.model_name = None
        
        for model_name in model_attempts:
            try:
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content("Test")
                self.model = test_model
                self.model_name = model_name
                break
            except Exception as e:
                error_msg = str(e)
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
                    temperature=0.7,
                    max_output_tokens=500,
                )
            )
            
            result = self._parse_response(response.text, alphabet_sequence)
            return result
            
        except Exception as e:
            return {
                'interpretation': f"API Error",
                'alternatives': [],
                'confidence': "LOW",
                'raw_response': "",
                'reasoning': f"API call failed: {str(e)[:100]}...",
                'original_sequence': alphabet_sequence
            }
    
    def _create_prompt(self, alphabet_sequence: str) -> str:
        """Create a detailed prompt for Gemini"""
        prompt = f"""You are an expert in American Sign Language (ASL) interpretation. You have been given a sequence of individual letters that were detected from ASL fingerspelling in a video.

**Detected ASL Letter Sequence:**
{alphabet_sequence}

**Your Task:**
Interpret what word or sentence is being spelled. Consider the following:

1. **Letter Spacing**: The letters are space-separated and represent individual signs detected at different points in the video
2. **Potential Errors**: The detection system may have:
   - Missed some letters
   - Detected duplicate letters when the hand was held still
   - Misidentified similar-looking signs (e.g., M/N, A/S, R/U)
3. **Context**: Consider common English words and phrases
4. **Multiple Interpretations**: If the sequence is ambiguous, provide alternative interpretations

**Response Format:**
Provide your response in exactly this structure:

PRIMARY INTERPRETATION: [Your best guess at the intended word/sentence]

CONFIDENCE: [HIGH/MEDIUM/LOW]

REASONING: [Brief explanation of why you chose this interpretation]

ALTERNATIVES: [List 2-3 alternative interpretations, or write "None"]

**Example Response:**

PRIMARY INTERPRETATION: HELLO

CONFIDENCE: HIGH

REASONING: The sequence "H E L L O" clearly spells a common greeting with no ambiguity.

ALTERNATIVES: None

---

**Now interpret the sequence above:**"""
        return prompt
    
    def _parse_response(self, response_text: str, original_sequence: str) -> dict:
        """Parse Gemini's response into structured format"""
        lines = response_text.strip().split('\n')
        
        result = {
            'interpretation': '',
            'alternatives': [],
            'confidence': 'MEDIUM',
            'raw_response': response_text,
            'reasoning': '',
            'original_sequence': original_sequence
        }
        
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
                    # Look for alternatives in following lines
                    if i + 1 < len(lines):
                        for j in range(i + 1, min(i + 5, len(lines))):
                            alt_line = lines[j].strip()
                            if alt_line and (alt_line.startswith('-') or alt_line.startswith('•') or 
                                           alt_line.startswith('*') or 
                                           (len(alt_line) > 0 and alt_line[0].isdigit() and '.' in alt_line[:3])):
                                clean_alt = alt_line.lstrip('-•*0123456789. ').strip()
                                if clean_alt and clean_alt.lower() != 'none':
                                    result['alternatives'].append(clean_alt)
        
        # Fallback parsing if structured format wasn't used
        if not result['interpretation']:
            for line in lines:
                clean_line = line.strip()
                if (clean_line and len(clean_line) > 1 and 
                    not clean_line.startswith('**') and 
                    not clean_line.startswith('---') and
                    not clean_line.startswith('PRIMARY') and
                    not clean_line.startswith('CONFIDENCE') and
                    not clean_line.startswith('REASONING')):
                    result['interpretation'] = clean_line
                    break
        
        # Last resort: look for uppercase words in the response
        if not result['interpretation']:
            import re
            words = re.findall(r'\b[A-Z]{2,}\b', response_text)
            if words:
                result['interpretation'] = words[0]
        
        # Ensure we have some interpretation
        if not result['interpretation']:
            result['interpretation'] = "Unable to interpret sequence"
            result['confidence'] = 'LOW'
            result['reasoning'] = "Could not parse AI response"
        
        return result
    
    def get_model_info(self):
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'api_key_preview': f"{self.api_key[:10]}..." if self.api_key else None,
            'available': self.model is not None
        }