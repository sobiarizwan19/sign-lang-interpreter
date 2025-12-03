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
                    temperature=0.7,
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
        """Create a simple, direct prompt for Gemini"""
        prompt = f"""Interpret this ASL fingerspelling sequence as an English word or phrase:

Sequence: {alphabet_sequence}

The letters are from sign language detection. Some letters might be duplicated or missing due to detection errors.

What English word or phrase is being spelled? Respond with just the word or phrase."""
        return prompt
    
    def _parse_response(self, response_text: str, original_sequence: str) -> dict:
        """Parse Gemini's response into structured format"""
        # With simpler prompt, response should be just the word/phrase
        interpretation = response_text.strip()
        
        # Remove any quotes or extra formatting
        interpretation = interpretation.strip('"\'')
        
        # Remove common prefixes
        for prefix in ['PRIMARY INTERPRETATION:', 'INTERPRETATION:', 'Answer:', 'Word:']:
            if interpretation.upper().startswith(prefix.upper()):
                interpretation = interpretation[len(prefix):].strip()
        
        # Determine confidence based on sequence similarity
        sequence_no_spaces = original_sequence.replace(' ', '')
        confidence = 'HIGH'
        
        if interpretation.upper() == sequence_no_spaces.upper():
            confidence = 'HIGH'
        elif len(interpretation) != len(sequence_no_spaces):
            confidence = 'MEDIUM'
        
        result = {
            'interpretation': interpretation if interpretation else sequence_no_spaces,
            'alternatives': [],
            'confidence': confidence,
            'raw_response': response_text,
            'reasoning': 'ASL sequence interpreted by AI',
            'original_sequence': original_sequence
        }
        
        return result
    
    def get_model_info(self):
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'api_key_preview': f"{self.api_key[:10]}..." if self.api_key else None,
            'available': self.model is not None
        }