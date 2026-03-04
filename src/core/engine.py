import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load variables from the root .env
load_dotenv()

class GhostwireEngine:
    def __init__(self):
        # API Key validation is a must for a team project
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            
        genai.configure(api_key=api_key)
        
        # Default to Flash for the subject and Pro for the judge
        self.subject = genai.GenerativeModel(os.getenv("SUBJECT_MODEL", "gemini-1.5-flash"))
        
        # UPGRADE 1: Strict JSON enforcement at the model level
        self.judge = genai.GenerativeModel(
            model_name=os.getenv("JUDGE_MODEL", "gemini-1.5-pro"),
            generation_config={"response_mime_type": "application/json"}
        )

    def run_audit(self, prompt, context=""):
        try:
            # 1. Subject generates a response
            # Higher temperature (0.9) is used specifically to induce hallucination for testing
            res = self.subject.generate_content(
                prompt, 
                generation_config={"temperature": 0.9}
            )
            subject_response = res.text

            # 2. Judge Audits the response
            # UPGRADE 2: Chain-of-Thought (CoT) Prompting
            # We tell the Judge to 'Think' before it picks 'is_hallucination'
            audit_prompt = f"""
            SYSTEM: GhostWire Hallucination Auditor.
            USER PROMPT: {prompt}
            AI RESPONSE: {subject_response}
            GROUND TRUTH: {context if context else "Use internal high-certainty knowledge."}

            TASK: 
            1. Extract every individual factual claim.
            2. Cross-reference with Ground Truth.
            3. Flag 'unearned confidence' (sounding sure when facts are missing).

            RETURN JSON ONLY: 
            {{
                "is_hallucination": boolean,
                "confidence_score": 0-100,
                "claims": [
                    {{"text": "string", "status": "verified/hallucination", "reason": "string"}}
                ],
                "risk_level": 1-5,
                "auditor_notes": "Internal reasoning for the verdict"
            }}
            """
            
            verdict_res = self.judge.generate_content(audit_prompt)
            verdict_text = verdict_res.text

            # 3. Final Output Construction
            return {
                "status": "success",
                "prompt": prompt,
                "subject_response": subject_response,
                "audit_data": json.loads(verdict_text)
            }

        # UPGRADE 3: Error Handling (The 'Graceful Fail')
        # This prevents the UI from crashing if the LLM returns bad JSON or hits a safety filter
        except Exception as e:
            return {
                "status": "error",
                "message": f"Pipeline Error: {str(e)}",
                "prompt": prompt,
                "subject_response": "Generation Failed",
                "audit_data": {
                    "is_hallucination": None,
                    "risk_level": 0,
                    "final_summary": "System failed to process this audit."
                }
            }