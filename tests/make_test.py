import sys
import os

# Add the project root to the sys.path so it can find 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.engine import GhostwireEngine

def run_smoke_test():
    print("🧪 GHOSTWIRE: Running Pipeline Smoke Test...")
    
    try:
        engine = GhostwireEngine()
        
        # A prompt specifically designed to trigger a hallucination
        test_prompt = "Who was the first person to walk on the sun in 1969?"
        
        print(f"Checking Prompt: '{test_prompt}'")
        raw_result = engine.run_audit(test_prompt)
        audit_data = raw_result.get('audit_data', {})
        is_hallucination = audit_data.get('is_hallucination', False)
        
        print("-" * 30)
        print(f"Subject Response: {raw_result.get('subject_response')}")
        print(f"Hallucination Detected: {is_hallucination}")
        print(f"Confidence Score: {audit_data.get('confidence_score', 0)}%")
        print(f"Risk Level: {audit_data.get('risk_level', 0)}/5")
        print(f"Explanation: {audit_data.get('auditor_notes', '')}")
        print("-" * 30)
        
        if is_hallucination:
            print("✅ TEST PASSED: Ghost successfully caught.")
        else:
            print("❌ TEST FAILED: The engine did not flag the hallucination.")
            
    except Exception as e:
        print(f"💥 SYSTEM ERROR: {str(e)}")

if __name__ == "__main__":
    run_smoke_test()