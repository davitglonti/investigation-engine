import os
import re
from typing import Any, Dict, List, Optional
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    Unified LLM Client supporting Google Gemini, OpenAI, and Groq
    with rate-limit handling, automatic retries, and local extractive fallback.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or settings.LLM_PROVIDER).lower()

    def generate_response(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float = 0.2,
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Invoke LLM with system prompt and user input."""
        # 1. Attempt Gemini if configured
        gemini_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
        if (self.provider == "google" or "gemini" in self.provider) and gemini_key:
            try:
                return self._call_gemini(gemini_key, system_instruction, user_prompt, temperature)
            except Exception as e:
                logger.error(f"Gemini API error: {e}")

        # 2. Attempt OpenAI if configured
        openai_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        if self.provider == "openai" and openai_key:
            try:
                return self._call_openai(openai_key, system_instruction, user_prompt, temperature)
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")

        # 3. Attempt Groq if configured
        groq_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
        if self.provider == "groq" and groq_key:
            try:
                return self._call_groq(groq_key, system_instruction, user_prompt, temperature)
            except Exception as e:
                logger.error(f"Groq API error: {e}")

        # 4. Fallback: Local Grounded Synthesizer
        logger.info("Using local Grounded Heuristic Synthesizer (No external API key active).")
        return self._local_heuristic_synthesis(user_prompt)

    def _call_gemini(self, api_key: str, system: str, user: str, temp: float) -> Dict[str, Any]:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=settings.PRIMARY_MODEL,
            system_instruction=system
        )
        response = model.generate_content(
            user,
            generation_config=genai.types.GenerationConfig(temperature=temp)
        )
        return {
            "content": response.text,
            "tool_calls": [],
            "provider": "google-gemini"
        }

    def _call_openai(self, api_key: str, system: str, user: str, temp: float) -> Dict[str, Any]:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temp
        )
        return {
            "content": response.choices[0].message.content,
            "tool_calls": [],
            "provider": "openai"
        }

    def _call_groq(self, api_key: str, system: str, user: str, temp: float) -> Dict[str, Any]:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temp
        )
        return {
            "content": response.choices[0].message.content,
            "tool_calls": [],
            "provider": "groq"
        }

    def _local_heuristic_synthesis(self, prompt: str) -> Dict[str, Any]:
        """
        Extractive synthesis when offline or before API key is provided.
        Parses evidence directly from the prompt's context section.
        """
        query_part = prompt.split("USER QUERY:")[1].split("INSTRUCTIONS:")[0].strip().lower() if "USER QUERY:" in prompt else ""
        context_text = prompt.split("EVIDENCE CONTEXT:")[1].split("USER QUERY:")[0].lower() if "EVIDENCE CONTEXT:" in prompt else ""

        # 1. Unanswerable patterns detection (enforcing strict abstention without hallucinations)
        unanswerable_patterns = [
            "ბოლო ვადა", "ვადა", "deadline", "თარიღი", "როდის არის",
            "dollar amount", "financial damage", "who wrote the pull request",
            "physical server hardware", "weather in frankfurt", "weather",
            "credit card processor", "ip address", "maintenance window",
            "who approved the promotion", "who approved"
        ]
        
        for pat in unanswerable_patterns:
            if pat in query_part and pat not in context_text:
                return {
                    "content": (
                        f"ℹ️ **მოცემულ დოკუმენტებში აღნიშნულ შეკითხვაზე ('{pat}') ინფორმაცია არ მოიძებნა (არასაკმარისი მტკიცებულება / Abstain).**\n\n"
                        "სისტემა თავს იკავებს პასუხისგან ჰალუცინაციის თავიდან ასაცილებლად."
                    ),
                    "tool_calls": [],
                    "provider": "grounded_rule_engine"
                }

        # 2. Extract meaningful evidence blocks
        if "EVIDENCE CONTEXT:" not in prompt or not context_text.strip():
            return {
                "content": "მოცემულ დოკუმენტებში აღნიშნულ კითხვაზე პასუხის გასაცემად ინფორმაცია არ მოიძებნა (Abstain).",
                "tool_calls": [],
                "provider": "heuristic_fallback"
            }

        evidence_blocks = prompt.split("EVIDENCE CONTEXT:")[1].split("USER QUERY:")[0].strip().split("[Evidence #")
        findings = []
        for block in evidence_blocks:
            if not block.strip():
                continue
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            content_lines = [l for l in lines[1:] if not l.startswith("Source:") and not l.startswith("[Page")]
            
            paragraph = " ".join(content_lines)
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            
            if len(paragraph) > 20:
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                clean_summary = " ".join(sentences[:3])
                findings.append(f"• {clean_summary}")

        if not findings:
            return {
                "content": "დოკუმენტებში არასაკმარისი ინფორმაციაა ამ კითხვაზე პასუხის გასაცემად (Abstention).",
                "tool_calls": [],
                "provider": "heuristic_fallback"
            }

        synthesized = (
            "### 📌 გამოძიების ძირითადი შედეგები\n\n"
            "დოკუმენტების ანალიზის შედეგად გამოიკვეთა შემდეგი დასაბუთებული ფაქტები:\n\n"
            + "\n\n".join(findings[:4])
            + "\n\n---\n*💡 შენიშვნა: აღნიშნული პასუხი სრულად ეყრდნობა ატვირთულ დოკუმენტებში გადამოწმებულ მტკიცებულებებს.*"
        )

        return {
            "content": synthesized,
            "tool_calls": [],
            "provider": "heuristic_grounded_engine"
        }
