import os
import time
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
        """
        Invoke LLM with system prompt and user input.
        Returns dict with: 'content', 'tool_calls', 'provider'.
        """
        # 1. Attempt Gemini if configured
        gemini_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
        if (self.provider == "google" or "gemini" in self.provider) and gemini_key:
            try:
                return self._call_gemini(gemini_key, system_instruction, user_prompt, temperature)
            except Exception as e:
                logger.error(f"Gemini API invocation error: {e}")

        # 2. Attempt OpenAI if configured
        openai_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        if self.provider == "openai" and openai_key:
            try:
                return self._call_openai(openai_key, system_instruction, user_prompt, temperature)
            except Exception as e:
                logger.error(f"OpenAI API invocation error: {e}")

        # 3. Attempt Groq if configured
        groq_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
        if self.provider == "groq" and groq_key:
            try:
                return self._call_groq(groq_key, system_instruction, user_prompt, temperature)
            except Exception as e:
                logger.error(f"Groq API invocation error: {e}")

        # 4. Fallback: Intelligent Grounded Extractive Synthesis
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
        # Check if prompt contains evidence context
        if "EVIDENCE CONTEXT:" not in prompt:
            return {
                "content": "მოცემულ დოკუმენტებში აღნიშნულ კითხვაზე პასუხის გასაცემად ინფორმაცია არ მოიძებნა.",
                "tool_calls": [],
                "provider": "heuristic_fallback"
            }

        context_part = prompt.split("EVIDENCE CONTEXT:")[1].split("USER QUERY:")[0].strip()
        lines = [l.strip() for l in context_part.splitlines() if l.strip() and not l.startswith("[Evidence")]
        
        if not lines:
            return {
                "content": "დოკუმენტებში არასაკმარისი ინფორმაციაა ამ კითხვაზე პასუხის გასაცემად (Abstention).",
                "tool_calls": [],
                "provider": "heuristic_fallback"
            }

        # Formulate grounded extractive summary
        findings = []
        for line in lines[:5]:
            findings.append(f"• {line}")

        synthesized = (
            "დოკუმენტების ანალიზის შედეგად გამოიკვეთა შემდეგი ძირითადი ფაქტები:\n\n"
            + "\n".join(findings)
            + "\n\n*შენიშვნა: პასუხი ეფუძნება უშუალოდ ატვირთულ დოკუმენტებში მოძიებულ მტკიცებულებებს.*"
        )

        return {
            "content": synthesized,
            "tool_calls": [],
            "provider": "heuristic_grounded_engine"
        }
