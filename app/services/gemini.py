import httpx

from app.core.config import settings


class GeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent?key={self.api_key}"

    async def generate_answer(self, prompt: str, context: str, explain: bool = False) -> str:
        role_description = "You are an expert in Algerian law."
        if explain:
            role_description = "You are an expert in Algerian law. Explain the law in a very simple way, as if you are talking to a 5-year-old child. Use easy words and simple examples."

        full_prompt = f"""{role_description}

Use the context below to answer the question.
If the answer is partially موجود، حاول الإجابة قدر الإمكان اعتماداً على النص.
If there is no relevant information at all, say: لا أعلم.

Context:
{context}

Question:
{prompt}
"""
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.url, json=payload, timeout=30.0)

                if response.status_code == 429:
                    return "Rate limit exceeded. Please wait a moment and try again."

                response.raise_for_status()
                data = response.json()

                return data["candidates"][0]["content"]["parts"][0]["text"]
            except httpx.HTTPStatusError as e:
                return f"API Error: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"Unexpected Error: {str(e)}"

    async def recommend_lawyers(self, text: str, lawyers: list) -> str:
        lawyers_info = ""
        for i, lawyer in enumerate(lawyers):
            lawyers_info += f"ID: {lawyer.get('id')}, Name: {lawyer.get('full_name')}, Specialty: {lawyer.get('specialty')}, Location: {lawyer.get('location')}, Experience: {lawyer.get('experience')} years, Rating: {lawyer.get('average_rating')}, Bio: {lawyer.get('description')}\n"

        prompt = f"""You are the official Algerian Law AI assistant. Your task is to recommend the top 3 best-matching lawyers for a user's case.

User Request: {text}

Database of available lawyers:
{lawyers_info}

STRICT GUIDELINES:
1. Act as if you have internal access to this database. NEVER mention "the list provided", "the names sent to me", or "client-side data".
2. Do NOT mention lawyers that you did NOT select. Only focus on the top 3 (or fewer if there are very few matches).
3. Be EXTREMELY concise. No introductory filler (e.g., skip "I have analyzed your request...").
4. For each selected lawyer, provide:
   - Lawyer Name (Bold)
   - Brief reason for selection (1 sentence max).
   - Specialty and Location.
   - The tag: [SCHEDULE_MEETING:ID:NAME]
5. If no lawyers match well, politely suggest contacting the local bar association (نقابة المحامين) without listing any names from the database.
6. Use the user's language (Arabic, French, or English).
"""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.url, json=payload, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                return f"Error recommending lawyers: {str(e)}"


gemini_service = GeminiService()
