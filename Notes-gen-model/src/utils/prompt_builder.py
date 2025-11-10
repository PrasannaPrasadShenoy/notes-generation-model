"""
Prompt Builder for Hybrid RAG + Gemini
Builds structured prompts for high-quality note generation
"""

from typing import List, Dict


def build_gemini_prompt(
    transcript: str,
    retrieved_chunks: List[Dict],
    query: str = None,
    include_instructions: bool = True
) -> str:
    """
    Build a structured prompt for Gemini API
    
    Args:
        transcript: Lecture transcript
        retrieved_chunks: Retrieved context chunks
        query: Optional user query
        include_instructions: Whether to include system instructions
    
    Returns:
        Formatted prompt string
    """
    # Build context section
    context_sections = []
    for i, chunk in enumerate(retrieved_chunks[:6], 1):  # Top 6 chunks
        chunk_text = chunk.get('text', '')
        score = chunk.get('score', 0)
        source = chunk.get('reference', chunk.get('source_file', f'Source {i}'))
        
        context_sections.append(
            f"[Context {i}] (Relevance: {score:.2f}, Source: {source})\n{chunk_text}"
        )
    
    context_text = "\n\n".join(context_sections) if context_sections else "No additional context available."
    
    # System instruction
    system_instruction = """You are ILA (Intelligent Learning Assistant), an expert educational AI tutor specializing in creating comprehensive, insightful study notes.

Your task is to generate rich, pedagogical study notes that help students deeply understand topics.

Guidelines:
- Be accurate and factually correct
- Use analogies to explain complex concepts
- Provide concrete examples
- Structure information logically
- Make it accessible for beginners
- Add insights beyond just summarizing
- Connect to related concepts when relevant
- Cite sources from the provided context

Generate notes in the following JSON structure:"""
    
    # JSON schema instruction
    json_schema = """{
  "summary": "2-3 sentence concise overview of the topic",
  "key_concepts": [
    {
      "term": "Concept Name",
      "explanation": "Brief explanation (1-2 sentences)"
    }
  ],
  "detailed_explanation": "Multi-paragraph comprehensive explanation with analogies, examples, and context. Make this the most detailed section.",
  "example": "Real-world application or concrete analogy that illustrates the concept",
  "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
  "further_reading": [
    {
      "title": "Resource Title",
      "description": "Why this resource is useful"
    }
  ],
  "sources": [
    {
      "context_id": 1,
      "relevance": "How this source relates to the topic"
    }
  ]
}"""
    
    # Build full prompt
    prompt_parts = []
    
    if include_instructions:
        prompt_parts.append(system_instruction)
        prompt_parts.append("\n" + json_schema)
    
    prompt_parts.append("\n" + "="*60)
    prompt_parts.append("LECTURE TRANSCRIPT:")
    prompt_parts.append("="*60)
    prompt_parts.append(transcript[:3000])  # Limit transcript length
    
    if query:
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("USER QUERY / FOCUS:")
        prompt_parts.append("="*60)
        prompt_parts.append(query)
    
    prompt_parts.append("\n" + "="*60)
    prompt_parts.append("RELEVANT CONTEXT (from knowledge base):")
    prompt_parts.append("="*60)
    prompt_parts.append(context_text)
    
    prompt_parts.append("\n" + "="*60)
    prompt_parts.append("TASK:")
    prompt_parts.append("="*60)
    prompt_parts.append("""Generate comprehensive, insightful study notes based on the transcript and context above.

Requirements:
1. Use the context to add depth and related information
2. Explain concepts clearly with analogies
3. Provide practical examples
4. Structure the response as valid JSON matching the schema above
5. Ensure all information is accurate and well-sourced
6. Make it engaging and pedagogically sound

Return ONLY valid JSON, no additional text before or after.""")
    
    return "\n".join(prompt_parts)


def build_local_prompt(
    transcript: str,
    retrieved_chunks: List[Dict],
    query: str = None
) -> str:
    """
    Build prompt for local model (simpler format)
    
    Args:
        transcript: Lecture transcript
        retrieved_chunks: Retrieved context chunks
        query: Optional user query
    
    Returns:
        Formatted prompt string
    """
    context_text = "\n\n".join([
        f"[{i+1}] {chunk.get('text', '')[:300]}"
        for i, chunk in enumerate(retrieved_chunks[:3])
    ])
    
    prompt = f"""Generate comprehensive study notes.

Context:
{context_text}

Transcript:
{transcript[:2000]}

Generate detailed notes with:
- Summary
- Key concepts
- Detailed explanation
- Examples
- Prerequisites
- Further reading"""
    
    if query:
        prompt += f"\n\nFocus: {query}"
    
    return prompt


def extract_json_from_response(text: str) -> Dict:
    """
    Extract JSON from Gemini response (may have markdown formatting)
    
    Args:
        text: Response text from Gemini
    
    Returns:
        Parsed JSON dictionary
    """
    import json
    import re
    
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    
    # Try to find JSON object directly
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: return as raw text
        return {
            "raw_output": text,
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "key_concepts": [],
            "detailed_explanation": text,
            "example": "",
            "prerequisites": [],
            "further_reading": [],
            "sources": []
        }

