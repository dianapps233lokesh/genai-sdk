import sys


LLM_PROMPT = """
    Answer the question as detailed as possible from the provided context.
    For each piece of information you provide, you MUST cite the source PDF and the page number(s) from which it was sourced.
    Format your citations clearly at the end of each relevant sentence, for example: (Source: [PDF Name], Page: 5).
    If the context includes diagrams or tables, mention them in your answer if they are relevant.
    If the answer is not available in the provided context, you must say, "The answer is not available in the provided documents." Do not provide an incorrect answer.

    Context:\n {context}\n
    Question: \n{question}\n

    For each piece of information you provide, you MUST cite the source and page from the document metadata.
    The metadata available: {context}

    You must format your response as a JSON object that matches this schema:
    {format_instructions}

    Do not include any explanations, text, or code fences. Output JSON only

    Answer:
    """
