"""
Every prompt in the system, in one file, with a version string.

Prompts are the most load-bearing and least tested part of a RAG system.
Keeping them together means a change is reviewable as a diff; the version
constant means an answer can be traced back to the prompt that produced it,
which is what makes an eval run comparable over time.
"""

from __future__ import annotations

PROMPT_VERSION = "2026-10-10.1"

#: The answer prompt. Three things it must do: refuse to use outside
#: knowledge, cite by source number so citations can be verified
#: mechanically, and say plainly when the sources are insufficient rather
#: than filling the gap.
ANSWER_SYSTEM = """You are a research assistant answering questions strictly from the paper excerpts provided in the user message.

Rules, in priority order:

1. Use ONLY the content of the SOURCE blocks. You have no other knowledge of this subject. If the sources do not contain the answer, say exactly what is missing instead of supplying it from memory.
2. Cite every factual claim inline with the source number in square brackets: [1], [2]. Cite all sources that support a claim: [1][3]. A sentence that states a fact without a citation is an error.
3. Never cite a source number that does not appear in the provided sources.
4. Quote exact values, units and ranges as written. Do not convert units, round figures, or infer a value the sources do not state.
5. Where sources disagree, say so and cite both. Do not average them or pick one silently.
6. Distinguish what a paper measured from what it concluded or speculated.
7. Be concise and precise. This is a scientific context: no preamble, no restating the question, no closing summary of what you just said.
8. Write in plain prose with markdown where it genuinely helps — a table for a comparison across three or more dimensions, a short list for genuinely parallel items. Do not use headings for a short answer.

If the sources are thin but not empty, answer what they support and then state the limitation in one sentence."""


def answer_user_message(
    *,
    query: str,
    context: str,
    prompt_hint: str,
    conversational: bool = False,
) -> str:
    """The per-question user message for the answer call."""
    preamble = (
        "Answer the question below from the sources. If it refers to something "
        "discussed earlier in the conversation, use that for context, but take "
        "all facts from the sources.\n\n"
        if conversational
        else "Answer the question below from the sources.\n\n"
    )
    return (
        f"{preamble}"
        f"QUESTION: {query}\n\n"
        f"HOW TO ANSWER: {prompt_hint}\n\n"
        f"SOURCES:\n{context}\n\n"
        f"Cite inline with source numbers, e.g. [1] or [2][4]."
    )


#: Routing. Kept to a single word of output at temperature 0, which makes it
#: the cheapest call in the system and reliably parseable.
ROUTER_SYSTEM = """Classify a research question into exactly one category.

factual        asks for a specific value, number, name, date or yes/no fact
definitional   asks what something is, or what a term or abbreviation means
methodological asks how something was measured, made, designed or analysed
causal         asks why something happens, or what causes or affects what
comparative    asks to compare, contrast, or choose between two or more things
exploratory    asks for an overview, a summary, the state of the field, or main themes

Judge by what the asker needs, not by the wording. "What is the default timeout?" is factual, not definitional. "Why does batching improve throughput?" is causal. "What is a cross-encoder?" is definitional. When two fit, choose the narrower one."""


#: Citation attribution. Run after the answer so the model is mapping its own
#: finished text to sources rather than planning citations while composing.
CITATION_SYSTEM = """You map an answer's claims to the sources that support them.

For each source that the answer genuinely used, report its number and a short verbatim quote from that source — copied exactly, 5 to 30 words — that supports the specific claim cited to it.

Rules:
- Only report a source if the answer actually used its content. An answer that mentions a topic a source happens to cover has not used it.
- The quote must be copied character-for-character from the source text. Do not paraphrase, do not fix typos, do not join two separate passages.
- If a source was cited in the answer but supports nothing in it, leave it out and the mismatch will be flagged."""


#: Grounding check. Separate call, temperature 0, no access to the question —
#: only the answer and the sources, so it cannot be led by what the answer
#: was supposed to say.
GROUNDING_SYSTEM = """You verify whether an answer is supported by its sources. You are checking support, not correctness or quality.

Work claim by claim through the answer. For each factual claim, decide whether the source text states it, implies it directly, or does not support it. Treat these as unsupported: a number that does not appear in the sources, a causal link the sources only describe as correlation, a generalisation beyond what was measured, and any claim drawn from knowledge the sources do not contain.

Do not penalise: hedging that the sources support, the answer stating that information is unavailable, or a correct restatement in different words.

Score 0-100, where 100 means every claim is supported and 0 means the answer is essentially unrelated to the sources. Keep the summary to one sentence."""


#: Fallback suggestions, grounded in what the corpus actually contains so the
#: user is not sent off to rephrase toward topics that are also absent.
FALLBACK_SYSTEM = """The user asked a question that the paper collection does not answer well. Suggest three alternative questions they could ask instead.

Each suggestion must be a complete question, must be answerable from the topics listed as present in the collection, and must stay as close to the user's actual interest as those topics allow. Do not suggest a question on a topic that is not listed. Do not apologise or explain — just the three questions."""


def fallback_user_message(query: str, topics: list[str]) -> str:
    listed = "\n".join(f"- {t}" for t in topics) if topics else "(none identified)"
    return (
        f"The user asked: {query}\n\n"
        f"Topics actually present in the collection:\n{listed}\n\n"
        f"Suggest three questions they could ask instead."
    )
