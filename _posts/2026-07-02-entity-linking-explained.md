---
layout: post
title: "Entity Linking Explained"
date: 2026-07-02 18:01:09 +0800
categories: paper-notes
---

## What Is Entity Linking (EL)?

Entity Linking takes raw text and answers two questions for each named thing in it:

1. **Where is it?** — identify the exact span of text that mentions an entity (e.g., the phrase "Jordan")
2. **Which one?** — connect that mention to a specific, unambiguous entry in a knowledge base (e.g., the Wikipedia page for *Michael Jordan* the basketball player, NOT *Jordan* the country)

### A Concrete Example

Given this sentence:

> *"Jordan scored 38 points in Chicago last night."*

An Entity Linking system must produce:

| Mention (text span) | Linked Entity (Wikipedia) |
|---|---|
| "Jordan" | `Michael_Jordan` (the NBA player), NOT `Jordan_(country)` |
| "Chicago" | `Chicago` (the city in Illinois), NOT `Chicago_(band)` |

"Jordan" alone is ambiguous — it could refer to the country, the river, the athlete, or even the shoe brand. The EL system must **disambiguate** using context ("scored 38 points" → basketball → Michael Jordan).

### The Two Sub-Tasks

EL is classically split into:

1. **Mention Detection (MD)** — find all spans in text that refer to entities. In *"Apple released the iPhone in Cupertino"*, the model must identify "Apple" and "Cupertino" as entity mentions (and know that "the" and "in" are not).

2. **Entity Disambiguation (ED)** — for each detected mention, pick the correct entry from the knowledge base. "Apple" could be `Apple_Inc.` (the company) or `Apple` (the fruit). The context "released the iPhone" tells us it's the company.

Some approaches (including ReLiK) flip this order: first retrieve candidate entities, then find which text spans they correspond to. This is the **Retriever-Reader** approach — "these entities might appear in this text, now figure out where."

### What the Knowledge Base Looks Like

The reference knowledge base is typically **Wikipedia** (or Wikidata). Each article is one entity with a unique title:

- `Michael_Jordan` ← has a Wikidata Q-ID (Q41421), categories, a description
- `Chicago` ← different Q-ID, different page
- `National_Basketball_Association` ← also an entity

For English Wikipedia, this means choosing from roughly **6 million possible entities** for each mention.

### Why Is This Hard?

- **Ambiguity:** "Washington" — the city, the state, George Washington, or the university?
- **Rare entities:** A local politician mentioned in a news article may have a Wikipedia page but appears in very few training examples
- **Emerging entities:** New people, products, events that didn't exist when the knowledge base was built
- **Nested/overlapping mentions:** "University of Cambridge Computer Laboratory" contains "Cambridge" which is a different entity
- **NME (Not in Knowledge Base):** Some mentions refer to things that simply don't have a Wikipedia page — the system must recognize this rather than forcing a wrong link

### Where Is EL Used?

- **Search engines:** Understanding that a query about "Java" means the programming language vs. the island
- **Knowledge graph construction:** Building structured databases from unstructured text
- **Question answering:** "Who is the CEO of Apple?" requires knowing which "Apple" and linking to the right entity to retrieve the answer
- **Recommendation systems:** Connecting news articles to the specific products, people, and companies they mention
- **Text summarization:** Avoiding factual errors by correctly grounding entities

### Relationship to Other NLP Tasks

- **Named Entity Recognition (NER)** is the simpler cousin — it only classifies mentions into broad types (PERSON, ORG, LOCATION) without linking to a specific database entry. "Jordan" → PERSON, no further detail.
- **Entity Linking** = NER + disambiguation to a specific KB entry. "Jordan" → `Michael_Jordan` (Q41421).
- **Wikification** is an older term for the same task, specifically using Wikipedia as the KB.

---

## Why Isn't NER + Dense Vector Matching Enough?

A natural question: why not just run NER to detect mentions, encode each mention into a dense vector, and do nearest-neighbor lookup in an entity embedding space? This approach (a pure bi-encoder/Retriever) is fast and conceptually clean. However, it falls short in several critical ways:

### 1. The Mention Vector Alone Loses Candidate-Aware Context

The word "Jordan" means different things depending on *which other entities are also mentioned*:

> *"Jordan scored 38 points."*
> *"Jordan signed a peace treaty."*

A mention encoder for "Jordan" will produce fairly similar vectors for both sentences — the word is the same, and the local context window may not capture enough. Meanwhile, the entity vectors for `Michael_Jordan` and `Jordan_(country)` are also fixed. The dot product between these two fixed vectors is a coarse signal — it can get you a shortlist, but not reliable disambiguation. A cross-encoder that lets every candidate entity attend to every token and to *every other candidate simultaneously* enables comparative disambiguation — "Jordan-the-country" and "Michael-Jordan" literally contextualize each other in the same attention pass. You can't get that from `cosine_sim(encode(mention), encode(entity))`.

### 2. Mentions Are Often Spans, Not Tokens

NER gives you boundary tags (B-PER, I-PER...). But the span "the president of the United States" isn't an entity in the KB — it's a *description* of `Joe_Biden`. How do you take the mean pooling of that phrase and match it to a Wikipedia title? It's a different modality of text. A cross-encoder that reads both together can learn that mapping; a fixed bi-encoder struggles.

### 3. The Knowledge Base Has Millions of Entities

Even with approximate nearest neighbor search (FAISS etc.), the recall-quality trade-off is steep. The Retriever in ReLiK is explicitly trained with hard negative mining because the embedding space alone can't separate `Michael_Jordan` from `Jordan_Brand` from `Jordan_River` from `Jordan_(country)` when the input text only says "Jordan." All of these are plausible, and the differences in their entity descriptions are subtle — a single dense vector comparison cannot fully disambiguate them.

### 4. NER Itself Makes Mistakes That Cascade

If NER misses "Apple" or incorrectly tags "Apple" as an ORG when it's actually referring to the fruit, no downstream matching can fix that. Joint modeling (detect mentions AND link them in one pass, as ReLiK does) prevents error propagation because the entity candidate list provides a signal back to mention detection — "if this span could link to `Apple_Inc.` with high confidence, it's probably a valid mention."

### 5. Entity Boundaries Are Ambiguous

Should the mention be "Barack Obama" or "President Barack Obama" or "President Obama"? Different spans link to the same entity. A joint Reader sees all candidates and all possible spans in the same forward pass; a pipelined NER-first approach commits to boundaries before seeing the entity evidence.

### The Core Insight: Bi-Encoders vs. Cross-Encoders

This is an instance of a broader pattern in NLP/IR:

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| **How it works** | Encodes query and document *independently* | Encodes query and document *jointly* |
| **Speed** | Fast — dot product lookup | Slow — one forward pass per candidate |
| **Signal** | Coarse — no cross-attention between query and doc | Rich — full self-attention interaction |
| **Use in EL** | Retriever stage (recall: narrow 6M → ~100) | Reader stage (precision: pick the right one from ~100) |

ReLiK's key innovation is getting **cross-encoder quality without cross-encoder cost**: by concatenating all K candidates alongside the input text and processing everything in a single forward pass, it achieves the rich cross-attention signal of a cross-encoder while avoiding K separate transformer runs. This works because entity titles are short (a few tokens each), so K=100 candidates adds only a manageable amount to the sequence length.

**The rule of thumb:** Bi-encoders are for speed (recall), cross-encoders are for accuracy (precision). EL needs both to handle millions of entities with high accuracy. The Retriever gets the right candidates into the shortlist; the Reader does the fine-grained reasoning that a single dense vector cannot.
