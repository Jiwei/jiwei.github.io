---
layout: post
title: "SQL Fuzzy and Partial String Matching — Complete Guide"
date: 2026-07-14 17:18:39 +0800
categories: paper-notes
---

---

## 1. The LIKE Operator — Foundations

### 1.1 Which Data Types Support LIKE

LIKE is fundamentally a pattern-matching operator for **character/string types** only. Support for other types depends on the database's implicit type conversion:

| Data Type | MySQL/MariaDB | PostgreSQL | SQL Server | Oracle |
|---|---|---|---|---|
| CHAR / VARCHAR / NVARCHAR / TEXT | ✅ Native | ✅ Native | ✅ Native | ✅ Native |
| Numeric (INT, DECIMAL) | ✅ via implicit cast | ❌ Error (must CAST explicitly) | ⚠️ Partial | ✅ via implicit cast |
| Date/Time (DATE, DATETIME) | ✅ via implicit cast | ❌ Error (must CAST explicitly) | ❌ Not advised | ❌ Error (ORA-01861) |

**Recommendation:** Always use `CAST(column AS CHAR)` for non-string types to avoid database-specific behavior:

```sql
-- Safe cross-database approach
SELECT * FROM users WHERE CAST(age AS CHAR) LIKE '2%';
SELECT * FROM orders WHERE CAST(order_date AS CHAR) LIKE '2026%';
```

### 1.2 LIKE Without Wildcards = Same as =

When no wildcards are present, LIKE behaves identically to `=`:

```sql
WHERE name LIKE 'abc'    -- functionally identical to =
WHERE name = 'abc'       -- same performance, same result
```

The database optimizer detects the absence of wildcards and treats `LIKE 'abc'` as an **exact point lookup**, using B-Tree indexes identically to `=`. The only difference is trailing-space handling (MySQL `=` pads spaces; `LIKE` does not).

> LIKE without wildcards offers **zero fuzzy matching capability**. It catches no typos, no partial matches, no similar sounds.

### 1.3 Wildcard Position Determines Index Usage (B-Tree Logic)

A B-Tree index is like a printed dictionary — sorted alphabetically from A to Z:

| Pattern | Index Used? | Operation | Performance |
|---|---|---|---|
| `LIKE 'abc'` | ✅ Yes | Index Seek / Lookup | ⚡ Super Fast |
| `LIKE 'abc%'` | ✅ Yes | Index Range Scan | 🚀 Fast |
| `LIKE '%abc'` | ❌ No | Full Table Scan | 🐢 Slow |
| `LIKE '%abc%'` | ❌ No | Full Table Scan | 🐢 Slow |

**Why leading wildcards break indexes:** A B-Tree is sorted left-to-right. Finding all values *ending* with "abc" requires scanning every entry — like finding all words ending in "-tion" in a dictionary sorted by first letter.

This applies equally to character columns — the data type doesn't matter; only the wildcard position matters.

---

## 2. True Fuzzy Matching Techniques

LIKE with wildcards is pattern matching, not fuzzy matching. For actual fuzziness, different tools are needed.

### 2.1 Edit Distance (Typo-Tolerant)

| Technique | What It Measures | Database Support |
|---|---|---|
| **Levenshtein Distance** | Min insertions/deletions/substitutions to transform A→B | PostgreSQL (`fuzzystrmatch`), MySQL (custom UDF only) |
| **Damerau-Levenshtein** | Same + transpositions (swap adjacent chars) | PostgreSQL (`fuzzystrmatch`) |
| **Jaro-Winkler** | Similarity weighted for matching prefixes | PostgreSQL (`pg_jellyfish`), SQL Server (CLR) |

**Critical limitation:** Levenshtein **cannot use standard indexes**. It forces a full table scan. Mitigation: pre-filter with indexed columns first, then apply Levenshtein only on the small filtered set.

```sql
-- PostgreSQL: pre-filter then fuzzy match
WITH filtered AS (
  SELECT * FROM users WHERE department = 'engineering'  -- indexed filter
)
SELECT * FROM filtered WHERE levenshtein(name, 'Jhon') <= 2;
```

### 2.2 Trigram Similarity (Index-Friendly Fuzzy)

Trigrams break strings into overlapping 3-character chunks. Similarity = fraction of shared trigrams.

```
"apple" → {"  a", " ap", "app", "ppl", "ple", "le "}
```

| Database | Availability | Index Support |
|---|---|---|
| **PostgreSQL** | `pg_trgm` extension | ✅ GIN/GiST — up to 1000× faster than Levenshtein |
| **MySQL** | Not native | — |
| **SQL Server** | Not native | — |

```sql
-- PostgreSQL pg_trgm
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_name_trgm ON users USING GIN (name gin_trgm_ops);

SELECT *, similarity(name, 'John Smith') AS score
FROM users
WHERE name % 'John Smith'    -- uses the index!
ORDER BY score DESC;
```

### 2.3 Phonetic Matching (Sound-Alike)

| Technique | Quality | Support |
|---|---|---|
| **Soundex** | Low — 4-char code, English-only | All major databases |
| **Metaphone** | Medium — better than Soundex | PostgreSQL (`fuzzystrmatch`) |
| **Double Metaphone** | Good — primary + alternate encoding | PostgreSQL (`fuzzystrmatch`) |

Soundex matches "Smith" with "Smyth" but not "Schmidt." All phonetic methods are English-centric.

### 2.4 Full-Text Search (FTS)

| Database | Syntax | Key Features |
|---|---|---|
| **PostgreSQL** | `to_tsvector(col) @@ to_tsquery('term')` | Stemming, ranking, thesaurus, best F1 scores |
| **MySQL** | `MATCH(col) AGAINST('term' IN BOOLEAN MODE)` | Word prefix with `*`, ngram parser for CJK |
| **SQL Server** | `CONTAINS(col, 'term')`, `FREETEXT(col, 'term')` | Thesaurus, proximity, weighting |
| **Oracle** | `CONTAINS(col, 'term')` (Oracle Text) | Rich linguistic features |

FTS excels at word-level linguistic matching (stemming: "running" → "run") but does **not** natively catch typos.

---

## 3. Database-by-Database Comparison

| Feature | PostgreSQL | MySQL | SQL Server | Oracle |
|---|---|---|---|---|
| Trigram index | ✅ `pg_trgm` | ❌ | ❌ | ❌ |
| Levenshtein | ✅ `fuzzystrmatch` | Custom UDF only | CLR only | UTL_MATCH available |
| Soundex | ✅ | ✅ | ✅ | ✅ |
| Metaphone | ✅ | ⚠️ Limited | ❌ | ❌ |
| FTS with stemming | ✅ Best quality | ✅ Good | ✅ Good | ✅ Good |
| Regex in queries | ✅ `~` operator | ✅ `REGEXP` | ⚠️ Limited | ✅ `REGEXP_LIKE` |
| LIKE on numeric column | ❌ Error | ✅ Implicit cast | ⚠️ Partial | ✅ Implicit cast |
| SIMILARITY function | ✅ `pg_trgm` | ❌ | ❌ | ❌ |

---

## 4. Low-Cardinality Columns and Performance

### 4.1 The Problem: Index Selectivity

The database optimizer's decision to use an index depends on **selectivity**:

```
selectivity = distinct_values / total_rows
```

| Scenario | Selectivity | Optimizer Behavior |
|---|---|---|
| Primary key, every row unique | 1.0 (very high) | Prefers index |
| 1M rows, 1000 distinct values | 0.001 (low) | May skip index, full scan |
| Gender column, 2 values | ~0 (extremely low) | Almost certainly full scan |

### 4.2 Practical Impact

For ENUM columns or columns with only a dozen distinct values across millions of rows:

- A standard B-Tree index may **not be used** by the optimizer — the cost of random I/O to look up rows exceeds the cost of a sequential scan
- **Use `=` for exact matching** on the distinct values (fast, can use index)
- **Then apply LIKE only on the filtered result set** (tiny, negligible cost)

```sql
-- Good: filter first with =, then LIKE on small result
SELECT * FROM orders 
WHERE status = 'pending'             -- exact match, uses index or efficient scan
  AND description LIKE '%urgent%';   -- only runs on pending rows
```

> Low cardinality does **not** make LIKE faster. But using `=` to pre-filter before LIKE completely sidesteps the problem.

### 4.3 Special Case: MySQL ENUM

ENUM values are stored internally as integers (1, 2, 3...). `=` on ENUM is fast. `LIKE` on ENUM forces a string conversion — slower, and won't use the internal integer representation.

---

## 5. Chinese (CJK) Substring Matching

### 5.1 The Challenge

Chinese text has unique properties that affect search:

```
"海尔集团深圳有限公司"
 ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
 Each character = 1 logical character = 3 bytes (utf8mb4)
 No spaces between words — no natural token boundaries
 A "word" can be 1-4+ characters
```

Example query: find "海尔集团" within "海尔集团深圳有限公司" (substring match).

### 5.2 Technique Comparison

#### Prefix Match — B-Tree Index (⚡ Simplest, Fastest)

For prefix matching, a standard B-Tree index works perfectly for Chinese because characters have a defined sort order in UTF-8/Unicode:

```sql
-- Works for prefix match only
SELECT * FROM companies WHERE name LIKE '海尔集团%';
-- Uses standard B-Tree index if one exists on `name`
```

> This is the optimal solution for Chinese prefix matching. Zero extra cost.

#### Arbitrary Substring Match — PostgreSQL pg_trgm (🚀 Recommended)

```sql
-- Install extension
CREATE EXTENSION pg_trgm;

-- Create trigram index
CREATE INDEX idx_name_trgm ON companies USING GIN (name gin_trgm_ops);

-- All of these use the index:
SELECT * FROM companies WHERE name LIKE '%海尔%';     -- mid-string
SELECT * FROM companies WHERE name LIKE '%有限公司';  -- suffix match
SELECT * FROM companies WHERE name LIKE '%深圳%';     -- anywhere
```

**How pg_trgm handles Chinese:** Each Chinese character is treated as a character-position in the trigram. The GIN index stores all trigrams, and queries match through shared trigrams — followed by exact verification. This is the best SQL-native solution for Chinese substring search in PostgreSQL.

For extremely large datasets, consider `pg_bigm` (bigram extension): bigrams can be more efficient than trigrams for CJK characters, producing smaller indexes.

#### Arbitrary Substring Match — MySQL FULLTEXT + ngram Parser

```sql
-- Create FULLTEXT index with ngram parser
ALTER TABLE companies ADD FULLTEXT INDEX idx_name_ft (name) WITH PARSER ngram;

-- Exact phrase match (双引号)
SELECT * FROM companies 
WHERE MATCH(name) AGAINST('"海尔集团"' IN BOOLEAN MODE);
```

**How the ngram parser works (default n=2):**

```
Index time:
"海尔集团深圳有限公司" 
→ bigram tokens: "海尔" "尔集" "集团" "团深" "深圳" "圳有" "有限" "限公" "公司"
  positions:        1      2      3      4      5      6      7      8      9

Query "海尔集团":
→ tokens: "海尔" "尔集" "集团"

Phrase match: tokens found at consecutive positions (1,2,3) → 匹配成功 ✅
```

#### Suffix Match — Reverse Column Trick (MySQL)

```sql
-- Store reversed string for suffix matching
ALTER TABLE companies ADD COLUMN name_reversed VARCHAR(255);
UPDATE companies SET name_reversed = REVERSE(name);
CREATE INDEX idx_name_rev ON companies (name_reversed);

-- Suffix query becomes prefix query, uses B-Tree index
SELECT * FROM companies 
WHERE name_reversed LIKE CONCAT(REVERSE('有限公司'), '%');
```

This is a clever hack: reversing the string turns suffix matching into prefix matching, making it indexable with a standard B-Tree.

#### Elasticsearch + IK Analyzer (Industrial Scale)

```
IK Analyzer tokenization:
"海尔集团深圳有限公司" → ["海尔集团", "深圳", "有限公司"]

Query "海尔集团" → directly hits token "海尔集团"
```

Industry standard for Chinese full-text search at scale. Supports word segmentation, synonyms, pinyin search, and traditional/simplified conversion.

### 5.3 FULLTEXT vs LIKE — Where They Differ for Chinese

| | LIKE | FULLTEXT + ngram |
|---|---|---|
| Single-character query (`LIKE '%海%'`) | ✅ Works | ❌ Fails (ngram_token_size limit, default 2) |
| Exact substring match | ✅ 100% precise | ✅ Via `"phrase"` syntax |
| Wildcard patterns (`%海_集团%`) | ✅ Supported | ❌ Not supported |
| Index acceleration with leading `%` | ❌ Full table scan | ✅ Uses FULLTEXT index |
| Relevance scoring | ❌ None | ✅ Built-in relevance ranking |
| Behavior determinism | 100% deterministic | Affected by `ngram_token_size`, `innodb_ft_min_token_size`, stopwords |

**For the specific use case of exact Chinese substring matching (≥2 characters):** FULLTEXT + `"phrase"` syntax works correctly and efficiently.

**For 100% precision with index acceleration:**

```sql
-- FULLTEXT for fast recall, LIKE for exact verification
SELECT * FROM companies 
WHERE MATCH(name) AGAINST('+海尔 +集团' IN BOOLEAN MODE)
  AND name LIKE '%海尔集团%';
```

FULLTEXT rapidly filters to candidates (eliminating 99.9% of irrelevant rows), LIKE does final exact verification on the tiny candidate set.

### 5.4 Chinese Substring Search Decision Matrix

| Scenario | Recommended Solution |
|---|---|
| Prefix match "海尔%" | B-Tree index + `LIKE '海尔%'` |
| Arbitrary substring, PostgreSQL | `pg_trgm` + GIN index |
| Arbitrary substring, MySQL | FULLTEXT ngram + `"phrase"` syntax |
| Suffix match only, MySQL | Reverse column trick + B-Tree index |
| 100M+ rows, high concurrency | Elasticsearch + IK Analyzer |
| Semantic search (understand "海尔" = company name) | PostgreSQL + zhparser |
| Need 100% guarantee + index | FULLTEXT filter + `LIKE` verification |

---

## 6. Performance Decision Tree

```
Need to search substrings?

├─ Is it always a prefix? (LIKE 'abc%')
│   └─ YES → Use standard B-Tree index. Done.
│
├─ Small table (<100K rows)?
│   └─ Use LIKE '%term%'. Performance hit is negligible.
│
├─ PostgreSQL?
│   ├─ Typo-tolerant?
│   │   └─ CREATE EXTENSION pg_trgm; GIN index; WHERE col % 'term'
│   ├─ Word-level search with stemming?
│   │   └─ tsvector + GIN index + @@ to_tsquery
│   └─ Exact edit distance on pre-filtered data?
│       └─ CTE pre-filter → levenshtein() on small result set
│
├─ MySQL?
│   ├─ Word-level search?
│   │   └─ FULLTEXT index + MATCH...AGAINST IN BOOLEAN MODE
│   ├─ Chinese substring match?
│   │   └─ FULLTEXT + ngram parser + "phrase" syntax + LIKE verification
│   └─ Substring with typo tolerance?
│       └─ Consider PostgreSQL migration, or external search engine
│
├─ SQL Server?
│   ├─ FTS: CONTAINS / FREETEXT
│   └─ Fuzzy: CLR integration or external engine
│
└─ Oracle?
    ├─ FTS: Oracle Text CONTAINS
    └─ Fuzzy: UTL_MATCH for edit distance (small data only)
```

---

## 7. Quick Reference — Per Use Case

| Use Case | Recommended Solution |
|---|---|
| Prefix autocomplete | B-Tree index + `LIKE 'term%'` |
| Typo-tolerant name search at scale | PostgreSQL `pg_trgm` with GIN index |
| English sound-alike name search | `METAPHONE()` or `SOUNDEX()` with functional index |
| Document search with stemming | FTS (`tsvector` PG, `FULLTEXT` MySQL, `CONTAINS` SQL Server) |
| Substring anywhere, large table | PostgreSQL `pg_trgm` GIN index (enables `LIKE '%term%'` to use index) |
| Chinese prefix match | B-Tree index + `LIKE '海尔%'` |
| Chinese arbitrary substring, MySQL | FULLTEXT ngram `"phrase"` + `LIKE` verification |
| Chinese arbitrary substring, PostgreSQL | `pg_trgm` + GIN index |
| CJK / multi-byte language search | PostgreSQL `pg_trgm` or MySQL ngram FTS parser |
| Exact typo distance on small filtered set | Levenshtein in CTE with pre-filtering |
| Very large scale, typo-tolerant, real-time | Elasticsearch or Meilisearch |
