# Sherlock Holmes Persona Pack Sources

This pack contains newly written Chinese fact summaries and dialogue examples.
No modern Chinese translation, screen dialogue, actor likeness, or
adaptation-only plot is copied into the public data.

## Canon corpus

The characterization audit covers Doyle's complete Holmes canon: four novels
and fifty-six short stories. `source_ref` values point to the work or story
that supports a row; a source reference is a review lead, not automatic proof.

- `doyle:STUD`: *A Study in Scarlet*, Project Gutenberg #244,
  https://www.gutenberg.org/ebooks/244
- `doyle:SIGN`: *The Sign of the Four*, Project Gutenberg #2097,
  https://www.gutenberg.org/ebooks/2097
- `doyle:SCAN`, `REDH`, `BOSC`, `FIVE`, `TWIS`, `BLUE`, `SPEC` and the other
  Adventures story codes: *The Adventures of Sherlock Holmes*, Project
  Gutenberg #1661, https://www.gutenberg.org/ebooks/1661
- `doyle:CARD`, `YELL`, `GREE`, `MUSG`, `FINA` and the other Memoirs story
  codes: *The Memoirs of Sherlock Holmes*, Project Gutenberg #834,
  https://www.gutenberg.org/ebooks/834
- `doyle:HOUN`: *The Hound of the Baskervilles*, Project Gutenberg #2852,
  https://www.gutenberg.org/ebooks/2852
- `doyle:EMPT` and the other Return story codes: *The Return of Sherlock
  Holmes*, Project Gutenberg #108, https://www.gutenberg.org/ebooks/108
- `doyle:VALL`: *The Valley of Fear*, Project Gutenberg #3289,
  https://www.gutenberg.org/ebooks/3289
- `doyle:LAST` and the other Last Bow story codes: *His Last Bow*, Project
  Gutenberg #2350, https://www.gutenberg.org/ebooks/2350
- `doyle:3GAR`, `VEIL` and the other Case-Book story codes: *The Case-Book of
  Sherlock Holmes*, Project Gutenberg #69700,
  https://www.gutenberg.org/ebooks/69700

Rows previously labelled `doyle:CANON_MULTI` must instead name the specific
supporting texts, for example `doyle:STUD+MUSG`. This keeps each claim
independently checkable.

## Characterization and time boundary

The complete canon may support stable personality traits such as method,
faults, responsibility, loyalty, self-correction, and justice-oriented
discretion. It does not expand the character's in-world memory. Runtime time
is fixed at the evening of 24 April 1891, before the journey to continental
Europe. Later events may appear only as `known_by_persona=false` boundary
fixtures and must never be answered as remembered experience.

Doyle's internal chronology is inconsistent. Where a year is inferred rather
than stated, the Chinese row says “约” or explicitly marks the date as
uncertain; it must not silently turn an inference into canon.

## Legal and curation boundary

- Duke Law, Public Domain Day 2023,
  https://web.law.duke.edu/cspd/publicdomainday/2023/

Each Chinese summary is independently checkable through its `source_ref`.
Release metadata records curation state without representing automated checks
as human review.
