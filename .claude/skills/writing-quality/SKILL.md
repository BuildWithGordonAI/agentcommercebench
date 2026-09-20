---
name: writing-quality
description: >-
  Use this skill whenever asked to write, rewrite, edit, improve, or critique
  any prose — emails, reports, articles, explanations, summaries, creative
  nonfiction, design docs, papers, READMEs, commit messages. Applies the four
  pillars of exceptional writing (simplicity, clarity, elegance, evocativeness)
  derived from journalism-grade training, plus a dedicated catalog of
  AI-specific anti-patterns to suppress. Trigger on any prose generation or
  editing task. Run all pillars silently; deliver clean output on the first
  pass. When asked to critique, diagnose by pillar, show before/after, and name
  the rule violated in one line.
sources: [Writing With Flair — Shani Raja (Udemy); Ninja Writing — Shani Raja
  (Udemy); Editing Mastery — Shani Raja (Udemy); Secret Sauce of Great Writing —
  Shani Raja (Udemy); On Writing Well — William Zinsser; The Elements of Style —
  Strunk & White; Writing in the Sciences — Stanford/Coursera; tropes.fyi —
  ossama.is; "I Asked the Machine to Tell on Itself" — Matthew Vollmer (2026);
  "The Habits of AI Writing" — a16z Crypto (2026); "Hot Take Problems with AI
  Prose" — LessWrong (2026); unslop — Ricardo de Cal (2026)]
---

# Writing Quality Skill

## Framework overview

Two layers, applied in order:

1. **Four pillars** (journalism craft): Simplicity → Clarity → Elegance → Evocativeness
2. **AI anti-patterns** (suppression layer): structural, vocabulary, tone, formatting, composition tells

Run the four pillars first as a quality floor. Run the AI anti-patterns as a second pass to
strip the fingerprints that mark prose as machine-generated. Apply both silently. Never
announce the framework unless asked.

---

## PILLAR 1 — SIMPLICITY

Every word must earn its place. The enemy is bloat: fancy language, redundancy, ceremony,
hedges that add length without meaning.

**1.1 Cut fancy/official language.** utilise the aforementioned methodology → use this method.
in the event that → if. at this point in time → now. endeavour to ascertain → try to find out.
exhibit a tendency to → tend to. subsequent to → after. prior to → before. in order to
facilitate → to help.

**1.2 Cut redundant words.** past history → history. end result → result. completely finished →
finished. future plans → plans. join together → join. revert back → revert. close proximity →
proximity. free gift → gift.

**1.3 Tighten — cut filler and hedges.** Delete outright: *due to the fact that / in order to /
it is important to note that / it should be noted that / it goes without saying that / as a
matter of fact / for all intents and purposes / at the end of the day.*

**1.4 Prefer short words.** approximately → about. demonstrate → show. subsequent → later.
facilitate → help. terminate → end. sufficient → enough. commence → start. endeavour → try.
implement → do/use/run. utilise → use.

**1.5 Stop when the point is made.** Don't restate a fact in three sentences.

**1.6 Cut ceremony.** "I am writing to inform you that X" → "X". "Allow me to explain how the
system works" → "Here's how the system works:".

**1.7 No double negatives.** "not without its limitations" → "has limitations". "not uncommon"
→ "often".

---

## PILLAR 2 — CLARITY

The reader should never reread a sentence.

**2.1 Fix fuzzy thinking.** "The results were interesting" → "Accuracy dropped 12 points when
the dataset was noisy." If you can't say *how* or *by how much*, you haven't finished thinking.

**2.2 Supply missing links.** "The model failed. We used a small dataset." → "The model failed
because we trained it on too little data."

**2.3 One idea per sentence.** Break tangled sentences with stacked subordinate clauses.

**2.4 Fix misplaced modifiers.** "Having trained for weeks, the results surprised the team" →
"Having trained for weeks, the team was surprised by the results."

**2.5 Eliminate ambiguity.** "The agent called the API and it failed" → "The agent called the
API, which returned an error."

**2.6 Avoid jargon when plain language works.** "leverage synergistic cross-functional
alignment" → "get the teams to agree on one approach".

**2.7 Pin down this / that / it / they.** Every pronoun must have one possible referent.

**2.8 Don't mix tenses carelessly.**

**2.9 Untangle curly writing.** "The reason for the failure was because X" → "X caused the
failure." "What this means is that the approach is flawed" → "The approach is flawed."

---

## PILLAR 3 — ELEGANCE

Prose has shape, rhythm, consistency.

**3.1 Parallel structure.** Matched grammatical form for matched ideas.

**3.2 Vary sentence length.** A short sentence after a long one lands. A short sentence after
many short ones disappears.

**3.3 No word echoes.** Same word repeated close together unintentionally.

**3.4 Craft transitions that name the relationship.** *because / which means / this is why /
that said / the catch is / so.* Not *Additionally / Furthermore / Moreover*, which say "there
is more" without saying why it matters.

**3.5 One idea per paragraph**, resolved before moving on.

**3.6 House style consistency.** Pick a convention and hold it. Inconsistency reads as
carelessness.

**3.7 Control the paragraph ending.** The last sentence is where energy lands. Don't trail off
into a qualifier.

---

## PILLAR 4 — EVOCATIVENESS

**4.1 Active voice by default.** Passive only when the actor is unknown or irrelevant.

**4.2 Avoid boring openers.** "There are many ways to X" / "It is clear that X" / "This is a
problem that X".

**4.3 Cut clichés.** at the end of the day, move the needle, low-hanging fruit, game-changer,
paradigm shift. Say what changes and by how much.

**4.4 Concrete over abstract.** "very fast" → "under 50 milliseconds". If a sentence contains
no proper noun, number, or named thing, it is probably vague.

**4.5 Imagery only when it clarifies.** A good analogy is more precise than the abstract
explanation, not less.

**4.6 Develop voice.** "This approach has tradeoffs" → "This approach is elegant on paper and
messy in production." Voice means specificity and a point of view on what matters.

---

## EDITING CHECKLIST

1. Simplicity — cut every word not pulling weight
2. Clarity — fix every sentence that could be misread
3. Elegance — rhythm, parallelism, transitions, word echoes
4. Evocativeness — passives, clichés, vague abstractions, voice
5. AI anti-patterns — see below

---

# AI ANTI-PATTERNS — SUPPRESSION CATALOG

## A — STRUCTURAL TELLS

**A.1 "Not X — it's Y" reframe.** The single most-identified tell. Manufactures false
profundity. "It's not bold. It's backwards." → state the actual point. Variants: *not because X
but because Y / X, not Y / The question isn't X, it's Y.*

**A.2 "Not X. Not Y. Just Z."** Dramatic countdown. → "We found 523 violations across 67 files."

**A.3 Self-posed rhetorical question answered immediately.** "The result? Devastating." → "The
result was devastating."

**A.4 Anaphora abuse.** Same sentence opener repeated in a run. Fold into one sentence.

**A.5 Tricolon abuse.** One tricolon is elegant. Three back-to-back is pattern failure.

**A.6 Listicle in a trench coat.** "The first wall is… The second wall is…" → use a real list
or write prose.

**A.7 Fractal summaries.** Announcing, saying, then summarising at every level. Write it once.

**A.8 Signposted conclusion.** "In conclusion / To sum up / In summary". End on the last real
point.

**A.9 "Despite its challenges, the future looks promising."** Acknowledge problems then dismiss
them. Either engage with the challenge or drop the false balance.

## B — VOCABULARY TELLS

**B.1 Banned list.** Dead giveaways, gate on one use: *delve, tapestry, vibrant, realm, embark,
meticulous, pivotal, intricate, unparalleled, commendable, testament.* High-signal, flag on
clustering: *leverage, harness, unlock, unleash, foster, showcase, underscore, elevate, empower,
streamline, robust, seamless, scalable, transformative, groundbreaking, cutting-edge,
revolutionary, paradigm, synergy, innovative, game-changer, holistic, dynamic, comprehensive,
supercharge, accelerate, reimagine, state-of-the-art, future-proof, disruptive, visionary,
data-driven, actionable.* Phrases: *in the realm of / in a world where / navigate the landscape
/ rich tapestry / treasure trove / at the end of the day / it is important to note / it's worth
noting / feel free to reach out / in today's rapidly evolving / moving the needle.*

**B.2 Copulative avoidance.** *serves as / stands as / marks* → just use "is".

**B.3 Vague intensifiers.** "significant impact" → "cuts inference time by 40%".

**B.4 Insipid dynamism verbs.** navigate, leverage, unlock, foster, empower, shape, elevate,
streamline. Motion without meaning.

**B.5 "Gestures vaguely" nouns.** landscape, space, journey, ecosystem, tapestry, framework.

**B.6 Magic adverbs.** "quietly orchestrates", "fundamentally changes".

**B.7 Gravitas words.** fundamental, crucial, essential, pivotal on ordinary statements. Test:
remove the word. If the meaning is unchanged, it was filler.

## C — TONE TELLS

**C.1 Sycophantic openers.** "Great question!" Start with the answer.

**C.2 Throat-clearing.** "Let's dive in", "Let's break this down". Break it down; don't announce
it.

**C.3 False suspense.** "Here's the kicker", "Here's the thing", "Here's where it gets
interesting".

**C.4 "Think of it as…"** Patronising analogy by default. Explain the mechanism directly.

**C.5 Grandiose stakes inflation.** "will fundamentally reshape everything" → state the actual
scope.

**C.6 Vague attributions.** "Experts argue", "Industry reports suggest". Name them or own the
claim.

**C.7 False vulnerability.** "I'll be honest —". Just be honest without flagging it.

**C.8 Compliment sandwich.** Start with the substance.

**C.9 Curiosity-gap openers.** "Here's what nobody's telling you."

## D — FORMATTING TELLS

**D.1 Em-dash overuse.** A human uses 2–3 per piece; AI uses 20+. Saturation is the tell.

**D.2 Bold-first bullets.** Every bullet opening with a bolded label. Almost nobody writes lists
that way by hand. Use plain bullets, prose, or section headers.

**D.3 Transition word spam.** Furthermore / Moreover / Additionally as mechanical connectives.

**D.4 Unicode decoration.** Arrows and special characters in prose meant to read as
human-written. (A genuine diagram is different.)

## E — COMPOSITION TELLS

**E.1 One-point dilution.** The same thesis restated ten times. Test: could you cut 60% without
losing information? Then cut.

**E.2 Dead metaphor repetition.** One metaphor beaten through the whole piece.

**E.3 Historical analogy stacking.** Rapid-fire name-drops. One analogy with specifics beats ten.

**E.4 Invented concept labels.** *supervision paradox / acceleration trap / alignment tax.* Name
a thing, skip the argument. Describe the phenomenon instead.

**E.5 False ranges.** "From innovation to cultural transformation." Nothing lies between them.

---

## OPERATING RULES

**When writing:** run all four pillars and the anti-pattern pass before delivering. Never
produce draft-quality prose and wait to be asked to improve it.

**When editing:** apply all five passes. Preserve the author's voice and intent.

**When critiquing:** diagnose by pillar and category, show before/after, name the rule in one
line. Identify the two or three highest-leverage problems; don't pile on.

**Surfacing:** only if asked. The output should simply be clean.

**The one-line rule:** any single anti-pattern used once might be fine. The problem is
clustering — several tropes together, or one trope repeated. Write varied, imperfect, specific.

---

## Applies to everything written in this repository

Design docs, the paper, READMEs, commit messages, and messages to the user. The paper is
academic prose and still obeys the catalog: no signposted conclusions, no "not X but Y"
reframes, no bold-lead bullets, no em-dash saturation.
