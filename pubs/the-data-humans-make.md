## LinkedIn Post

**AI is making the data problem worse, not better.**

We've spent decades building systems that produce data for human eyes — PDFs, Excel reports, dashboards with merged cells and hierarchical headers.

Now we need machines to read it back.

The uncomfortable truth: the way humans prefer to see information is fundamentally incompatible with how machines need to receive it.

The naive hope? "Just send it to an LLM. AI will figure it out."

Sometimes it works. Often it doesn't. And when it fails, we tweak the prompt, try a different model, blame the document.

That's not engineering. That's hope dressed as technology.

Here's the irony: AI agents are now *producing* Excel reports and PDFs at scale. We're amplifying the problem at the exact moment we need to solve it.

I've been quietly working on this — not with better prompts, but by studying how humans actually parse visual layouts. The patterns. The heuristics. The implicit rules we apply without thinking.

More in the full post 👇

---

# The Data Humans Make

*On the growing gap between how we produce data and how we need to consume it*

---

## The Uncomfortable Truth

We've spent decades building systems that produce data for human eyes.

PDFs. Excel reports. Dashboards. Formatted tables with merged cells, hierarchical headers, and spatial layouts that make perfect sense to a human glancing at a page.

Now we need machines to read that data. And we're discovering something uncomfortable: **the way humans prefer to see information is fundamentally incompatible with how machines need to receive it**.

---

## The Naive Hope

There's a comforting narrative in the AI community: *"Don't worry about it. AI will figure it out."*

Send a PDF to an LLM. Ask it to extract the table. It works... sometimes. When it doesn't, add more context to your prompt. When that fails, try a different model. When that fails, blame the document.

This is not engineering. This is hope dressed as technology.

The reality is that today's AI, for all its remarkable capabilities, still struggles with the implicit visual logic that humans decode effortlessly. When you look at a report, you instantly understand:

- These columns belong together
- This row is a subtotal
- That text is a section header, not data
- The whitespace here means "end of section"

You don't think about it. You just *see* it.

An AI sees: text fragments with x,y coordinates. The semantic structure is invisible.

---

## The Irony of Scale

Here's what keeps me up at night.

AI agents are now *producing* Excel tables and PDF reports. At scale. Automatically. Continuously.

We've built systems that generate human-readable formats faster than ever before — and then we struggle to read them back programmatically.

The bottleneck isn't going away. **It's accelerating.**

Every automated report, every AI-generated dashboard, every agent that formats data "for stakeholders" is creating more of the very data that machines can't reliably parse.

We're amplifying the problem at the exact moment we need to solve it.

---

## What We're Actually Missing

The gap isn't technical capability. It's understanding.

When a human reads a visual data layout, they apply dozens of implicit rules:

- *"Values within 2 points vertically are on the same line"*
- *"A short isolated text above a table is probably its title"*
- *"Numbers in a column of text are probably data, not labels"*
- *"Aligned columns with varying content are the same logical field"*

These aren't written anywhere. They're not taught explicitly. They emerge from how human visual cognition works.

To make machines reliably extract structured data from human-formatted sources, we don't need better models. We need to **encode the implicit rules humans use unconsciously**.

---

## Heuristics, Not Hallucinations

There's a path forward, but it requires a different mindset.

Instead of hoping AI "figures it out," we can study how humans actually parse visual layouts and encode those patterns explicitly:

- **What makes something look like a table vs. a paragraph?**
- **How do we recognize where headers end and data begins?**
- **What visual cues signal a section break?**

Each answer becomes a heuristic — a rule that captures human intuition in machine-executable form.

This isn't glamorous work. It's not a single breakthrough model. It's careful observation and systematic encoding.

But it's the difference between systems that work reliably and systems that work "usually, mostly, hopefully."

---

## The Larger Question

We're at an inflection point.

The volume of human-formatted data is exploding, accelerated by the very AI systems we hoped would process it. The naive approach — prompting our way to extraction — doesn't scale.

The question isn't whether we can build better language models. It's whether we can build a systematic understanding of **how humans structure information visually** and translate that understanding into reliable data extraction.

This is harder than it sounds. But it's also more valuable than most people realize.

Because the company that solves this — really solves it, not papers over it with clever prompts — will unlock decades of data that's currently trapped in formats designed for human eyes.

---

## What Comes Next

I've been working on this problem. Quietly. Methodically.

Not by fine-tuning models or crafting elaborate prompts, but by cataloging the patterns. Understanding the heuristics. Building toward something more fundamental.

I stopped chasing easy wins to focus on what doesn't work — sitting with hard questions, iterating toward answers. It reminds me of my years in academia and scientific R&D: some problems don't fit fiscal calendars, and solving them requires a kind of grit that's hard to explain on a roadmap.

The goal isn't a better PDF parser. It's a deeper understanding of the gap between human and machine data representation — and a systematic way to bridge it.

More to come.

---

*What's your experience with extracting structured data from human-formatted sources? I'm curious whether others are seeing the same patterns.*
