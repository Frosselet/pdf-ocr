## LinkedIn Post

**We started with AI. Then we started fixing what AI got wrong. That's when the real work began.**

Every time we fixed an LLM extraction failure, we noticed the same thing: the model understood the table. The problem was mechanical. Counting rows. Matching headers to schema fields. Recognizing that a layout with labels down the left side is transposed.

So we built deterministic heuristics. As a filter in front of the LLM, not a replacement for it. Today they handle the majority of documents we process. The LLM sees only the genuinely ambiguous cases.

The industry is moving the other way: bigger models, vision APIs, more AI. We get the pitch. We also see what it costs: stochastic output, opaque debugging, subtle failures that get harder to catch as models get smarter.

But the deeper question is about the data itself. Most of what you need to process arrives from outside your organization, in formats you didn't choose, governed by standards you didn't set. You accommodate it or you don't use it. That makes extraction a circular economy problem: heal what arrives, reinject it in better form, let the improvement compound.

Where do you draw the line between what heuristics should handle and what AI should handle? We draw it deliberately, and we keep pushing it.

Full article below.

---

# Where the Line Is

*On drawing the boundary between structure and semantics in document extraction, and why that boundary matters more than which side you're on.*

*This is the second article in a series. The first, [The Data Humans Make](the-data-humans-make.md), examines the growing gap between how we produce data and how we need to consume it.*

---

## The AI-first starting point

We started where everyone starts: hand the document to an LLM. Compress it into text, write a prompt, get structured output. It works remarkably well on simple tables. For about a week, it feels like the problem is solved.

Then you hit your first multi-row wrapped header. Or a transposed layout where field labels are rows, not columns. Or a table long enough that the model silently truncates the output halfway through. Or three data sources whose layouts look nothing alike but must produce identical schema output. The LLM fails *plausibly*. It invents column names that sound right but don't exist in the source. It miscounts section boundaries by one row, and the error cascades silently through every downstream record. It works 80% of the time, and the other 20% is where the real work lives.

## The slide toward deterministic

We didn't plan to write heuristics. We planned to write better prompts. But each time we fixed an LLM failure, we noticed the same pattern: the fix was about *mechanics*, not *understanding*. The model understood the table fine. The problem was counting rows. Matching column headers to schema fields. Detecting that a section label is a category name, not a data value. Recognizing that a table with three columns and field names down the left side is transposed.

These are translation problems. Converting visual structure into data structure. And they have deterministic solutions that are faster, cheaper, and perfectly reproducible.

So we started building them. As a filter in front of the LLM, rather than a replacement for it. Try alias-based column mapping first. If every header in the table matches a known schema alias, you're done: zero LLM calls, zero hallucination risk, zero latency. If the table is transposed, detect it structurally and build the records directly. If the LLM does run, validate its output by counting rows from the source text and correcting any miscounts before they propagate.

Today, the deterministic path handles the majority of real-world documents we process. The LLM sees only the genuinely ambiguous cases, and even then, deterministic post-processing validates its output.

## Against the current

This is at odds with what the industry is doing. The dominant trajectory since 2023 is *more* AI, not less. Vision models that see the page as an image. Larger context windows that swallow entire documents. Multi-modal architectures that skip text extraction entirely. The pitch is compelling: why write rules when the model can see?

We understand the pitch. We also understand what it costs. Vision calls are expensive: in dollars, in latency, in the debugging opacity of a system where you can't trace why a field was mapped the way it was. And the fundamental problem remains: LLMs are stochastic. Run the same document twice, get slightly different output. Change the model version, get different output. The smarter the model gets, the more subtle the failures become. Fewer spectacular errors, more quiet ones.

*Mechanical* work should be done *mechanically*. *Interpretive* work should be done by a model that can interpret. The real question is where you draw the line between structure and semantics. We draw it deliberately, and we keep pushing it.

## Heuristics as science

Every deterministic heuristic we write is designed from first principles, from a human understanding of how documents are structured, not from fitting rules to specific test files.

"Tables where field labels occupy column 0 and data occupies columns 1–N" is a structural class that describes thousands of real documents. "Provider X's quarterly report" is a test case. We encode the former; the latter is what we validate against.

This distinction matters because overfitted heuristics are worse than no heuristics. A rule that improves results on one source but degrades or risks degrading results on another is wrong, regardless of net metrics. Every heuristic must be safe for all documents that match its structural preconditions, including documents we haven't seen yet.

## Contracts as domain boundaries

Source-specific knowledge (aliases, column name variations, category names, formatting quirks) lives in declarative contracts, not in code. One contract defines the canonical schema for a document type. Multiple sources, multiple layouts, one contract, one code path. Swap the contract for a different domain, and the same pipeline extracts data from entirely different document formats.

The pipeline is generic. The contract carries the domain. This separation is what lets us add new document types without writing new code, and what prevents source-specific hacks from accumulating in the codebase.

If this sounds familiar, it should. It is Zhamak Dehghani's [Data Mesh](https://martinfowler.com/articles/data-mesh-principles.html) principle of domain-driven ownership applied at the extraction layer. The contract is a data product specification: it declares what the domain produces, what schema it conforms to, and what aliases bridge the gap between source terminology and canonical output. The pipeline is self-serve infrastructure: domain teams define contracts; the platform executes them. Federated governance lives in the schema: every output, regardless of source, conforms to the same canonical structure.

## The data you don't control

All of this might sound like a story about engineering taste: deterministic where possible, stochastic where necessary, contracts to separate domain from pipeline. And it is. But the reason it matters goes beyond taste.

Consider what actually happens inside an organization that consumes data. You set standards. You define schemas, naming conventions, output formats. You invest in governance. This takes months, sometimes years, because it's cultural work as much as technical work, and cultures don't change in a sprint.

And then most of the data you actually need arrives from outside. Suppliers, regulators, partners, public agencies, third-party platforms. It arrives in whatever format the sender chose, governed by their standards (if they have any), structured for their purposes, not yours. You don't get to set the rules for data you receive. You accommodate it, or you don't use it.

This is what makes the problem circular. Your organization consumes external data in poor formats. You clean it, structure it, enrich it, and produce outputs, which then become someone else's external data. They receive it, struggle with your formatting choices, and clean it again. Every participant in this economy is simultaneously producing data that others find hard to consume and consuming data that others produced without them in mind.

The temptation is to focus on the production side: if everyone just produced better data, the problem would disappear. But "everyone" is every organization, every government, every team that has ever exported to PDF. The production side will not be solved by standards bodies or best-practice documents. Not at the scale and speed at which data moves.

What *can* be solved, continuously, incrementally, at the point of consumption, is the healing. Take what arrives. Understand its structure despite its formatting. Extract the data that matters. Reinject it into the economy in a better form: schema-conformant, machine-readable, semantically grounded. Every document you heal is one less document the next consumer has to fight with.

This is data for good as a daily practice. A continuous, quiet process of absorbing messy inputs and producing clean outputs, applied at every node where human-formatted data enters a system. The improvement compounds. The economy gets incrementally better, because enough participants chose to heal what they received.

This is why the line between structure and semantics is a practical question. Building systems that heal data reliably, repeatably, at the speed at which it arrives, requires knowing which parts of the problem are mechanical and which require judgment. Mechanical problems are the ones you can solve at scale.

## The honest assessment

Is this the right bet? We don't know. If vision models reach the point where they reliably count rows, never hallucinate column names, produce identical output on identical input, and cost nothing, then our deterministic layer becomes unnecessary overhead. That's a real possibility, and maybe not a distant one.

But we observe that each generation of models reduces *spectacular* failures while introducing *subtle* ones. They don't confuse headers with data anymore. They still occasionally merge two adjacent columns, miscount section boundaries, or silently drop a row. These are exactly the failures that deterministic validation catches, and they're the failures that matter most when you're producing data that feeds into downstream systems where accuracy is non-negotiable.

Our bet is that the boundary between "what heuristics should handle" and "what the LLM should handle" will keep moving, but will never disappear. Structure is structure. A table with three columns and labels down the left side will always be transposed, regardless of what model is interpreting it. The question is whether you detect that before or after an API call that might get it wrong.

We'd rather be wrong about where the line is and adjust it than pretend the line doesn't exist.

---

## Part of a composite solution

We want to be precise about what we're claiming. This is one layer, the structural extraction layer, designed to compose with the work being done above, around, and beyond it.

What strikes us, looking at the field today, is how many independent lines of thought are converging on the same underlying insight: that data, metadata, semantics, and agency are facets of one problem, and they need architectures that treat them as such.

Zhamak Dehghani's [Data Mesh](https://martinfowler.com/articles/data-mesh-principles.html) gave this insight its organizational form: domains own their data, data is a product, infrastructure serves the domains, governance is federated. Our contracts are a small expression of this: domain-owned specifications that declare what data a source produces, what schema it conforms to, and what aliases bridge the gap between source terminology and canonical output. The pipeline doesn't know the domain; the contract does.

But a data product needs to be *findable*, and its structure needs to be *legible* across domains. This is the problem Ole Olesen-Bagneux works on with the [Meta Grid](https://www.actian.com/blog/data-management/what-is-the-meta-grid/), a federated metadata architecture built on the observation that metadata is inherently scattered, and that the right response is connection, not centralization. Our contracts are metadata. Today they live as JSON files alongside the code. The question is how they participate in a wider ecosystem where other teams, other tools, other domains can discover and reason about them. The architecture for that already exists in principle; the connective tissue is being built.

And then there is the question of meaning. Jessica Talisman's [Ontology Pipeline](https://jessicatalisman.substack.com/p/the-ontology-pipeline) provides a methodology for building semantic structure progressively: controlled vocabularies, metadata standards, taxonomies, thesauri, knowledge graphs, each layer preparing for the next. The instinct is to place this after extraction, as a way of enriching what comes out. But ontology matters just as much at the input. Our contracts are ontological artifacts: they encode how a domain understands its data *before* a single document is opened. The aliases, the schema fields, the category definitions are knowledge, not just strings. They express what you need from a document and how you will interpret it. You make sense of data before it enters your system, because the contract already carries that sense.

This reinforces the circularity. Extracted data enriches domain knowledge. Domain knowledge refines contracts. Contracts drive better extraction. The ontology pipeline is not a DAG with a start and an end. It is a directed cycle: knowledge feeds extraction, extraction feeds knowledge. Our pipeline sits inside that cycle, at the point where documents become structured records, shaped by the ontological choices that precede it and producing data that will inform the next iteration of those choices.

And then there is the question of who, or what, does all of this. An ecosystem of specialized agents that discover each other, collaborate, and maintain trust. Eric and Davis Broda's [Agentic Mesh](https://www.oreilly.com/library/view/agentic-mesh/9798341621633/) describes this architecture: autonomous agents operating within a governed ecosystem where trustworthiness, traceability, and reliability are structural properties, built in from the start. A document extraction agent that is deterministic where possible and stochastic only where necessary is a natural inhabitant of such a mesh. Its behavior is explainable, its outputs are auditable, and its boundaries are clear. It knows what it does and, equally important, what it doesn't.

What connects these threads is a shared recognition that the old dichotomies (centralized vs. distributed, rules vs. AI, data vs. metadata, structure vs. semantics) are false choices. The emerging architectures dissolve them. Data products carry their own metadata. Ontologies ground themselves in controlled vocabularies that start as simple alias lists. Agents earn trust through deterministic guarantees where the problem permits it, and bounded stochastic behavior where it doesn't. Meshes are meshes precisely because no single node claims to be the whole.

Our work occupies one position in this landscape: the point where human-formatted documents become structured data products. A necessary piece, but only a piece, designed with the awareness that the pieces around it are being built by others, in ways we can connect with but do not need to control.

That openness is the point.

---

*[First article: The Data Humans Make →](the-data-humans-make.md)*

*[Technical documentation →](../src/docpact/README.md)*
