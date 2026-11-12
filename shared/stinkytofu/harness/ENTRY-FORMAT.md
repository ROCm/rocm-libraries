# Entry format

Read this before you write or change a database entry. The rules for using entries are in `DATABASE.md`.

One Markdown file per entry, with frontmatter:

```markdown
---
name: <short-slug>
kind: pattern | insight | finding | threshold | procedure | memory
status: candidate | confirmed     # not used by memories
certainty: verified | measured | hypothesis   # for patterns, findings, insights
triggers: <a few keywords, comma separated, that tell an agent when this entry applies>
arch: <the architecture it was observed on, for example gfx1250, or `any` if it does not depend on one>
elsewhere: unknown | likely | no   # does it hold on other architectures? `unknown` is the honest default
source: <page URL, repo file path, PR or issue number, experiment id, or person; may be empty for a memory>
source_revision: <the page's revision number when the entry was written, if the source is a page>
confirmed_by: <name>              # set by a human, together with status
last_verified: <date>
scope: <target or kernel family this applies to>
---
> **Claim:** <one sentence a person can agree or disagree with>
> **Certainty:** <same as the field>. **Evidence:** <what, where, when, how many runs>
> **Not checked:** <the biggest gap>
> **Next test:** <one line, for measured and hypothesis>
> **To confirm:** read this card. If the claim, the evidence, and "not checked" are right, set
> `status: confirmed` and `confirmed_by: <name>`.

<details: how to detect it, why it matters, what was ruled out, how to apply>
```

**The review card.** The first lines of the body are a short card, so a person can decide in under a
minute whether the entry is worth noting down. Everything after it is detail for whoever uses the entry.

**Certainty** says how sure the claim is. It is set by whoever writes the entry, and a person can change it.
- `verified`: a fix, an independent source (the hardware docs, the code), or a repeated test confirms it.
- `measured`: seen in our runs, but the cause is not proven.
- `hypothesis`: an explanation or a rule that has not been tested yet. Treat it as a place to start.

**Architecture.** `arch` says where the evidence comes from, and `elsewhere` says what is believed about the
other architectures. For an entry that depends on an architecture, the card's "Not checked" line should say
that the other architectures were not checked.

Certainty is not status. `status` says whether a person reviewed the entry. `certainty` says how sure the
claim is. A reviewed entry can still be a hypothesis.

`index.md` is generated from the entries' frontmatter. Do not edit it by hand. It has one short line per
entry: `- [<name>](<path>) | <kind> | <certainty> | <status> | <arch> | <triggers>`. Use `-` for the certainty of a
procedure or a memory. When you add or change an entry, regenerate the index with the database's index
script. If the database has no script, add or fix the line yourself in that format.

A source can be a page, a file in a repo, a PR or issue, an experiment, or a person. Only a page needs
`source_revision`. If the source is a file in this repo, point to it and read it when needed, and do
not copy it into the entry. A memory (what a run learned) may have no source. Any other entry with no
`source` is not accepted in review. `last_verified` shows which entries are going
stale.
