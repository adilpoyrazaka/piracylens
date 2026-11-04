# Data card: Google copyright removals from Web Search

This card describes the dataset as PiracyLens uses it: where it comes from, which snapshot every figure is based on, how the film/TV scope was drawn, and what the data cannot tell. It covers the takedown data used by the audit, forecasting and domain clustering tasks. The video data for copy detection is described separately.

## 1. Dataset and snapshot

The data is the Google Transparency Report bulk export of copyright removal requests for Web Search. Each request is a notice sent by a reporting organization on behalf of a copyright owner, asking Google to remove URLs from search results. For each request Google reports how many URLs it removed, how many were not in its index, how many it took no action on, and how many are still pending.

### Snapshot

| Field | Value |
| --- | --- |
| File | `google-websearch-copyright-removals.zip` |
| Source | https://storage.googleapis.com/transparencyreport/google-websearch-copyright-removals.zip |
| Downloaded | 2026-10-26 |
| Size | 15,739,542,261 bytes |
| SHA-256 | `6ce3aaf6a31bc834660b1d8644c5796fa42a67f910da5671fb48908659f5707d` |
| Requests covered | 2011-03-10 to 2026-10-24 |

Google updates the export regularly. A new download is a new snapshot: it will not reproduce the figures in this card exactly and should be recorded separately.

### Tables

The export contains three CSV files, about 89.7 GB unpacked. They are converted to Parquet without loss:

| Table | One row per | Rows | Key columns |
| --- | --- | ---: | --- |
| `requests` | takedown request | 57,273,319 | request time (UTC), owner and reporter id and name, four URL counters |
| `domains` | request × domain | 963,823,791 | domain, the same four counters |
| `urls_no_action` | URL with no action taken | 456,643,466 | domain, URL |

How the conversion was checked:

- The `domains` and `urls_no_action` row counts equal the CSV line counts minus the header.
- `requests` has 739 fewer rows than lines, and every one of the 739 is a line break inside an owner name. No rows are lost.
- The four counters sum to the same totals in `requests` and `domains`: 15,133,536,607 removed, 4,732,078,771 not in index, 952,920,600 no action, 82,095,678 pending.
- The `from_abuser` column is `false` in every row of all three tables, so it carries no information and is not used.

## 2. Terms of use

Neither the Transparency Report page nor the README inside the export contains a license statement. Without an explicit license, this project treats the data conservatively:

- **No redistribution.** The raw files and the derived Parquet files stay on the local machine; `data/` is git-ignored. The repository publishes code and aggregate figures only.
- **Non-commercial use.** The project is a portfolio and educational prototype. If it moved to commercial use, its use of this data would be reviewed again before that step.
- **Personal names.** Owner and reporter names are free text written by the senders, and some of them are the names of private individuals. Public documents, figures and the demo show organization names only. Individuals appear in counts, never by name.
- **Neutral wording.** The data shows what was sent and what Google decided. It does not show whether a request was justified or who actually sent it. Public text describes measured behaviour, such as "requests filed under the owner's own name", and does not attach intent to a named organization.

## 3. Window and counts

All results come from one modeling window: requests sent from **2021-01-01 to 2026-10-24**, the last request time in the snapshot. Earlier years are used only as feature history for the start of the window, never as training or test examples. Two planned checks test whether results depend on this choice: a forecasting run trained from 2019, and an audit run restricted to 2023 onwards.

Starting in 2021 is a design choice, not a measured optimum. Two measurements support it. The share of volume held by the largest owners barely changes between start years 2019, 2021 and 2023. And from 2021 the series has five full yearly cycles for seasonal forecasting.

| Year | Requests |
| --- | ---: |
| 2019 (history only) | 1,484,751 |
| 2020 (history only) | 1,888,343 |
| 2021 | 2,808,222 |
| 2022 | 3,127,082 |
| 2023 | 5,487,995 |
| 2024 | 7,418,286 |
| 2025 | 11,013,863 |
| 2026 (to 24 October) | 16,944,795 |
| **Window total** | **46,800,243** |

The whole snapshot holds 57,273,319 requests from 2011-03-10 onwards. In the window:

- Volume grows every year. The first ten months of 2026 already exceed all of 2025 by 54%. The data does not say whether this is real growth or a change in how requests are filed or reported (section 8).
- 2026 is a partial year, so forecasting leaves the last incomplete period out of evaluation.
- 2,060 requests (0.004%) cannot be assigned to an owner: 2 have no owner name, and 2,058 have a name made only of symbols or invisible characters, which normalizes to nothing (section 5). They stay in every denominator and are counted as out of scope.

## 4. How the scope was defined

The models work on film/TV takedowns only. The scope is defined by a rule, and the owner list is that rule applied:

> A copyright owner is in scope if its main business is film or TV production, distribution, broadcasting or streaming.

Out of scope: sports rights holders, games and software, music, books, press and education, webtoons, manga and novels, adult content, private individuals, intermediaries that cannot be identified, and compound credits naming several companies from different groups. Anime production, broadcasting and streaming count as film/TV; manga publishing does not. For a telecom or technology company, the named entity decides: its TV or streaming unit is in, the company as a whole is out.

### Candidates

Candidates were generated once, from four sources fixed in advance, over requests in the window. The parameters are in `configs/scope_candidates.yaml` and were committed before any candidate list existed. Every threshold is a heuristic choice.

| Source | What it catches | Threshold | Candidates |
| --- | --- | --- | ---: |
| (a) Volume ranking | Owner keys with high request volume | ≥5,000 requests | 465 |
| (b) Russian TV protection network | Owners served by four reporter groups that protect Russian TV and film | ≥500 requests through these reporters | 141 |
| (c) Film/TV agencies | Owners served by 19 reporting agencies picked by hand as working mainly on film/TV | ≥500 requests through these agencies | 514 |
| (d) Name-prefix scan | Owner keys starting with a pre-written list of group and brand names | ≥5 requests | 624 |

Together with a seed list of 104 well-known film/TV owner keys, the sources produced **1,302 candidates**. Sources overlap, so the column does not sum to the total. The agencies in (c) were picked from a table of reporter volumes and seed shares, and the choice was committed before any owner candidate was generated.

### Review

Each candidate was decided by hand against the rule, in batches by request volume. The review followed six written rules:

1. The scope rule above decides, not the candidate source.
2. Owners are grouped by their corporate owner at the end of the data (section 7).
3. For telecom and technology companies, the named entity decides.
4. An owner with no company or web evidence is in only if at least 80% of its requests came through the film/TV agencies **and** its name is a specific company name with film/TV wording (film, cinema, studio, TV, channel, broadcasting and similar). The 80% threshold is heuristic.
5. A clear name variant of an in-scope brand is in, except compound credits, generic words and claimants that cannot be tied to an organization.
6. No evidence, no scope.

No per-owner audit label, no-action rate or model output was looked at before the scope was frozen. That way the list could not be steered toward a result.

| Decision | Reason | Owner keys |
| --- | --- | ---: |
| In | film/TV rule | 499 |
| In | seed list reconfirmed | 104 |
| Out | books, press, education | 134 |
| Out | private individuals | 119 |
| Out | no evidence | 107 |
| Out | music | 83 |
| Out | games, software | 56 |
| Out | webtoons, manga, novels | 47 |
| Out | sports | 28 |
| Out | compound credits | 22 |
| Out | unidentified intermediaries | 19 |
| Out | prefix matched an unrelated name | 19 |
| Out | government bodies | 13 |
| Out | adult | 11 |
| Out | gambling | 11 |
| Out | whole company not film/TV | 11 |
| Out | telecom or internet company | 8 |
| Out | claimants that cannot be tied to an organization | 6 |
| Out | other | 5 |
| | **Total** | **1,302** (603 in, 699 out) |

Later checks reversed 8 decisions, each with its reason recorded: six out → in, two in → out. The final scope holds **603 owner keys in 228 groups**, frozen with the git tag `scope-v1`.

Every decision is listed in `docs/scope_review_summary.csv`. In-scope keys appear with their group, brand and labels. Out-of-scope keys are named for categories whose owners are organizations, such as sports, games and software, telecoms and government bodies. Private individuals, unidentified claimants and all other out-of-scope categories appear there only as counts per reason.

## 5. Owner grouping

### Why not `owner_id`

The data has an `owner_id` column, but it does not identify an owner: it splits one name across many ids. Among single-owner requests since 2019 there are 6.49 M ids for 1.58 M names. 312,331 names carry more than one id, and 67,684 ids carry more than one name. Grouping by id would split one company's requests into many pieces, so owners are grouped by their name instead.

### From name to key

`normalize_owner` turns each raw owner name into a key with fixed, tested rules:

- decode HTML entities and remove invisible Unicode characters
- fold case
- fix Latin look-alike letters inside Cyrillic words, then transliterate Cyrillic to Latin
- strip quotes and separators
- remove legal-form suffixes as whole words only (for example LLC, GmbH, OOO)
- collapse repeated tokens

There is no fuzzy matching, so two different companies cannot be merged by a similarity score. On this snapshot 1,702,749 distinct names become 1,544,395 keys, a reduction of about 9%. One streaming service's 25 spellings, for example, collapse into one key.

49 names consist only of symbols or invisible characters and produce no key. They account for the 2,058 window requests counted as unassignable in section 3.

### From key to group

Normalization cannot know that two different names belong to the same company. That mapping is kept by hand in `configs/owner_aliases.yaml`, which has three levels:

| Level | Meaning | Count |
| --- | --- | ---: |
| Key | A normalized owner name | 603 |
| Brand | The name the market knows, such as a channel or a streaming service | 349 |
| Group | The corporate owner at the end of the data (section 7) | 228 |

Each brand carries its operating `market`. Each group carries the owner's `region` and a `content_type` (`tv`, `film` or `mixed`). These three fields are heuristic labels, used for breakdowns and features, not as measured facts. A repository test checks that the file is well formed and that no key appears twice, and every key is checked against the data before a commit.

### Reporting organizations

Reporter names go through the same normalization. `configs/reporter_aliases.yaml` merges the accounts of one organization (28 keys, 12 brands, 8 groups), for example an agency that renamed itself twice and moved its filings to a new account each time. An account joins a group only when its client portfolio shows it is the same organization. Accounts that file only under their own name stay separate. Variants with fewer than 100 requests in the window are not listed, a heuristic cut-off (section 8).

## 6. Coverage

All figures are request counts in the window. A request counts once, even when its owner name maps to several keys. No audit label or outcome was read. Produced by `python -m piracylens.data.scope_metrics`.

### Scope share

**8,247,558 of 46,800,243 window requests (17.62%) are in scope.** The rest belong to owners outside the rule (music, books, games, sports and others), to owners never caught as candidates, or have no owner key.

### Recall within the film/TV reporters

How much of the work of known film/TV reporters does the scope capture?

| Reporter set | Window requests | In scope | Out of scope: reviewed and left out | Out of scope: never a candidate |
| --- | ---: | ---: | ---: | ---: |
| Russian TV protection network (4 groups) | 5,687,570 | 96.4% | 171,731 | 30,456 |
| Other film/TV agencies (15) | 2,377,149 | 71.4% | 567,203 | 112,423 |

- **Network.** Almost all of the network gap was reviewed and left out for lack of evidence (96.4% of the reviewed-out volume).
- **Agencies.** The lower recall is mostly by design: these agencies also protect games and software (31% of the reviewed-out volume), books (27%), sports (14%) and webtoons (10%), which the rule excludes. Only 6.5% was left out for lack of evidence.
- **Agency set.** Four of the 15 agency accounts are owners filing for themselves, so part of this set's in-scope volume is the owners' own filing.

**Read these recalls with care.** Sources (b) and (c) produced the candidates from these same reporters, at ≥500 requests per owner, so both recalls are partly circular. They measure how many large film/TV clients of these reporters the review missed. They do not measure how much film/TV piracy the scope covers in general. An earlier draft measured 89.9% on a five-name network list. That figure used a different reporter set and older keys, so it is not comparable.

### Film/TV volume left out

Some reviewed owners may be film/TV but were left out because the evidence was missing, the name was a compound credit, or the claimant could not be identified. Together they hold **550,408 requests (1.18% of the window)**, nearly all for lack of evidence. This is a bound from reviewed candidates only. Film/TV owners never caught by any source are not measured.

### What the scope consists of

| Breakdown | Share of in-scope requests |
| --- | ---: |
| Largest group: VGTRK (Russian state broadcaster) | 31.7% |
| Brands operating in the Russian market | 68.5% |
| Evidence: company knowledge / reporter evidence / web source | 86.7% / 12.3% / 1.0% |
| Caught by volume ranking or name prefix | 97.2% |
| Caught only through the network or agency reporters | 2.7% |
| Caught only by the seed list | 0.07% |

The scope is concentrated: one group holds almost a third of the volume, and the Russian market more than two thirds. Results will therefore be reported per group and with VGTRK excluded as well. Only 2.7% of the in-scope volume depends on reporter-based sources, which limits how far reporter selection can bias the audit model (section 9).

### Threshold sensitivity

Lowering the volume-ranking threshold from 5,000 to 2,000 requests would add 2,413 owner keys holding 6,545,188 requests. They are not reviewed, and most would fall outside the rule, so the scope share could rise by **at most 13.99 percentage points** under that threshold. The real change would be much smaller, but it is not measured.

## 7. Ownership at the end of the data

Owners are grouped by their corporate owner on **2026-10-24**, the last day of the data, with no exceptions. Media companies change hands. Grouping by today's owner would put requests from before a sale under a buyer that did not yet exist as their owner. Grouping by each request's own date would split one catalogue's series at every sale. One fixed date keeps each group's history in one piece. The operational differences, such as which agency filed or which market a brand serves, stay at brand level and in the reporter features.

### Method

The method was written down before any ownership search:

- **Owner.** The controlling corporate parent on 2026-10-24, through a majority stake or consolidated control. The chain stops at the top operating media or corporate group, never at a state, fund, family or individual. Banks, insurers and investment holdings count as investors, so the group stops at the operating company below them.
- **Telecom and technology parents.** A TV or streaming unit is grouped under its parent company. The brand keeps the unit's name.
- **Edge cases.** A minority stake does not move a group. A company in insolvency or court restructuring keeps its group until control actually transfers, and a pending sale does not move it. A compound name goes to the group of the first company it names.
- **Sources.** Company filings and annual reports, regulators or major press, each dated on or before 2026-10-24. Encyclopedia pages served only as pointers.
- **Order.** Owners without a group first, then the 50 largest named groups by volume (93.8% of in-scope volume), then a lighter screen of the remaining 135 small groups.

### Result

| Status | Meaning | Groups | Share of in-scope requests |
| --- | --- | ---: | ---: |
| Verified | A primary source or major press states ownership or control | 49 | 60.3% |
| Partial | The source implies group membership but not control, or no parent was found in one search | 46 | 33.6% |
| Screened | Checked only for membership in an existing group or a telecom/tech parent; owner not sourced | 131 | 5.8% |
| Open | Identity not resolved | 2 | 0.4% |

The check changed 53 of the 228 groups: 35 new groups formed when ungrouped owners were placed, 11 were renamed to their parent, and 7 merged into another group. The largest change follows Paramount Skydance's acquisition of Warner Bros. Discovery, which closed on 2026-10-06, before the data end: the Paramount and Warner brands now form one group.

A third of the volume is only `partial`, mostly because ownership of private Russian media companies is not published. These 13 groups were placed by their own filing name because no parent was found in one search: Mauris Film, Prior, A Serial, R1, Media Hub Distribution, M3, CM, Kinofon, Diamond Films, Desyat Bukv, Volgafilm, Gemplex and Mars Media. Two of them, R1 and CM, are filed only through one agency, StarMedia, which also produces content. They may be its labels, which would be a missed merge. Without evidence they remain separate groups.

Sources for the changes that move the most volume:

- Paramount Skydance and Warner Bros. Discovery, closed 2026-10-06: [Forbes](https://www.forbes.com/sites/alisondurkee/2026/10/06/skydance-is-here-paramount-closes-110-billion-warner-bros-merger-after-months-of-controversy/), [Bloomberg](https://www.bloomberg.com/news/articles/2026-10-06/paramount-closes-warner-bros-merger-in-historic-hollywood-deal)
- Sky Deutschland to RTL Group (Bertelsmann), closed 2026-06-01: [Broadband TV News](https://www.broadbandtvnews.com/2026/06/01/rtl-group-closes-acquisition-of-sky-deutschland/)
- Russian holdings: Gazprom-Media ([Kommersant, 2025-04-16](https://kommersant.ru/doc/7658392)), National Media Group ([State Media Monitor](https://statemediamonitor.com/2025/08/national-media-group/))

The full record, one row per group with its source and access date, is kept privately with the review.

### Changes after the data end

Some deals were pending on 2026-10-24 and are not applied:

- JTBC was in court-led restructuring with no closed sale.
- The Tving–Wavve merger had not closed.
- Sky's agreed acquisition of ITV Media & Entertainment is expected to close in the second half of 2027.

A later snapshot may need these groups updated.

## 8. Known limitations

### Source data

- **"No action" is Google's decision.** It is not a judgment that the request was wrong. Google lists its own reasons, such as insufficient information or the URL not containing the work. A model trained on this label predicts Google's decision, not the merit of a request.
- **The no-action URL list is incomplete.** `urls_no_action` holds 456.6 M URLs, 48% of the 952.9 M no-action count. Coverage varies by year: 8% in 2013, 62% in 2017, 93–97% in 2023–2025, 54% in 2026. The cause is not documented. The audit model therefore works on request × domain, where both outcomes are fully counted, and takes no features from this file.
- **Page total and bulk total differ.** On 2026-10-30 the Transparency Report page showed 17,917,574,286 URLs since 2011-03-10. The four counters in this snapshot sum to about 20.90 billion. The page is newer yet about 3 billion smaller; the cause was not investigated. Models use row-level counters, which agree exactly between the two tables (section 1), so the label is not affected.
- **Growth in 2026 is unexplained.** The first ten months of 2026 exceed all of 2025 by 54% (section 3). The data cannot separate real growth from a change in filing or reporting.
- **`owner_id` splits names.** The cause is not documented (section 5). It does not affect grouping, which uses names.
- **Request order.** `request_id` is not chronological, so all time splits use the request timestamp.
- **Unused columns.** `lumen_url` links to third-party notice pages that are not used. `from_abuser` is always false (section 1).

### Names and normalization

- **Untrusted free text.** Owner and reporter names are typed by senders. They contain script-injection attempts, HTML entities, and invisible or text-direction characters. Normalization decodes them strictly. The demo will escape every name before display, and this card does not reproduce any payload.
- **Names without a key.** 49 names, 2,058 window requests (section 3).
- **Incomplete transliteration.** 35 keys keep some Cyrillic letters, from Kazakh, Mongolian and Serbian, that the transliteration table does not cover. Normalizing such a key again can change it. They hold 556 window requests (0.0012%), none in scope.
- **Legal forms not removed.** Some forms stay in keys: spelled-out Russian forms and suffixes such as `s.l.u`, `sp z o.o`, `pvt`, `llp`. In scope, the affected variants are joined by hand in the alias file. Outside the scope, one company may still appear under more than one key.
- **Acronyms.** Pairs such as `tvb` and `television broadcasts` share no word. The variant check could not join them, and only the hand review could.
- **Reporter variants below 100 requests** are not in the reporter alias file, so small variants of an agency count as separate reporters.

### Scope and grouping

- **Compound credits** name companies from different groups in one string. They are left out of scope (22 keys) and fall inside the 1.18% bound in section 6.
- **Possible missed merges.** R1 and CM (section 7).
- **Unresolved identities.** Two in-scope groups are `open` (0.4% of in-scope volume). One key, `mbc`, is shared by two different broadcasters. `canal 13` could belong to channels in two countries. Both stay in scope under separate placeholder groups; the forecasting task decides their treatment in its pre-registration.
- **Heuristic labels.** `region`, `market` and `content_type` are judgment labels, not measured attributes (section 5).
- **Ownership certainty.** 40% of in-scope volume rests on `partial`, `screened` or `open` ownership (section 7).

## 9. Intended use

### What this data is for

- Training and evaluating the PiracyLens prototype's audit, forecasting and domain-clustering models on film/TV takedowns, within the window and scope above.
- Reporting how the takedown process behaves in aggregate: volumes, Google's decisions, and the domains named.

### What it is not for

- **Judging individual parties.** The data records what was filed and what Google decided. It cannot show whether a request was justified, whether a site was infringing, or who actually stood behind an account. No result from this project should be read as a finding about a named owner, agency or site.
- **Identifying people.** Names of private individuals are never shown or analysed one by one (section 2).
- **Measuring the piracy market.** Takedown volume reflects how much rights holders file, not how much piracy exists. The project makes no market-size claims. Business value is reported only through metrics measured on this data.

### How far results generalize

Results describe **this film/TV scope**, not film/TV piracy in general:

- **Concentration.** The scope is concentrated: VGTRK holds 31.7% of in-scope requests and the Russian market 68.5% (section 6). Results will be reported per group and with VGTRK excluded as well.
- **Reporter-based selection.** Some owners entered the scope only through the reporters who filed for them, and the audit model also uses the reporter as a feature. That could flatter its results. The effect is small (2.7% of in-scope volume, section 6) and is tested twice: the model is also evaluated only on owners found by volume ranking or name prefix, and a second model is trained without reporter features. Both results are reported next to the main one.
- **Window.** All results come from 2021-01-01 to 2026-10-24. Two planned checks test sensitivity to the start year (section 3).
- **Snapshot.** Every figure belongs to the snapshot in section 1. A newer export will give different numbers.

### Versioning

This card describes scope version `scope-v1`, fixed by a git tag on the commit that holds `configs/owner_aliases.yaml` and this card. After the tag:

- A bug fix to the scope is recorded in a change log with its reason.
- Any other change, such as adding owners later, does not alter the main results. It is reported as a separate sensitivity experiment.
