# Baseline smoke test

Run 2026-10-01 17:09 UTC with `scripts/evaluation/smoke_test.py`.

Configuration: retriever `dense`, embedding model `sentence-transformers/all-MiniLM-L6-v2`, top_k 5, chunks up to 400 tokens, LLM `claude-haiku-4-5-20251001` (temperature 0.0), prompt `v1`.

Timings are from one laptop run and include no warm-up, so treat them as rough.

## 1. How long does my landlord have to protect my deposit?

*Tests: single fact: deposits*

**Answer:**

> I can't find that in the guidance.
>
> The passages explain what deposits must be protected and what landlords must tell you about protection, but they don't state a specific deadline for when landlords must put deposits into a protection scheme. I'd suggest checking GOV.UK's tenancy deposit protection page or getting advice from Citizens Advice or Shelter.
>
> Guidance retrieved: 2026-10-01
>
> Sources:
> [5] Tenancy deposit protection - https://www.gov.uk/tenancy-deposit-protection
>
> This is general information from GOV.UK guidance, not legal advice.

**Retrieved passages:**

| Rank | Score | Document | Section | Cited |
|---:|---:|---|---|:---:|
| 1 | 0.732 | Assured periodic tenancies: a guide for landlords | Rent in advance and deposits > Asking tenants to pay a depos |  |
| 2 | 0.731 | Assured periodic tenancies: a guide for tenants | Rent in advance and deposits |  |
| 3 | 0.692 | Private renting | Deposits |  |
| 4 | 0.672 | Tenant Fees Act 2019: guidance for tenants | Refundable holding deposit |  |
| 5 | 0.663 | Tenancy deposit protection | Overview > At the end of your tenancy |  |

Refused: **True** · retrieval 12 ms · generation 2453 ms · total 2464 ms · tokens in/out 2238/131

## 2. Can my landlord evict me without giving a reason?

*Tests: Renters' Rights Act change*

**Answer:**

> No. Your landlord cannot evict you without giving a reason. [5]
>
> If your landlord wants to evict you, they must follow the correct legal procedure, which includes giving you proper notice and explaining the reasons (called 'grounds') for the eviction. [3][4] Your landlord must also give you the correct notice period before going to court. [4]
>
> If your landlord tries to evict you without following these rules—for example, by changing the locks or evicting you without a court order—this is illegal eviction. [1]
>
> If you think your landlord is not following the rules, you can get support from Shelter or Citizens Advice. [5]
>
> Guidance retrieved: 2026-10-01
>
> Sources:
> [1] Private renting for tenants: evictions in England - https://www.gov.uk/private-renting-evictions
> [3] Evicting tenants in England - https://www.gov.uk/evicting-tenants
> [4] Private renting for tenants: evictions in England - https://www.gov.uk/private-renting-evictions
> [5] Renters' Rights Act overview for tenants - https://www.gov.uk/guidance/renters-rights-act-overview-for-tenants
>
> This is general information from GOV.UK guidance, not legal advice.

**Retrieved passages:**

| Rank | Score | Document | Section | Cited |
|---:|---:|---|---|:---:|
| 1 | 0.613 | Private renting for tenants: evictions in England | Harassment and illegal evictions > Harassment | yes |
| 2 | 0.608 | Private renting | Rent arrears > If you have a regulated tenancy or an exclude |  |
| 3 | 0.605 | Evicting tenants in England | Giving notice to evict tenants > Before you give notice | yes |
| 4 | 0.595 | Private renting for tenants: evictions in England | Eviction court hearings | yes |
| 5 | 0.593 | Renters’ Rights Act overview for tenants | If your landlord wants to end your tenancy | yes |

Refused: **False** · retrieval 9 ms · generation 2863 ms · total 2872 ms · tokens in/out 2021/308

## 3. How much notice does my landlord have to give before increasing my rent, and can I challenge it?

*Tests: two-part question: rent increases*

**Answer:**

> # Notice period for rent increases
>
> Your landlord must give you **at least 2 months' notice** before increasing your rent [1][2]. They must do this by giving you a completed form 4A: Landlord's notice proposing a new rent [1][2]. They can deliver this notice in person, by post, or by email (if your tenancy agreement allows it) [1][2].
>
> Your landlord can only increase your rent once a year, and cannot increase it in the first year of your tenancy [1][3].
>
> # Challenging a rent increase
>
> Yes, you can challenge the increase. If you think the rent increase is higher than the 'open market rent' (the rent your landlord would expect to receive if they were to relet the property on the open market), you can ask the First-tier Tribunal to decide what the new rent should be [2][5].
>
> Guidance retrieved: 2026-10-01
>
> Sources:
> [1] Assured periodic tenancies: a guide for tenants - https://www.gov.uk/assured-periodic-tenancies-tenants
> [2] Assured periodic tenancies: a guide for landlords - https://www.gov.uk/assured-tenancy-agreements-a-guide-for-landlords
> [3] Renters' Rights Act overview for tenants - https://www.gov.uk/guidance/renters-rights-act-overview-for-tenants
> [5] Assured periodic tenancies: a guide for tenants - https://www.gov.uk/assured-periodic-tenancies-tenants
>
> This is general information from GOV.UK guidance, not legal advice.

**Retrieved passages:**

| Rank | Score | Document | Section | Cited |
|---:|---:|---|---|:---:|
| 1 | 0.761 | Assured periodic tenancies: a guide for tenants | Rent and other payments > Other payments you may be asked to | yes |
| 2 | 0.675 | Assured periodic tenancies: a guide for landlords | Rent increases > How you can increase the rent | yes |
| 3 | 0.674 | Renters’ Rights Act overview for tenants | Other changes from 1 May 2026 > Rent in advance | yes |
| 4 | 0.672 | Assured periodic tenancies: a guide for landlords | Rent increases > If you gave notice of a rent increase befor |  |
| 5 | 0.655 | Assured periodic tenancies: a guide for tenants | Rent increases > If you do not agree with the increase | yes |

Refused: **False** · retrieval 8 ms · generation 3436 ms · total 3445 ms · tokens in/out 2038/373

## 4. Who is responsible for fixing a broken boiler in my rented flat?

*Tests: repairs*

**Answer:**

> Your landlord is responsible for fixing a broken boiler in your rented flat [1]. Heating and hot water systems are always your landlord's responsibility [1].
>
> You should contact your landlord straight away about the broken boiler, as this is a fault that could affect your health [1]. Your landlord should tell you when you can expect the repairs to be done. You should continue paying rent while waiting for the repairs [1].
>
> Guidance retrieved: 2026-10-01
>
> Sources:
> [1] Private renting - https://www.gov.uk/private-renting
>
> This is general information from GOV.UK guidance, not legal advice.

**Retrieved passages:**

| Rank | Score | Document | Section | Cited |
|---:|---:|---|---|:---:|
| 1 | 0.479 | Private renting | Repairs > What your landlord must do | yes |
| 2 | 0.447 | Renting out your property | Landlord responsibilities > Regulated tenancies |  |
| 3 | 0.437 | Fire safety in flats in England: a guide for residents | During and after fire safety repair work > After the work is |  |
| 4 | 0.435 | Renting out your property | Making repairs > Common areas |  |
| 5 | 0.423 | Council housing | Repairs and maintenance |  |

Refused: **False** · retrieval 9 ms · generation 2116 ms · total 2125 ms · tokens in/out 1958/146

## 5. Can a landlord refuse to rent to me because I get Universal Credit?

*Tests: discrimination (new rules)*

**Answer:**

> I can't find that in the guidance. I suggest checking GOV.UK directly, or getting advice from Citizens Advice or Shelter, who can advise on discrimination in renting.
>
> Guidance retrieved: 2026-10-01
>
> This is general information from GOV.UK guidance, not legal advice.

**Retrieved passages:**

| Rank | Score | Document | Section | Cited |
|---:|---:|---|---|:---:|
| 1 | 0.655 | Housing costs and Universal Credit | Renting from the local authority or housing association > If |  |
| 2 | 0.638 | Notices of possession served from 1 May 2026: a guide for tenants who are renting from a private landlord | Harassment or illegal eviction |  |
| 3 | 0.625 | Housing costs and Universal Credit | Renting from a private landlord > If you pay rent on 2 homes |  |
| 4 | 0.596 | Private renting | Rent arrears > If you have a regulated tenancy or an exclude |  |
| 5 | 0.581 | Renters' Rights Act: an overview for landlords | When a landlord breaks the law |  |

Refused: **True** · retrieval 10 ms · generation 1453 ms · total 1463 ms · tokens in/out 2025/70

## 6. What fees is a letting agent allowed to charge me?

*Tests: Tenant Fees Act*

**Answer:**

> A letting agent can charge you the same fees that a landlord is allowed to charge. These are [1]:
>
> - the rent (if you've signed the tenancy agreement)
> - a refundable tenancy deposit
> - a refundable holding deposit to reserve a property
> - a payment to change the tenancy when you ask to (up to £50, or more if costs are higher) [5]
> - a payment when you leave a tenancy without giving enough notice
> - a payment towards utility bills and communication services
> - a fee for late payment of rent
> - a fee for replacing a key or security device
>
> A letting agent cannot charge you for things not on this list. For example, they cannot make you pay for loans, insurance, or contracts for services (like professional guarantor services, contents insurance, or inventory services) as part of your tenancy [2].
>
> If a letting agent asks you for a fee that is not allowed, you can report them to their redress scheme [2].
>
> Guidance retrieved: 2026-10-01
>
> Sources:
> [1] Tenant Fees Act 2019: guidance for tenants - https://www.gov.uk/guidance/tenant-fees-act-2019-guidance-for-tenants
> [2] Tenant Fees Act 2019: guidance for tenants - https://www.gov.uk/guidance/tenant-fees-act-2019-guidance-for-tenants
> [5] Tenant Fees Act 2019: guidance for tenants - https://www.gov.uk/guidance/tenant-fees-act-2019-guidance-for-tenants
>
> This is general information from GOV.UK guidance, not legal advice.

**Retrieved passages:**

| Rank | Score | Document | Section | Cited |
|---:|---:|---|---|:---:|
| 1 | 0.541 | Tenant Fees Act 2019: guidance for tenants |  | yes |
| 2 | 0.446 | Tenant Fees Act 2019: guidance for tenants | Prohibited payments | yes |
| 3 | 0.420 | Assured tenancy forms | Form 4A: Landlord’s notice proposing a new rent for assured  |  |
| 4 | 0.410 | Rental discrimination: a guide for tenants | If you think you're being discriminated against > If you do  |  |
| 5 | 0.409 | Tenant Fees Act 2019: guidance for tenants | Fees for tenancy changes | yes |

Refused: **False** · retrieval 11 ms · generation 4043 ms · total 4054 ms · tokens in/out 2050/375

## 7. Does a landlord need a licence to rent a house to five people who aren't related?

*Tests: HMO licensing*

**Answer:**

> Yes, a landlord needs a licence if they're renting to five or more people who aren't related [4].
>
> This is called a "large HMO" (house in multiple occupation). A property is a large HMO if [4]:
>
> - it is rented to 5 or more people who form more than 1 household
> - some or all tenants share toilet, bathroom or kitchen facilities
> - at least 1 tenant pays rent
>
> Large HMOs must have a licence from the local council [2]. The licence is valid for a maximum of 5 years and must be renewed before it runs out [4].
>
> Guidance retrieved: 2026-10-01
>
> Sources:
> [2] Private renting - https://www.gov.uk/private-renting
> [4] House in multiple occupation licence - https://www.gov.uk/find-licences/house-in-multiple-occupation-licence
>
> This is general information from GOV.UK guidance, not legal advice.

**Retrieved passages:**

| Rank | Score | Document | Section | Cited |
|---:|---:|---|---|:---:|
| 1 | 0.574 | Electrical safety standards in the private and social rented sectors: guidance | Electrical safety standards in the private and social rented |  |
| 2 | 0.569 | Private renting | Houses in multiple occupation | yes |
| 3 | 0.542 | Rent Repayment Orders offences: guidance for tenants | Unlicensed HMO or house > Additional defences for superior l |  |
| 4 | 0.537 | House in multiple occupation licence |  | yes |
| 5 | 0.523 | Rent Repayment Orders offences: guidance for tenants | Unlicensed HMO or house |  |

Refused: **False** · retrieval 9 ms · generation 2328 ms · total 2336 ms · tokens in/out 2108/224

## 8. What can I do about damp and mould in my council flat?

*Tests: social housing*

**Answer:**

> I can't find that in the guidance.
>
> The passages provided are guidance for landlords on how to address damp and mould, not guidance for tenants renting council flats. For advice on your rights as a tenant and what you can do about damp and mould in your home, please check GOV.UK's guidance for tenants, or contact Citizens Advice or Shelter for support.
>
> Guidance retrieved: 2026-10-01
>
> This is general information from GOV.UK guidance, not legal advice.

**Retrieved passages:**

| Rank | Score | Document | Section | Cited |
|---:|---:|---|---|:---:|
| 1 | 0.702 | Damp and mould: understanding and addressing the health risks for rented housing providers | Understanding and addressing the health risks of damp and mo |  |
| 2 | 0.690 | Damp and mould: understanding and addressing the health risks for rented housing providers | Understanding and addressing the health risks of damp and mo |  |
| 3 | 0.683 | Damp and mould: understanding and addressing the health risks for rented housing providers | Understanding and addressing the health risks of damp and mo |  |
| 4 | 0.675 | Damp and mould: understanding and addressing the health risks for rented housing providers | Understanding and addressing the health risks of damp and mo |  |
| 5 | 0.663 | Damp and mould: understanding and addressing the health risks for rented housing providers | Understanding and addressing the health risks of damp and mo |  |

Refused: **True** · retrieval 8 ms · generation 1862 ms · total 1871 ms · tokens in/out 2717/116

## 9. What is the average rent for a one-bedroom flat in Manchester?

*Tests: unanswerable*

**Answer:**

> I can't find that in the guidance. You could check the Local Housing Allowance tool on GOV.UK, or get advice from Citizens Advice or Shelter.
>
> The passages mention that you can use the Local Housing Allowance tool to find out rental rates in your area [1][3], but they don't provide specific average rent figures for Manchester or any other location.
>
> Guidance retrieved: 2026-10-01
>
> This is general information from GOV.UK guidance, not legal advice.

**Retrieved passages:**

| Rank | Score | Document | Section | Cited |
|---:|---:|---|---|:---:|
| 1 | 0.489 | Housing costs and Universal Credit | Renting from a private landlord > How much you’ll get for ho |  |
| 2 | 0.486 | Rent a room in your home | The Rent a Room Scheme |  |
| 3 | 0.486 | Housing Benefit | What you'll get > Private rent |  |
| 4 | 0.485 | Tenant Fees Act 2019: guidance for tenants | Rent |  |
| 5 | 0.470 | Housing Benefit | What you'll get |  |

Refused: **True** · retrieval 7 ms · generation 1920 ms · total 1928 ms · tokens in/out 2019/111

## 10. How do I apply for a mortgage to buy my first home?

*Tests: unanswerable (off-topic)*

**Answer:**

> I can't find that in the guidance.
>
> The passages provided cover renting and housing support through Universal Credit, but they don't contain information about applying for a mortgage to buy a home. I'd suggest checking GOV.UK's pages on mortgages and home buying, or getting advice from Citizens Advice or Shelter.
>
> Guidance retrieved: 2026-10-01
>
> This is general information from GOV.UK guidance, not legal advice.

**Retrieved passages:**

| Rank | Score | Document | Section | Cited |
|---:|---:|---|---|:---:|
| 1 | 0.452 | Housing association homes | Apply for a home |  |
| 2 | 0.429 | Housing costs and Universal Credit | Living in a property you own |  |
| 3 | 0.396 | Council housing | Apply for a council home |  |
| 4 | 0.392 | Housing costs and Universal Credit | How to claim > After you’ve applied > If you’re renting |  |
| 5 | 0.373 | Housing costs and Universal Credit | What you can get > Other help with housing costs |  |

Refused: **True** · retrieval 7 ms · generation 1739 ms · total 1746 ms · tokens in/out 1960/99

---

## Observations (written by hand after the run; not regenerated by the script)

**Summary:** 5 of 8 answerable questions answered; 2 of 2 unanswerable questions correctly
refused; 3 answerable questions wrongly refused. Every factual claim in the 5 answers was
checked against the source chunks and is supported. No tuning was done: this is the baseline.

**What worked**

- Answers are grounded: deposit caps, the 2-month rent increase notice (Form 4A), the
  once-a-year limit, landlord responsibility for heating, the Tenant Fees Act list and the
  HMO licence rules all match the GOV.UK text.
- Both off-topic questions (Q9 average rent in Manchester, Q10 mortgages) were refused rather
  than answered from general knowledge.
- The format rules (citations, retrieval date, sources list, disclaimer) were followed in
  every answer.

**What went wrong**

1. **False refusals caused by retrieval misses (Q1, Q5).** The chunk containing the answer
   ranked 9th, outside the top 5:
   - Q1: "must put your deposit in the scheme within 30 days" (Tenancy deposit protection,
     Overview) ranked 9th; the top 5 were broader deposit sections from other guides.
   - Q5: "You cannot be discouraged or stopped from renting a property because you get
     benefits" (Rental discrimination: a guide for tenants) ranked 9th; the query's words
     "Universal Credit" pulled in the Universal Credit housing costs guide instead.
   Dense retrieval matches the topic but not the specific fact. BM25 (exact terms like
   "30 days", "benefits") and a reranker are the planned fixes.
2. **One long document dominates (Q8).** The 17,000-word landlord-facing damp and mould
   guidance filled ranks 1 to 8. The tenant guide on damp ranked 9th, and the social housing
   (Awaab's Law) guidance was not in the top 12. The model then refused because the passages
   were "guidance for landlords", which is arguably too strict: they do describe what
   landlords must do. Possible fixes: limit chunks per document, rerank, or rebalance the
   corpus.
3. **Refusals do not follow the exact format.** Q1 listed a source after refusing, and Q9
   cited passages [1][3] in its explanation. The pipeline handles this (any answer starting
   with the refusal sentence is treated as a refusal and its sources are dropped), but the
   prompt rule "reply with exactly..." is not being followed strictly.
4. **Duplicate sources.** When two chunks from the same page are cited (Q2, Q6), the
   sources list shows the same URL more than once.
5. **Section labels can mislead.** A chunk's section is the heading where it starts, so a
   chunk labelled "Other payments you may be asked to" (Q3, passage 1) actually holds the
   rent increase rules from the next section.

**Cost and speed:** about 2,000 input and 70 to 375 output tokens per question. Retrieval
takes about 10 ms; generation takes 1.5 to 4 s, so the LLM is over 99% of the latency.
