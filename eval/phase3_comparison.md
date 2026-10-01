# Retrieval comparison

Generated 2026-10-01 17:27 UTC by `scripts/compare_retrieval.py` (no LLM calls). Top 5 chunks per setup; candidate pool 20, RRF k = 60, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` on cpu. **(new)** marks a chunk that is not in the dense top 5. Scores are on different scales per setup (cosine, BM25, RRF, cross-encoder logit) and are only comparable within a column. Timings are the median of 3 warm runs on a laptop CPU, in milliseconds.

### How long does my landlord have to protect my deposit?

| Rank | dense | bm25 | hybrid | hybrid+rerank |
|---:|---|---|---|---|
| 1 | Assured periodic tenancies: a guide for land… / Asking tenants to pay a deposit (0.73) | Tenancy deposit protection / At the end of your tenancy (13.12) | Tenancy deposit protection / At the end of your tenancy (0.03) | Tenancy deposit protection / At the end of your tenancy (7.58) |
| 2 | Assured periodic tenancies: a guide for tena… / Rent in advance and deposits (0.73) | Tenancy deposit protection / If your landlord does not protect … (12.85) **(new)** | Private renting / Deposits (0.03) | Tenancy deposit protection / What happens next (6.79) **(new)** |
| 3 | Private renting / Deposits (0.69) | Deposit protection schemes and landlords / If you do not protect your tenants… (11.94) **(new)** | Deposit protection schemes and landlords / If you do not protect your tenants… (0.03) **(new)** | Private renting / Getting your deposit back (6.22) **(new)** |
| 4 | Tenant Fees Act 2019: guidance for tenants / Refundable holding deposit (0.67) | Tenancy deposit protection / What happens next (11.61) **(new)** | Tenancy deposit protection / What happens next (0.03) **(new)** | Tenant Fees Act 2019: guidance for tenants / Refundable holding deposit (6.10) |
| 5 | Tenancy deposit protection / At the end of your tenancy (0.66) | Deposit protection schemes and landlords / At the end of the tenancy (11.48) **(new)** | Assured periodic tenancies: a guide for tena… / Rent in advance and deposits (0.03) | Private renting / Deposits (5.75) |
| ms | **8 ms** | **0 ms** | **8 ms** (dense 7 + bm25 1 + fusion 0) | **681 ms** (dense 9 + bm25 1 + fusion 0 + rerank 671) |

Chunk IDs: dense: assured-tenancy-agreements-a-guide-for-landlords-006-cea51aed, assured-periodic-tenancies-tenants-004-4ad46185, private-renting-012-cdc466bc, tenant-fees-act-2019-guidance-for-tenants-002-20a1c4d9, tenancy-deposit-protection-001-01726dfa; bm25: tenancy-deposit-protection-001-01726dfa, tenancy-deposit-protection-002-7ededbcb, deposit-protection-schemes-and-landlords-002-26d3f6a4, tenancy-deposit-protection-003-61172024, deposit-protection-schemes-and-landlords-001-be626a5c; hybrid: tenancy-deposit-protection-001-01726dfa, private-renting-012-cdc466bc, deposit-protection-schemes-and-landlords-002-26d3f6a4, tenancy-deposit-protection-003-61172024, assured-periodic-tenancies-tenants-004-4ad46185; hybrid+rerank: tenancy-deposit-protection-001-01726dfa, tenancy-deposit-protection-003-61172024, private-renting-013-90122dc7, tenant-fees-act-2019-guidance-for-tenants-002-20a1c4d9, private-renting-012-cdc466bc

### Can my landlord evict me without giving a reason?

| Rank | dense | bm25 | hybrid | hybrid+rerank |
|---:|---|---|---|---|
| 1 | Private renting for tenants: evictions in En… / Harassment (0.61) | Renters’ Rights Act overview for tenants / Ending your tenancy (13.80) **(new)** | Evicting tenants in England / Before you give notice (0.03) | Assured periodic tenancies: a guide for tena… / If you do not leave the property (4.77) **(new)** |
| 2 | Private renting / If you have a regulated tenancy or… (0.61) | Rent Repayment Orders: guidance for tenants / Harassment, aggression or intimida… (12.99) **(new)** | Guide to the Renters’ Rights Act / Table 1: Grounds for possession (0.03) **(new)** | Private renting for tenants: evictions in En… / Rules your landlord must follow (3.84) **(new)** |
| 3 | Evicting tenants in England / Before you give notice (0.60) | Private renting for tenants: evictions in En… / Rules your landlord must follow (12.37) **(new)** | Private renting for tenants: evictions in En… / Rules your landlord must follow (0.03) **(new)** | Private renting for tenants: evictions in En… / Harassment (3.69) |
| 4 | Private renting for tenants: evictions in En… / Eviction court hearings (0.59) | Evicting tenants in England / Before you give notice (12.03) | Renters’ Rights Act overview for tenants / Ending your tenancy (0.03) **(new)** | Notices of possession served before 1 May 20… / What to do if you were served with… (3.16) **(new)** |
| 5 | Renters’ Rights Act overview for tenants / If your landlord wants to end your… (0.59) | Notices of possession served before 1 May 20… / What to do if you were served with… (12.01) **(new)** | Assured periodic tenancies: a guide for tena… / If you do not leave the property (0.03) **(new)** | Notices of possession served before 1 May 20… / Notices of possession served befor… (3.04) **(new)** |
| ms | **8 ms** | **0 ms** | **8 ms** (dense 6 + bm25 1 + fusion 0) | **748 ms** (dense 7 + bm25 1 + fusion 0 + rerank 740) |

Chunk IDs: dense: private-renting-evictions-011-a5909d8d, private-renting-011-dd80feb3, evicting-tenants-002-d7580eb3, private-renting-evictions-004-9896c26e, renters-rights-act-overview-for-tenants-003-3124e9a4; bm25: renters-rights-act-overview-for-tenants-002-03be3728, rent-repayment-orders-guidance-for-tenants-011-9e32630a, private-renting-evictions-000-b1aa97b5, evicting-tenants-002-d7580eb3, notices-of-possession-served-before-1-may-2026-a-guide-for-tenants-who-are-renting-from-a-private-landlord-002-8987cf3a; hybrid: evicting-tenants-002-d7580eb3, guide-to-the-renters-rights-act-015-e1cfdea7, private-renting-evictions-000-b1aa97b5, renters-rights-act-overview-for-tenants-002-03be3728, assured-periodic-tenancies-tenants-017-42dd98f7; hybrid+rerank: assured-periodic-tenancies-tenants-017-42dd98f7, private-renting-evictions-000-b1aa97b5, private-renting-evictions-011-a5909d8d, notices-of-possession-served-before-1-may-2026-a-guide-for-tenants-who-are-renting-from-a-private-landlord-002-8987cf3a, notices-of-possession-served-before-1-may-2026-a-guide-for-tenants-who-are-renting-from-a-private-landlord-001-04af01ca

### How much notice does my landlord have to give before increasing my rent, and can I challenge it?

| Rank | dense | bm25 | hybrid | hybrid+rerank |
|---:|---|---|---|---|
| 1 | Assured periodic tenancies: a guide for tena… / Other payments you may be asked to… (0.76) | Apply for an open market rent determination / If you think the landlord’s notice… (17.83) **(new)** | Apply for an open market rent determination / If you think the landlord’s notice… (0.03) **(new)** | Apply for an open market rent determination / If you think the landlord’s notice… (9.02) **(new)** |
| 2 | Assured periodic tenancies: a guide for land… / How you can increase the rent (0.67) | Assured periodic tenancies: a guide for tena… / Damage from pets (13.47) **(new)** | Assured periodic tenancies: a guide for land… / How you can increase the rent (0.03) | Renters' Rights Act: an overview for landlor… / Increasing rent (6.99) **(new)** |
| 3 | Renters’ Rights Act overview for tenants / Rent in advance (0.67) | Renters' Rights Act: an overview for landlor… / Increasing rent (13.30) **(new)** | Renters’ Rights Act overview for tenants / Rent in advance (0.03) | Assured periodic tenancies: a guide for land… / If you gave notice of a rent incre… (6.52) |
| 4 | Assured periodic tenancies: a guide for land… / If you gave notice of a rent incre… (0.67) | Apply for an open market rent determination / Overview (13.19) **(new)** | Apply for an open market rent determination / Overview (0.03) **(new)** | Assured periodic tenancies: a guide for land… / How you can increase the rent (6.26) |
| 5 | Assured periodic tenancies: a guide for tena… / If you do not agree with the incre… (0.65) | Assured periodic tenancies: a guide for land… / If the tenancy began before 1 May … (12.81) **(new)** | Assured periodic tenancies: a guide for land… / If you gave notice of a rent incre… (0.03) | Assured periodic tenancies: a guide for tena… / If you do not agree with the incre… (6.08) |
| ms | **8 ms** | **1 ms** | **10 ms** (dense 8 + bm25 2 + fusion 0) | **764 ms** (dense 8 + bm25 2 + fusion 0 + rerank 754) |

Chunk IDs: dense: assured-periodic-tenancies-tenants-007-e6f5972b, assured-tenancy-agreements-a-guide-for-landlords-009-50696406, renters-rights-act-overview-for-tenants-001-7e5c1e5e, assured-tenancy-agreements-a-guide-for-landlords-010-c4d98cb7, assured-periodic-tenancies-tenants-008-4c040cab; bm25: apply-for-an-open-market-rent-determination-001-656a1dc6, assured-periodic-tenancies-tenants-011-f99c37af, renters-rights-act-an-overview-for-landlords-001-3c70f520, apply-for-an-open-market-rent-determination-000-1b270cdc, assured-tenancy-agreements-a-guide-for-landlords-018-81fb0746; hybrid: apply-for-an-open-market-rent-determination-001-656a1dc6, assured-tenancy-agreements-a-guide-for-landlords-009-50696406, renters-rights-act-overview-for-tenants-001-7e5c1e5e, apply-for-an-open-market-rent-determination-000-1b270cdc, assured-tenancy-agreements-a-guide-for-landlords-010-c4d98cb7; hybrid+rerank: apply-for-an-open-market-rent-determination-001-656a1dc6, renters-rights-act-an-overview-for-landlords-001-3c70f520, assured-tenancy-agreements-a-guide-for-landlords-010-c4d98cb7, assured-tenancy-agreements-a-guide-for-landlords-009-50696406, assured-periodic-tenancies-tenants-008-4c040cab

### Who is responsible for fixing a broken boiler in my rented flat?

| Rank | dense | bm25 | hybrid | hybrid+rerank |
|---:|---|---|---|---|
| 1 | Private renting / What your landlord must do (0.48) | Fire safety in flats in England: a guide for… / Get help and advice (15.63) **(new)** | Private renting / What your landlord must do (0.03) | Awaab’s Law Phase 1: Guidance for social lan… / 5 Requirement to take emergency ac… (1.30) **(new)** |
| 2 | Renting out your property / Regulated tenancies (0.45) | Awaab’s Law Phase 1: Guidance for social lan… / 3.3 An ‘emergency hazard’ (11.41) **(new)** | Fire safety in flats in England: a guide for… / Get help and advice (0.03) **(new)** | Private renting / What your landlord must do (1.08) |
| 3 | Fire safety in flats in England: a guide for… / If you have concerns after the wor… (0.44) | Fire safety in flats in England: a guide for… / What to do if your building is not… (10.60) **(new)** | Awaab’s Law Phase 1: Guidance for social lan… / 3.3 An ‘emergency hazard’ (0.03) **(new)** | Renting out your property / Common areas (0.30) |
| 4 | Renting out your property / Common areas (0.43) | Fire safety in flats in England: a guide for… / Overview (10.21) **(new)** | Council housing / Repairs and maintenance (0.03) | Awaab’s Law Phase 1: Guidance for social lan… / 2.2 Buildings or land for which th… (-1.92) **(new)** |
| 5 | Council housing / Repairs and maintenance (0.42) | Private renting / What your landlord must do (9.84) | Fire safety in flats in England: a guide for… / What to do if your building is not… (0.03) **(new)** | Fire safety in flats in England: a guide for… / Overview (-4.20) **(new)** |
| ms | **8 ms** | **1 ms** | **9 ms** (dense 8 + bm25 1 + fusion 0) | **732 ms** (dense 10 + bm25 1 + fusion 0 + rerank 722) |

Chunk IDs: dense: private-renting-007-0ef7722d, renting-out-a-property-003-02c87e3d, fire-safety-flats-england-residents-010-50ffd931, renting-out-a-property-004-6fc7e721, council-housing-005-41fab045; bm25: fire-safety-flats-england-residents-003-868ee4e5, awaabs-law-guidance-for-social-landlords-019-3678d221, fire-safety-flats-england-residents-004-f384d83a, fire-safety-flats-england-residents-000-a2e5ec05, private-renting-007-0ef7722d; hybrid: private-renting-007-0ef7722d, fire-safety-flats-england-residents-003-868ee4e5, awaabs-law-guidance-for-social-landlords-019-3678d221, council-housing-005-41fab045, fire-safety-flats-england-residents-004-f384d83a; hybrid+rerank: awaabs-law-guidance-for-social-landlords-035-aed0bf2e, private-renting-007-0ef7722d, renting-out-a-property-004-6fc7e721, awaabs-law-guidance-for-social-landlords-009-4bbdd007, fire-safety-flats-england-residents-000-a2e5ec05

### Can a landlord refuse to rent to me because I get Universal Credit?

| Rank | dense | bm25 | hybrid | hybrid+rerank |
|---:|---|---|---|---|
| 1 | Housing costs and Universal Credit / If your landlord applies for an al… (0.65) | Rental discrimination: a guide for landlords / If your insurance contract stops c… (19.89) **(new)** | Housing costs and Universal Credit / If you pay rent on 2 homes (0.03) | Housing costs and Universal Credit / If your landlord applies for an al… (5.79) |
| 2 | Notices of possession served from 1 May 2026… / Harassment or illegal eviction (0.64) | Rental discrimination: a guide for tenants / If you’re not sure if something is… (19.84) **(new)** | Housing costs and Universal Credit / If your landlord applies for an al… (0.03) | Rental discrimination: a guide for tenants / If you’re not sure if something is… (5.69) **(new)** |
| 3 | Housing costs and Universal Credit / If you pay rent on 2 homes (0.62) | Housing costs and Universal Credit / If you pay rent on 2 homes (19.02) | Private renting / If you have a regulated tenancy or… (0.03) | Housing costs and Universal Credit / Renting from a private landlord (5.32) **(new)** |
| 4 | Private renting / If you have a regulated tenancy or… (0.60) | Housing costs and Universal Credit / What you can get (18.71) **(new)** | Notices of possession served from 1 May 2026… / Harassment or illegal eviction (0.03) | Private renting / If you have a regulated tenancy or… (5.21) |
| 5 | Renters' Rights Act: an overview for landlor… / When a landlord breaks the law (0.58) | Rent Repayment Orders: guidance for tenants / Ongoing offences (17.66) **(new)** | Housing costs and Universal Credit / If the money you get for housing d… (0.03) **(new)** | Housing costs and Universal Credit / What you can get (4.90) **(new)** |
| ms | **7 ms** | **1 ms** | **8 ms** (dense 7 + bm25 2 + fusion 0) | **717 ms** (dense 9 + bm25 1 + fusion 0 + rerank 706) |

Chunk IDs: dense: housing-and-universal-credit-010-261aefa5, notices-of-possession-served-from-1-may-2026-a-guide-for-tenants-who-are-renting-from-a-private-landlord-001-779f5e52, housing-and-universal-credit-006-f155ed88, private-renting-011-dd80feb3, renters-rights-act-an-overview-for-landlords-003-7128c0e9; bm25: rental-discrimination-landlords-003-85ea8c3f, rental-discrimination-tenants-003-e717e477, housing-and-universal-credit-006-f155ed88, housing-and-universal-credit-000-7dfa0d31, rent-repayment-orders-guidance-for-tenants-009-7b93b12d; hybrid: housing-and-universal-credit-006-f155ed88, housing-and-universal-credit-010-261aefa5, private-renting-011-dd80feb3, notices-of-possession-served-from-1-may-2026-a-guide-for-tenants-who-are-renting-from-a-private-landlord-001-779f5e52, housing-and-universal-credit-007-38d5ab7b; hybrid+rerank: housing-and-universal-credit-010-261aefa5, rental-discrimination-tenants-003-e717e477, housing-and-universal-credit-003-0376ab29, private-renting-011-dd80feb3, housing-and-universal-credit-000-7dfa0d31

### What fees is a letting agent allowed to charge me?

| Rank | dense | bm25 | hybrid | hybrid+rerank |
|---:|---|---|---|---|
| 1 | Tenant Fees Act 2019: guidance for tenants | Tenant Fees Act 2019: guidance for tenants | Tenant Fees Act 2019: guidance for tenants | Tenant Fees Act 2019: guidance for tenants |
| 2 | Tenant Fees Act 2019: guidance for tenants / Prohibited payments (0.45) | Tenant Fees Act 2019: guidance for tenants / Prohibited payments (17.07) | Tenant Fees Act 2019: guidance for tenants / Prohibited payments (0.03) | Assured periodic tenancies: a guide for land… / If the tenancy started before 1 Ma… (0.13) **(new)** |
| 3 | Assured tenancy forms / Form 4A: Landlord’s notice proposi… (0.42) | Solve a residential property dispute / Overview (11.52) **(new)** | Repossessing your privately rented property … / How to pay court fees (0.03) **(new)** | Renting out your property / Property owned by a company (-0.04) **(new)** |
| 4 | Rental discrimination: a guide for tenants / If you do not want to speak to the… (0.41) | Renting out your property / Property owned by a company (11.26) **(new)** | Rental discrimination: a guide for tenants / If you do not want to speak to the… (0.03) | Tenant Fees Act 2019: guidance for tenants / Prohibited payments (-0.25) |
| 5 | Tenant Fees Act 2019: guidance for tenants / Fees for tenancy changes (0.41) | Notices of possession served before 1 May 20… / If you were served with a section … (10.89) **(new)** | Assured periodic tenancies: a guide for tena… / After a property is advertised (0.03) **(new)** | Assured periodic tenancies: a guide for tena… / Rent in advance and deposits (-0.48) **(new)** |
| ms | **7 ms** | **1 ms** | **9 ms** (dense 8 + bm25 1 + fusion 0) | **755 ms** (dense 8 + bm25 1 + fusion 0 + rerank 746) |

Chunk IDs: dense: tenant-fees-act-2019-guidance-for-tenants-000-092427b7, tenant-fees-act-2019-guidance-for-tenants-006-c4801050, assured-tenancy-forms-006-c5c6ec0a, rental-discrimination-tenants-006-13e88089, tenant-fees-act-2019-guidance-for-tenants-004-f2817222; bm25: tenant-fees-act-2019-guidance-for-tenants-000-092427b7, tenant-fees-act-2019-guidance-for-tenants-006-c4801050, housing-tribunals-000-8337d87e, renting-out-a-property-010-d04a6683, notices-of-possession-served-before-1-may-2026-a-guide-for-tenants-who-are-renting-from-a-private-landlord-006-63e2fcbd; hybrid: tenant-fees-act-2019-guidance-for-tenants-000-092427b7, tenant-fees-act-2019-guidance-for-tenants-006-c4801050, repossessing-your-privately-rented-property-after-1-may-2026-007-5b6ffeef, rental-discrimination-tenants-006-13e88089, assured-periodic-tenancies-tenants-003-aa708cb3; hybrid+rerank: tenant-fees-act-2019-guidance-for-tenants-000-092427b7, assured-tenancy-agreements-a-guide-for-landlords-002-2058f7b9, renting-out-a-property-010-d04a6683, tenant-fees-act-2019-guidance-for-tenants-006-c4801050, assured-periodic-tenancies-tenants-004-4ad46185

### Does a landlord need a licence to rent a house to five people who aren't related?

| Rank | dense | bm25 | hybrid | hybrid+rerank |
|---:|---|---|---|---|
| 1 | Electrical safety standards in the private a… / 2. Which homes do the regulations … (0.57) | Rent Repayment Orders offences: guidance for… / Selective licensing (15.80) **(new)** | Rent Repayment Orders offences: guidance for… / Unlicensed HMO or house (0.03) | Rent Repayment Orders offences: guidance for… / Unlicensed HMO or house (4.14) |
| 2 | Private renting / Houses in multiple occupation (0.57) | Rent Repayment Orders offences: guidance for… / Unlicensed HMO or house (15.69) | Rent Repayment Orders offences: guidance for… / The landlord took all reasonably p… (0.03) | House in multiple occupation licence |
| 3 | Rent Repayment Orders offences: guidance for… / The landlord took all reasonably p… (0.54) | Rent Repayment Orders offences: guidance for… / Finding out if your landlord has c… (15.53) **(new)** | Private renting / Houses in multiple occupation (0.03) | Rent Repayment Orders offences: guidance for… / Finding out if your landlord has c… (2.10) **(new)** |
| 4 | House in multiple occupation licence | Rent Repayment Orders offences: guidance for… / The landlord took all reasonably p… (15.29) | House in multiple occupation licence | Rent Repayment Orders offences: guidance for… / Selective licensing (1.79) **(new)** |
| 5 | Rent Repayment Orders offences: guidance for… / Unlicensed HMO or house (0.52) | House in multiple occupation licence | Rent Repayment Orders offences: guidance for… / Finding out if your landlord has c… (0.03) **(new)** | Rent Repayment Orders offences: guidance for… / The landlord does not have a licen… (1.29) **(new)** |
| ms | **7 ms** | **1 ms** | **10 ms** (dense 8 + bm25 2 + fusion 0) | **744 ms** (dense 7 + bm25 2 + fusion 0 + rerank 735) |

Chunk IDs: dense: electrical-safety-standards-in-the-private-and-social-rented-sectors-guidance-001-b67ae218, private-renting-014-25fe8b07, rent-repayment-orders-offences-guidance-for-tenants-012-1c6812fd, house-in-multiple-occupation-licence-000-5ea74667, rent-repayment-orders-offences-guidance-for-tenants-003-8f4d21bb; bm25: rent-repayment-orders-offences-guidance-for-tenants-004-67ce3558, rent-repayment-orders-offences-guidance-for-tenants-003-8f4d21bb, rent-repayment-orders-offences-guidance-for-tenants-005-4b6d569a, rent-repayment-orders-offences-guidance-for-tenants-012-1c6812fd, house-in-multiple-occupation-licence-000-5ea74667; hybrid: rent-repayment-orders-offences-guidance-for-tenants-003-8f4d21bb, rent-repayment-orders-offences-guidance-for-tenants-012-1c6812fd, private-renting-014-25fe8b07, house-in-multiple-occupation-licence-000-5ea74667, rent-repayment-orders-offences-guidance-for-tenants-005-4b6d569a; hybrid+rerank: rent-repayment-orders-offences-guidance-for-tenants-003-8f4d21bb, house-in-multiple-occupation-licence-000-5ea74667, rent-repayment-orders-offences-guidance-for-tenants-005-4b6d569a, rent-repayment-orders-offences-guidance-for-tenants-004-67ce3558, rent-repayment-orders-offences-guidance-for-tenants-009-323939be

### What can I do about damp and mould in my council flat?

| Rank | dense | bm25 | hybrid | hybrid+rerank |
|---:|---|---|---|---|
| 1 | Damp and mould: understanding and addressing… / Following up to ensure the issue h… (0.70) | Private renting / What your landlord must do (10.53) **(new)** | Damp and mould: understanding and addressing… / How professionals can support land… (0.03) | Damp and mould: understanding and addressing… / How professionals can support land… (5.10) |
| 2 | Damp and mould: understanding and addressing… / Responding to reports of damp and … (0.69) | Damp and mould: understanding and addressing… / How professionals can support land… (9.95) | Damp and mould: understanding and addressing… / Identifying and addressing damp an… (0.03) | Damp and mould: understanding and addressing… / Collaborative working with other p… (4.82) **(new)** |
| 3 | Damp and mould: understanding and addressing… / Identifying and addressing damp an… (0.68) | Damp and mould: understanding and addressing… / Types of professional advice (9.90) **(new)** | Damp and mould: understanding and addressing… / Addressing building deficiencies (0.03) **(new)** | Damp and mould: understanding and addressing… / Identifying and addressing damp an… (4.18) |
| 4 | Damp and mould: understanding and addressing… / How professionals can support land… (0.67) | Damp and mould: understanding and addressing… / Complying with the standards (9.63) **(new)** | Damp and mould: understanding and addressing… / Mould (0.03) **(new)** | Damp and mould: understanding and addressing… / Annex D: energy efficiency funding (4.09) **(new)** |
| 5 | Damp and mould: understanding and addressing… / Responding with urgency and sensit… (0.66) | Damp and mould: understanding and addressing… / Collaborative working with other p… (9.56) **(new)** | Damp and mould: understanding and addressing… / Identifying the root causes of dam… (0.03) **(new)** | Damp and mould: understanding and addressing… / Collaborative working with other p… (4.05) **(new)** |
| ms | **7 ms** | **1 ms** | **8 ms** (dense 7 + bm25 1 + fusion 0) | **764 ms** (dense 8 + bm25 1 + fusion 0 + rerank 755) |

Chunk IDs: dense: damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-040-3a2a0af8, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-001-d26af37a, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-021-8d9aaf98, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-042-888e7eb5, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-025-b84902ff; bm25: private-renting-007-0ef7722d, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-042-888e7eb5, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-041-9247b0bd, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-016-65565edc, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-054-aa159ac4; hybrid: damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-042-888e7eb5, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-021-8d9aaf98, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-030-a9703650, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-023-28f1631e, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-026-7adb953e; hybrid+rerank: damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-042-888e7eb5, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-054-aa159ac4, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-021-8d9aaf98, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-062-93be587b, damp-and-mould-understanding-and-addressing-the-health-risks-for-rented-housing-providers-052-ccc7ed4c

### What is the average rent for a one-bedroom flat in Manchester?

| Rank | dense | bm25 | hybrid | hybrid+rerank |
|---:|---|---|---|---|
| 1 | Housing costs and Universal Credit / If you’re 35 or older and live alo… (0.49) | Housing Benefit / Sharing bedrooms (9.38) **(new)** | Housing costs and Universal Credit / If you’re 35 or older and live alo… (0.03) | Housing costs and Universal Credit / If you’re 35 or older and live alo… (-4.20) |
| 2 | Rent a room in your home / The Rent a Room Scheme (0.49) | Housing costs and Universal Credit / If you’re 35 or older and live alo… (9.15) | Housing Benefit / What you'll get (0.03) | Awaab’s Law Phase 1: Guidance for social lan… / Scenario - Social landlord making … (-7.01) **(new)** |
| 3 | Housing Benefit / Private rent (0.49) | Rent Repayment Orders offences: guidance for… / Converted building HMO test (8.17) **(new)** | Housing Benefit / Sharing bedrooms (0.03) **(new)** | Housing Benefit / Private rent (-7.67) |
| 4 | Tenant Fees Act 2019: guidance for tenants / Rent (0.49) | Awaab’s Law Phase 1: Guidance for social lan… / Scenario - Social landlord making … (8.16) **(new)** | Housing Benefit / Private rent (0.03) | Guide to the Renters’ Rights Act / Awaab’s Law (-7.78) **(new)** |
| 5 | Housing Benefit / What you'll get (0.47) | Guide to the Renters’ Rights Act / Awaab’s Law (8.00) **(new)** | Rent a room in your home / The Rent a Room Scheme (0.02) | Rent a room in your home / The Rent a Room Scheme (-7.87) |
| ms | **7 ms** | **1 ms** | **9 ms** (dense 8 + bm25 1 + fusion 0) | **749 ms** (dense 8 + bm25 1 + fusion 0 + rerank 740) |

Chunk IDs: dense: housing-and-universal-credit-004-d76720ca, rent-room-in-your-home-001-65e9288a, housing-benefit-004-770d2d71, tenant-fees-act-2019-guidance-for-tenants-001-9a065de3, housing-benefit-002-cc98ce6e; bm25: housing-benefit-003-bc2554f7, housing-and-universal-credit-004-d76720ca, rent-repayment-orders-offences-guidance-for-tenants-008-a58607bf, awaabs-law-guidance-for-social-landlords-044-a603aa06, guide-to-the-renters-rights-act-041-707a52f0; hybrid: housing-and-universal-credit-004-d76720ca, housing-benefit-002-cc98ce6e, housing-benefit-003-bc2554f7, housing-benefit-004-770d2d71, rent-room-in-your-home-001-65e9288a; hybrid+rerank: housing-and-universal-credit-004-d76720ca, awaabs-law-guidance-for-social-landlords-044-a603aa06, housing-benefit-004-770d2d71, guide-to-the-renters-rights-act-041-707a52f0, rent-room-in-your-home-001-65e9288a

### How do I apply for a mortgage to buy my first home?

| Rank | dense | bm25 | hybrid | hybrid+rerank |
|---:|---|---|---|---|
| 1 | Housing association homes / Apply for a home (0.45) | Guide to the Renters’ Rights Act / Table 1: Grounds for possession (10.57) **(new)** | Housing costs and Universal Credit / Living in a property you own (0.03) | Solve a residential property dispute / Overview (-0.59) **(new)** |
| 2 | Housing costs and Universal Credit / Living in a property you own (0.43) | Council housing / Secure tenancy (8.46) **(new)** | Council housing / Getting an offer (0.03) **(new)** | Council housing / Apply for a council home (-3.06) |
| 3 | Council housing / Apply for a council home (0.40) | Council housing / Council housing fraud (8.43) **(new)** | Solve a residential property dispute / Overview (0.03) **(new)** | Housing costs and Universal Credit / Other help with housing costs (-3.68) |
| 4 | Housing costs and Universal Credit / If you’re renting (0.39) | Solve a residential property dispute / Overview (8.09) **(new)** | Housing costs and Universal Credit / If you’re renting (0.03) | Guide to the Renters’ Rights Act / Table 1: Grounds for possession (-4.27) **(new)** |
| 5 | Housing costs and Universal Credit / Other help with housing costs (0.37) | Council housing / Getting an offer (7.64) **(new)** | Housing costs and Universal Credit / Other help with housing costs (0.03) | Solve a residential property dispute / Help you can get (-4.70) **(new)** |
| ms | **8 ms** | **1 ms** | **10 ms** (dense 8 + bm25 1 + fusion 0) | **712 ms** (dense 10 + bm25 1 + fusion 0 + rerank 701) |

Chunk IDs: dense: housing-association-homes-000-cb64ea9b, housing-and-universal-credit-011-e4d5ab11, council-housing-000-cb664f5a, housing-and-universal-credit-013-2819dd91, housing-and-universal-credit-001-44ec6251; bm25: guide-to-the-renters-rights-act-010-31af1476, council-housing-002-91ed67db, council-housing-007-efd0a424, housing-tribunals-000-8337d87e, council-housing-001-667fa428; hybrid: housing-and-universal-credit-011-e4d5ab11, council-housing-001-667fa428, housing-tribunals-000-8337d87e, housing-and-universal-credit-013-2819dd91, housing-and-universal-credit-001-44ec6251; hybrid+rerank: housing-tribunals-000-8337d87e, council-housing-000-cb664f5a, housing-and-universal-credit-001-44ec6251, guide-to-the-renters-rights-act-010-31af1476, housing-tribunals-002-5e452474

---

## Observations (written by hand; not regenerated by the script)

Qualitative notes on 10 questions. **Nothing was tuned on these**: candidate pool 20,
RRF k = 60 and the reranker model are defaults, and formal comparison happens in Phase 5 with
the full test set. Ranks below for chunks outside the top 5 come from a separate run with
k = 20 over the same index.

**How much each setup changes the top 5** (chunks not in the dense top 5, per question):
BM25 3.9 of 5 on average, hybrid 2.1, hybrid + reranker 2.8. BM25 finds quite different
chunks from dense search, which is what makes fusing them worthwhile.

**Where BM25 or the reranker helped**

- **Q5 (refused for Universal Credit).** The rule "You cannot be discouraged or stopped from
  renting a property because you get benefits" (Rental discrimination: a guide for tenants)
  is not in the dense top 20. BM25 ranks it **#2** (exact term "benefits"). Plain hybrid only
  ranks it #12, because RRF gives a chunk found by one retriever about half the score of
  chunks found by both. The reranker reads the text and lifts it back to **#2**.
- **Q1 (deposit deadline).** The "within 30 days" chunk moves from #9 (dense) to **#6**
  (hybrid + reranker): closer, but still just outside the top 5. BM25 alone ranks it #16
  because the tokeniser has no stemming, so the query's "protect" does not match
  "protected" or "protection" in the chunk.
- **Q3 (rent increase).** BM25 and the reranker bring in "Apply for an open market rent
  determination" (how to challenge an increase) and "Renters' Rights Act: Increasing rent",
  adding dedicated pages for the second half of the question alongside the "If you do not
  agree with the increase" section dense search already found.
- **Q7 (HMO licence).** Hybrid drops the unrelated electrical safety chunk that dense ranked
  #1, and the reranker then promotes the "House in multiple occupation licence" page from #4
  (dense and hybrid) to #2. It also drops "Private renting: Houses in multiple occupation",
  which has the large-HMO definition, so the gain is partial.

**Where they did not help, or made things worse**

- **Q8 (damp in a council flat).** Every setup is dominated by the 17,000-word landlord-facing
  damp guidance (all 5 slots for hybrid and hybrid + reranker). Neither BM25 nor reranking
  fixes one long document crowding out the rest; that needs per-document limits or diversity
  in the results.
- **Q6 (letting agent fees).** The reranker keeps the right first chunk, but demotes
  "Prohibited payments" from #2 to #4 and fills slots 2, 3 and 5 with weak matches (scores
  near or below zero), such as "Property owned by a company".
- **Q4 (broken boiler).** The reranker puts Awaab's Law (social landlords' emergency repair
  duties) above the private-renting repairs chunk. Relevant to repairs, but less relevant
  to a typical private tenant.
- **Q2 (eviction without a reason).** The reranker drops "Renters' Rights Act overview: Ending
  your tenancy" (hybrid #4), which explains that no-fault evictions have ended, in favour
  of procedural chunks about notices.

**Reranker scores as an "unanswerable" signal.** For the two unanswerable questions, the
reranker's best score is low (-4.2 for Manchester rents, -0.6 for mortgages) and most scores
are strongly negative, while answerable questions mostly have top scores of 4 to 9. That
could support a refusal threshold, but Q4's best score (1.3) and Q6's weak tail show the
ranges overlap, so any threshold must be set on the Phase 4 test set, not on these examples.

**Latency** (median of 3 warm runs per question, laptop CPU):

| Setup | Typical retrieval time |
|---|---|
| dense | 7-8 ms |
| BM25 | 0-1 ms |
| hybrid | 8-10 ms (fusion itself under 1 ms) |
| hybrid + reranker | 680-765 ms |

The cross-encoder adds about **0.7 s per question** (20 candidate pairs of up to 400 tokens
on CPU), roughly 80 times the cost of hybrid retrieval. That is still less than the
1.5-4 s the LLM takes, so it would add roughly 20-40% to end-to-end latency. Smaller candidate
pools or a GPU would reduce it; Phase 5 should measure the accuracy/latency trade-off.

**Ideas to test in Phase 5 (not applied now):** stemming in the BM25 tokeniser; cap on
chunks per document; candidate pool size (10 / 20 / 50); including title and section in the
dense embedding, as BM25 already does; a reranker score threshold for refusals.
