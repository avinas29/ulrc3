"""Deterministic fixtures covering every supported content type.

These are hand-written to contain *checkable* facts: numbers, dates, URLs,
identifiers, constraints and negations that the guarantee tests assert on.
"""

from __future__ import annotations

MARKDOWN_DOC = """# Billing Service Guide

## Overview

The billing service handles invoicing for all Acme Corporation Limited customers.
It is important to note that the service is basically a thin wrapper around the
ledger, and it actually just forwards requests. Acme Corporation Limited operates
this service in three regions.

## Limits

The API must never return more than 100 invoices per page. Requests are rate
limited to 60 per minute; exceeding that returns HTTP 429. The retry budget is
3 attempts with exponential backoff starting at 250ms.

For example, a client fetching 450 invoices will issue 5 paginated requests.
For instance, another client fetching 90 invoices issues a single request.

## Endpoints

POST /v2/invoices creates an invoice. Returns 201 on success.
GET /v2/invoices/{id} fetches one invoice. Returns 404 if unknown.

| field | type | required |
| --- | --- | --- |
| customer_id | string | yes |
| amount_cents | integer | yes |
| currency | string | no |
| due_date | date | no |

## Deprecations

The v1 endpoints were removed on 2024-03-15. Migration guide is at
https://docs.acme.io/billing/migrate-v2. Do not use the legacy
`/v1/invoice` path.

As previously mentioned, the API must never return more than 100 invoices per
page, and it is worth noting that this limit is not configurable.
"""

PYTHON_CODE = '''"""Payment processing helpers.

This module is part of the Acme billing stack.
Copyright (c) 2024 Acme Corporation Limited. All rights reserved.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TIMEOUT_MS = 250


@dataclass
class Receipt:
    """A payment receipt."""

    receipt_id: str
    amount_cents: int
    currency: str = "USD"


class PaymentGateway:
    """Talks to the upstream processor."""

    def __init__(self, api_key: str, base_url: str = "https://api.acme.io/v2") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self._session = None

    def charge(self, customer_id: str, amount_cents: int, currency: str = "USD") -> Receipt:
        """Charge a customer and return a Receipt.

        Raises ValueError when amount_cents is not positive.
        """
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        rid = _make_id(customer_id, amount_cents)
        logger.info("charging %s for %d", customer_id, amount_cents)
        return Receipt(receipt_id=rid, amount_cents=amount_cents, currency=currency)

    def refund(self, receipt_id: str, amount_cents: int) -> Receipt:
        """Refund part or all of a receipt."""
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        return Receipt(receipt_id=receipt_id, amount_cents=-amount_cents)


def _make_id(customer_id: str, amount_cents: int) -> str:
    seed = f"{customer_id}:{amount_cents}"
    return f"rcpt_{abs(hash(seed)) % 10**12:012d}"


def duplicate_make_id(user_id: str, cents: int) -> str:
    seed = f"{user_id}:{cents}"
    return f"rcpt_{abs(hash(seed)) % 10**12:012d}"


def unused_helper(values: list[int]) -> float:
    """Compute the geometric mean of positive values."""
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values if v > 0) / len(values))
'''

CONVERSATION = """user: Hey there! Good morning.
assistant: Certainly! Good morning. How can I help you today?
user: I need to set up billing for our new tenant. My name is Priya and I work at Northwind.
assistant: Sure thing, happy to help. Could you tell me the plan you want?
user: We want the Enterprise plan. Also I prefer metric units in all reports.
assistant: Great question. The Enterprise plan costs $4,200 per month and includes 50 seats.
user: Thanks! Actually, correction: we want the Business plan, not Enterprise.
assistant: Understood. The Business plan is $1,800 per month with 20 seats.
user: Perfect. Let's go with Business. Invoice us on the 1st of each month.
assistant: No problem at all.
user: One more thing, never email invoices to billing@northwind.example - use ap@northwind.example instead.
assistant: Got it, noted.
user: Cool, thanks!
assistant: You're welcome! Anything else?
user: What was the seat count again?
"""

LOGS = """2024-03-15T10:22:01Z INFO  api.gateway request GET /v2/invoices status=200 dur=12ms
2024-03-15T10:22:01Z INFO  api.gateway request GET /v2/invoices status=200 dur=15ms
2024-03-15T10:22:02Z INFO  api.gateway request GET /v2/invoices status=200 dur=11ms
2024-03-15T10:22:02Z INFO  api.gateway request GET /v2/invoices status=200 dur=19ms
2024-03-15T10:22:03Z WARN  billing.worker retry 1/3 for order 88213 after 250ms
2024-03-15T10:22:04Z WARN  billing.worker retry 2/3 for order 88213 after 500ms
2024-03-15T10:22:05Z ERROR billing.worker payment failed for order 88213: gateway timeout after 3 retries
2024-03-15T10:22:05Z INFO  api.gateway request GET /v2/invoices status=200 dur=14ms
2024-03-15T10:22:06Z INFO  api.gateway request GET /v2/invoices status=200 dur=13ms
2024-03-15T10:22:07Z INFO  api.gateway request POST /v2/invoices status=201 dur=44ms
2024-03-15T10:22:08Z FATAL db.pool connection pool exhausted: 100/100 in use
2024-03-15T10:22:09Z INFO  api.gateway request GET /v2/invoices status=200 dur=16ms
"""

JSON_DOC = """{
  "tenant_id": "northwind",
  "plan": "business",
  "seats": 20,
  "monthly_cents": 180000,
  "features": ["invoicing", "reporting", "sso"],
  "invoices": [
    {"id": "inv_001", "amount_cents": 180000, "currency": "USD", "status": "paid", "due_date": "2024-03-01"},
    {"id": "inv_002", "amount_cents": 180000, "currency": "USD", "status": "paid", "due_date": "2024-04-01"},
    {"id": "inv_003", "amount_cents": 180000, "currency": "USD", "status": "open", "due_date": "2024-05-01"},
    {"id": "inv_004", "amount_cents": 180000, "currency": "USD", "status": "open", "due_date": "2024-06-01"},
    {"id": "inv_005", "amount_cents": 195000, "currency": "USD", "status": "open", "due_date": "2024-07-01"}
  ],
  "contact": {"name": "Priya", "email": "ap@northwind.example", "phone": "+1-555-0100"}
}"""

LEGAL = """1. Definitions

1.1 "Agreement" means this master services agreement dated 2024-01-15.
1.2 "Services" refers to the billing and invoicing services described in Schedule A.
1.3 "Fees" means the amounts payable under Section 4.

2. Term

2.1 This Agreement commences on 2024-02-01 and continues for 24 months.
2.2 Either party may terminate for convenience on 60 days written notice.

3. Obligations

3.1 The Provider shall maintain 99.9% uptime measured monthly.
3.2 The Provider must not disclose Confidential Information to third parties.
3.3 Notwithstanding Section 3.2, disclosure is permitted where required by law.

4. Fees

4.1 The Customer shall pay $1,800 per month within 30 days of invoice.
4.2 Late payments accrue interest at 1.5% per month.
"""

SUPPORT_TICKET = """Ticket #48213 - Priority: High - Opened 2024-03-15 09:14

Customer: Northwind Ltd (tenant northwind, plan business)

Hi, thanks for the quick reply earlier! I really appreciate it.

We are seeing failed payments on order 88213. The error says "gateway timeout
after 3 retries". This started around 10:22 UTC today. Our card ending 4242
works fine on other services.

Thanks again for looking into this!

--
Agent note: reproduced. Gateway latency spiked to 4200ms at 10:22. Pool
exhausted (100/100). Escalating to platform team.

Agent note: Customer must be refunded $1,800 if the charge posted twice.
Do not retry the charge manually.
"""

RAG_CHUNKS = [
    "Acme Corporation Limited was founded in 2011 and is headquartered in Dublin, Ireland. "
    "The company employs 1,240 people across three regions.",
    "Acme's billing platform processes approximately 4.2 million invoices per month. "
    "Peak throughput is 900 invoices per second.",
    "The company was founded in 2011 with headquarters in Dublin. It has around 1,240 employees "
    "spread over three regions.",
    "Acme's flagship product is the Ledger API. The Ledger API exposes endpoints at "
    "https://api.acme.io/v2 and requires an API key.",
    "Quarterly revenue for Q4 2024 was $18.4M, up 22% year over year. Gross margin was 71%.",
    "Support is available 24/7 via support@acme.io. The SLA guarantees a 4 hour first response.",
    "In Q4 2024 the company reported revenue of $18.4M, a 22% increase year over year, "
    "with gross margin of 71%.",
    "Acme Corporation Limited operates data centres in Dublin, Frankfurt and Virginia.",
]

ALL_FIXTURES = {
    "markdown": MARKDOWN_DOC,
    "code": PYTHON_CODE,
    "conversation": CONVERSATION,
    "logs": LOGS,
    "json": JSON_DOC,
    "legal": LEGAL,
    "support": SUPPORT_TICKET,
}
