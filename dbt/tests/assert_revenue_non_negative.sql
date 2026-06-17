select cik, company_name, period_end, revenues
from {{ ref('mart_financial_summary') }}
where revenues < 0
  and revenues is not null
