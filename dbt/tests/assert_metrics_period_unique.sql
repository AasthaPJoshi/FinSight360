select cik, fiscal_year, count(*) as cnt
from {{ ref('mart_financial_summary') }}
where form = '10-K'
group by cik, fiscal_year
having count(*) > 1
