select
    risk_tier,
    industry_group,
    count(*) as company_count,
    round(avg(pre_ml_risk_score), 1) as avg_risk_score,
    round(avg(gross_margin) * 100, 1) as avg_gross_margin_pct
from {{ ref('mart_risk_candidates') }}
group by risk_tier, industry_group
order by avg_risk_score desc
