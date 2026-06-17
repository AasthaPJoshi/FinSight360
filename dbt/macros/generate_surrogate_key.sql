{% macro generate_surrogate_key(fields) %}
    md5(
        {%- for field in fields %}
            coalesce(cast({{ field }} as varchar), '')
            {% if not loop.last %} || '|' || {% endif %}
        {%- endfor %}
    )
{% endmacro %}
