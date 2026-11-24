
-- Q1 Revenue by Plan

        SELECT
            p.plan_name,
            SUM(r.amount_inr) AS total_revenue,
            ROUND(AVG(r.amount_inr), 2) AS average_revenue,
            COUNT(DISTINCT s.subscriber_id) AS subscriber_count

        FROM recharges r

        INNER JOIN subscribers s
            ON r.subscriber_id = s.subscriber_id

        INNER JOIN plans p
            ON s.plan_id = p.plan_id

        WHERE LOWER(r.status) = 'success'

        GROUP BY p.plan_name

        ORDER BY total_revenue DESC;
    

-- Q2 Never Successfully Recharged

        SELECT
            s.subscriber_id,
            s.city,
            s.tenure_months

        FROM subscribers s

        LEFT JOIN recharges r
            ON s.subscriber_id = r.subscriber_id
            AND LOWER(r.status) = 'success'

        WHERE r.recharge_id IS NULL

        ORDER BY s.subscriber_id;
    

-- Q3 Ranking Within City

        WITH subscriber_revenue AS (
            SELECT
                s.subscriber_id,
                s.city,
                SUM(r.amount_inr) AS total_success

            FROM subscribers s

            INNER JOIN recharges r
                ON s.subscriber_id = r.subscriber_id

            WHERE LOWER(r.status) = 'success'

            GROUP BY s.subscriber_id, s.city
        ),

        ranked AS (
            SELECT
                subscriber_id,
                city,
                total_success,

                RANK() OVER (
                    PARTITION BY city
                    ORDER BY total_success DESC
                ) AS rank_value,

                DENSE_RANK() OVER (
                    PARTITION BY city
                    ORDER BY total_success DESC
                ) AS dense_rank_value,

                ROW_NUMBER() OVER (
                    PARTITION BY city
                    ORDER BY total_success DESC, subscriber_id
                ) AS row_number_value

            FROM subscriber_revenue
        )

        SELECT *
        FROM ranked
        ORDER BY city, row_number_value;
    

-- Q4 Running Total

        SELECT
            subscriber_id,
            recharge_id,
            recharge_date,
            amount_inr,

            SUM(amount_inr) OVER (
                ORDER BY recharge_date, recharge_id
                ROWS BETWEEN UNBOUNDED PRECEDING
                AND CURRENT ROW
            ) AS running_total

        FROM recharges

        WHERE subscriber_id = 'TP00010'
          AND LOWER(status) = 'success'

        ORDER BY recharge_date, recharge_id;
    
