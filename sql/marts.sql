-- SPOONLY analysis marts (DuckDB). Reads data/raw only.
-- Run through src/build_marts.py, which sets the working directory to the repo root.

-- 1. Merchant-week: one row per merchant per active week, with platform economics.
CREATE OR REPLACE TABLE mart_merchant_week AS
SELECT
    w.merchant_id, m.area, m.category, m.rating, m.delivery_min, m.join_week, m.aov_won,
    w.week, c.week_start, c.holiday,
    w.orders, w.repeat_orders, w.orders - w.repeat_orders                    AS new_orders,
    w.gmv_won, w.merchant_disc_rate, w.platform_disc_rate,
    w.merchant_disc_rate + w.platform_disc_rate                              AS total_disc_rate,
    w.merchant_disc_won, w.platform_disc_won,
    COALESCE(pc.platform_rate, 0)                                            AS campaign_rate,   -- randomised campaign cell
    0.10 * w.gmv_won - w.platform_disc_won                                   AS platform_net_won -- take rate 10% minus platform-funded discount
FROM read_csv_auto('data/raw/merchant_week.csv') w
JOIN read_csv_auto('data/raw/merchants.csv') m USING (merchant_id)
JOIN read_csv_auto('data/raw/calendar.csv') c USING (week)
LEFT JOIN read_csv_auto('data/raw/platform_campaigns.csv') pc
       ON pc.week = w.week AND pc.area = m.area AND pc.category = m.category;

-- 2. Weekly KPI tree: GMV = orders x AOV; orders = new + repeat.
CREATE OR REPLACE TABLE mart_kpi_week AS
SELECT week, MIN(week_start) AS week_start,
       COUNT(*) FILTER (WHERE orders > 0)                       AS active_merchants,
       SUM(orders) AS orders, SUM(new_orders) AS new_orders, SUM(repeat_orders) AS repeat_orders,
       SUM(gmv_won) AS gmv_won, SUM(gmv_won) / NULLIF(SUM(orders), 0) AS aov_won,
       SUM(merchant_disc_won) AS merchant_disc_won, SUM(platform_disc_won) AS platform_disc_won,
       SUM(platform_net_won) AS platform_net_won
FROM mart_merchant_week GROUP BY week ORDER BY week;

-- 3. Merchant profile from the first half (weeks 1-26): inputs for segmentation and levers.
--    Campaign weeks are left out of the demand level so a lucky coupon cell does not move a merchant's tier.
CREATE OR REPLACE TABLE mart_merchant_h1 AS
SELECT merchant_id, ANY_VALUE(area) AS area, ANY_VALUE(category) AS category,
       ANY_VALUE(rating) AS rating, ANY_VALUE(delivery_min) AS delivery_min, ANY_VALUE(aov_won) AS aov_won,
       ANY_VALUE(join_week) AS join_week,
       AVG(orders)  FILTER (WHERE week BETWEEN 14 AND 26 AND campaign_rate = 0) AS orders_wk_q2,
       AVG(orders)  FILTER (WHERE week BETWEEN 1 AND 13 AND campaign_rate = 0)  AS orders_wk_q1,
       AVG(gmv_won) FILTER (WHERE week BETWEEN 14 AND 26 AND campaign_rate = 0) AS gmv_wk_q2,
       SUM(repeat_orders) FILTER (WHERE week <= 26) / NULLIF(SUM(orders) FILTER (WHERE week <= 26), 0) AS repeat_share,
       AVG(CASE WHEN merchant_disc_rate > 0 THEN 1 ELSE 0 END) FILTER (WHERE week <= 26) AS disc_weeks_share
FROM mart_merchant_week
WHERE week <= 26
GROUP BY merchant_id;

-- 4. Reorder-coupon holdout: one row per customer over the 12-week window.
CREATE OR REPLACE TABLE mart_reorder_customer AS
SELECT customer_id, ANY_VALUE(segment) AS segment, ANY_VALUE(treated) AS treated,
       SUM(CAST(coupon_sent AS INT)) AS coupons_sent, SUM(CAST(ordered AS INT)) AS orders,
       SUM(CAST(redeemed AS INT)) AS redemptions
FROM read_parquet('data/raw/reorder_coupon_test.parquet')
GROUP BY customer_id;

-- 5. First-order coupon switchback: one row per area-week.
CREATE OR REPLACE TABLE mart_first_order AS
SELECT area, week, ANY_VALUE(coupon_on) AS coupon_on, COUNT(*) AS new_customers,
       SUM(CAST(used_coupon AS INT)) AS coupon_users, AVG(CAST(reordered_8w AS DOUBLE)) AS reorder_8w
FROM read_csv_auto('data/raw/first_order_coupon_test.csv')
GROUP BY area, week ORDER BY area, week;
