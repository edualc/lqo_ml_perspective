SELECT t1.name, count(*)
FROM
site as s,
so_user as u1,
question as q1,
answer as a1,
tag as t1,
tag_question as tq1
WHERE
q1.owner_user_id = u1.id
AND a1.question_id = q1.id
AND a1.owner_user_id = u1.id
AND s.site_id = q1.site_id
AND s.site_id = a1.site_id
AND s.site_id = u1.site_id
AND s.site_id = tq1.site_id
AND s.site_id = t1.site_id
AND q1.id = tq1.question_id
AND t1.id = tq1.tag_id
AND (s.site_name in ('askubuntu','math'))
AND (t1.name in ('abstract-algebra','definite-integrals','differential-geometry','discrete-mathematics','elementary-number-theory','exponential-function','linear-algebra','notation','partitioning','permissions','probability-distributions','random-variables'))
AND (q1.score >= 0)
AND (q1.score <= 0)
AND (u1.downvotes >= 0)
AND (u1.downvotes <= 10)
GROUP BY t1.name