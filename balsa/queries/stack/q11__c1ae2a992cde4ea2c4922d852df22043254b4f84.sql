SELECT COUNT(*)
FROM
tag as t,
site as s,
question as q,
tag_question as tq
WHERE
t.site_id = s.site_id
AND q.site_id = s.site_id
AND tq.site_id = s.site_id
AND tq.question_id = q.id
AND tq.tag_id = t.id
AND (s.site_name in ('stackoverflow'))
AND (t.name in ('asp.net-mvc-3','bash','c++11','command-line','dataframe','date','function','google-chrome','hibernate','linux','visual-studio'))
AND (q.score >= 10)
AND (q.score <= 1000)
