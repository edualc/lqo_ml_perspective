
select distinct account.display_name
from
tag t1, site s1, question q1, tag_question tq1, so_user u1,
tag t2, site s2, question q2, tag_question tq2, so_user u2,
account
where
-- group theory askers
s1.site_name='unix' AND
t1.name  = 'text-processing' AND
t1.site_id = s1.site_id AND
q1.site_id = s1.site_id AND
tq1.site_id = s1.site_id AND
tq1.question_id = q1.id AND
tq1.tag_id = t1.id AND
q1.owner_user_id = u1.id AND
q1.site_id = u1.site_id AND

-- D&D askers
s2.site_name='stackoverflow' AND
t2.name  = 'command-line' AND
t2.site_id = s2.site_id AND
q2.site_id = s2.site_id AND
tq2.site_id = s2.site_id AND
tq2.question_id = q2.id AND
tq2.tag_id = t2.id AND
q2.owner_user_id = u2.id AND
q2.site_id = u2.site_id AND

-- intersect
u1.account_id = u2.account_id AND
account.id = u1.account_id;

