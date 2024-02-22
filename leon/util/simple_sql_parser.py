import re

import networkx as nx


def _CanonicalizeJoinCond(join_cond):
    """join_cond: 4-tuple"""
    t1, c1, operation, t2, c2 = join_cond
    if t1 < t2:
        return join_cond
    else:
        # operation needs to be inverted
        if '>' in operation:
            # a1.col1 > a2.col2 ==> a2.cols < a1.col1
            operation = operation.replace('>','<')
        elif '<' in operation:
            # a1.col1 <= a2.col2 ==> a2.cols >= a1.col1
            operation = operation.replace('<','>')
        
        return t2, c2, operation, t1, c1


def _DedupJoinConds(join_conds):
    """join_conds: list of 5-tuple (t1, c1, operation, t2, c2)."""
    canonical_join_conds = [_CanonicalizeJoinCond(jc) for jc in join_conds]
    return sorted(set(canonical_join_conds))


def _GetJoinConds(sql):
    """Returns a list of join conditions in the form of (t1, c1, t2, c2)."""
    join_cond_pat = re.compile(
        r"""
        (\w+)            # 1st table
        \.               # the dot "."
        (\w+)            # 1st table column
        \s*              # optional whitespace
        (>=|<=|>|<|=)    # the equal sign "=" (to catch the operation)
        \s*              # optional whitespace
        (\w+)            # 2nd table
        \.               # the dot "."
        (\w+)            # 2nd table column
        """, re.VERBOSE)
    and_pattern = 'AND' if 'AND' in sql else 'and'
    
    join_conds = []
    for sql_part in sql.split(and_pattern):
        if 'interval' in sql_part:
            continue
        join_conds.extend(join_cond_pat.findall(sql_part))

    # handle the special 'interval' patterns in STACK queries
    # (also changed in plans_lib.py -> #to_sql() / #KeepRelevantJoins())
    stack_pat = re.compile(r"(\w+)\.(\w+)\s*(>=|<=|>|<|=)\s*(\w+)\.(\w+\s*\+\s*'\w+\s*\w+'::interval)", re.VERBOSE)
    interval_conds = stack_pat.findall(sql)

    return _DedupJoinConds(join_conds + interval_conds)


def _GetGraph(join_conds):
    g = nx.MultiGraph()
    for t1, c1, operation, t2, c2 in join_conds:
        g.add_edge(t1, t2, join_keys={t1: c1, t2: c2}, operation=operation)
    return g


def _FormatJoinCond(tup):
    t1, c1, operation, t2, c2 = tup
    return f"{t1}.{c1} {operation} {t2}.{c2}"


def ParseSql(sql, filepath=None, query_name=None):
    """Parses a SQL string into (nx.Graph, a list of join condition strings).

    Both use aliases to refer to tables.
    """

    join_conds = _GetJoinConds(sql)
    graph = _GetGraph(join_conds)
    join_conds = [_FormatJoinCond(c) for c in join_conds]
    return graph, join_conds


if __name__ == '__main__':
    query = """
    select COUNT(distinct account.display_name)
    from
    tag t1, site s1, question q1, answer a1, tag_question tq1, so_user u1,
    account
    where
    -- answerers posted at least 1 yr after the question was asked
    s1.site_name='math' and
    t1.name = 'optimization' and
    t1.site_id = s1.site_id and
    q1.site_id = s1.site_id and
    tq1.site_id = s1.site_id and
    tq1.question_id = q1.id and
    tq1.tag_id = t1.id and
    a1.site_id = q1.site_id and
    a1.question_id = q1.id and
    a1.owner_user_id = u1.id and
    a1.site_id = u1.site_id and
    a1.creation_date >= q1.creation_date + '1 year'::interval and

    -- to get the display name
    account.id = u1.account_id;
    """

    join_graph, join_conds = ParseSql(query)
    import code; code.interact(local=dict(globals(), **locals()))


    a, b = ParseSql(
        'SELECT MIN(mc.note) AS production_note, MIN(t.title) AS movie_title,MIN(t.production_year) AS movie_year FROM company_type AS ct,info_type AS it,movie_companies AS mc,movie_info_idx AS mi_idx,title AS t WHERE ct.kind = \'production companies\'AND it.info = \'top 250 rank\'AND mc.note NOT LIKE \'%(as Metro-Goldwyn-Mayer Pictures)%\'AND (mc.note LIKE \'%(co-production)%\'OR mc.note LIKE \'%(presents)%\')AND ct.id = mc.company_type_id AND t.id = mc.movie_id AND t.id = mi_idx.movie_id AND mc.movie_id = mi_idx.movie_id AND it.id = mi_idx.info_type_id;')
    print(a)
    print(b)
