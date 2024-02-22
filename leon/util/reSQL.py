import re


def _FormatJoinCond(tup):
    t1, c1, operation, t2, c2 = tup
    return f"{t1}.{c1} {operation} {t2}.{c2}"


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

    result = [_FormatJoinCond(c) for c in join_conds + interval_conds]
    return result


def deleteUnless(oldlist):
    newlist = []
    for i in oldlist:
        if i == '' or i == ';':
            continue
        else:
            if i[-1] == ';':
                i = i[:len(i) - 1]
            newlist.append(i)
    return newlist


def dealwithBetween(andlist):
    newlist = []
    temFlag = -1
    for i in range(0, len(andlist)):    
        between_pattern = 'BETWEEN' if 'BETWEEN' in andlist[i] else 'between'
        if between_pattern in andlist[i]:
            newlist.append(andlist[i] + ' AND ' + andlist[i + 1])
            temFlag = i + 1
        if i > temFlag:
            newlist.append(andlist[i])
    return newlist


def getFliters(sqlfile):
    with open(sqlfile, 'r') as f:
        data = f.read().splitlines()
        data = [line for line in data if not line.startswith('--')]
        sql = ' '.join(data)

    where_pattern = 'WHERE' if 'WHERE' in sql else 'where'
    and_pattern = 'AND' if 'AND' in sql else 'and'

    joins = _GetJoinConds(sql)
    begin = sql.index(where_pattern)
    tem = sql[begin:].replace(where_pattern, '')

    # If this is not sorted by descending length, there might be replacements of 
    # sub-pattern, such as "q1.site_id = s1.site_id" replacing part of
    # "tq1.site_id = s1.site_id" in STACK
    for i in sorted(joins, key=len, reverse=True):
        tem = tem.replace(i, '')
    temandList = []
    for i in tem.split(and_pattern):
        temandList.append(i.strip())
    andList = deleteUnless(temandList)
    andList = dealwithBetween(andList)
    return andList


def getSelectExp(sqlfile):
    with open(sqlfile, 'r') as f:
        data = f.read().splitlines()
        data = [line for line in data if not line.startswith('--')]
        sql = ' '.join(data)

    select_pattern = 'SELECT' if 'SELECT' in sql else 'select'
    from_pattern = 'FROM' if 'FROM' in sql else 'from'
    
    begin = sql.index(select_pattern)
    end = sql.index(from_pattern)
    tem = sql[begin:end].replace(select_pattern, '')
    selectExplist = []
    for i in tem.split(','):
        selectExplist.append(i.strip())
    return selectExplist


if __name__ == '__main__':
    # sqlFiles = './../balsa/queries/join-order-benchmark/1b.sql'
    sqlFiles = './../balsa/queries/stack/q4__q4-002.sql'
    print(getFliters(sqlFiles))
    s = getSelectExp(sqlFiles)
    for i in s:
        print(i[i.index('(') + 1:i.index(')')].split('.')[0])
