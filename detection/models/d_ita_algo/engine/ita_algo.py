def get_prob(ying, yang, voca_weight):
    ying_len = len(ying)
    yang_len = len(yang)

    if not ying_len or not yang_len:
        return 0.0

    min_len = max(ying_len, yang_len)
    search_range = (min_len // 2) - 1
    if search_range < 0:
        search_range = 0

    ying_flags = [False] * ying_len
    yang_flags = [False] * yang_len

    ying_sub = 0
    yang_sub = 0    
    for i in range(len(yang)):
        if voca_weight.get(yang[i], None) != None:
            additional_point = int(voca_weight[yang[i]])
            val = pow(additional_point, 1.3)
            ying_sub += val
            yang_sub += val
        else:
            ying_sub += 1
            yang_sub += 1
    
    common_chars = 0
    cur = 0
    cnt = 1
    for i, ying_ch in enumerate(ying):
        low = i - search_range if i > search_range else 0
        hi = i + search_range if i + search_range < yang_len else yang_len - 1
        for j in range(low, hi+1):
            if not yang_flags[j] and yang[j] == ying_ch:
                if voca_weight.get(ying_ch, None) != None:
                    ying_flags[i] = yang_flags[j] = True
                    additional_point = int(voca_weight[ying_ch])
                    val = pow(additional_point, 1.3)
                    common_chars += val
                else:
                    common_chars += 1
                cnt += 1
                cur += 1
                break                
    # short circuit if no characters match
    if not common_chars:
        return 0.0

    # adjust for similarities in nonmatched characters
    common_chars = float(common_chars)
    a = common_chars / ying_sub
    b = common_chars / yang_sub
    weight = (a + b) / 2

    return weight
