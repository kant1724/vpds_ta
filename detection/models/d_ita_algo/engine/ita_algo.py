def get_prob(ying, yang, voca_weight):
    ying_len = len(ying)
    yang_len = len(yang)

    if not ying_len or not yang_len:
        return 0.0

    min_len = max(ying_len, yang_len)
    search_range = (min_len // 2) - 1
    if search_range < 0:
        search_range = 0

    ying_sub = 0
    already_1 = {}
    for i in range(len(yang)):
        if already_1.get(yang[i], None) == True:
            continue
        if voca_weight.get(yang[i], None) != None:
            additional_point = int(voca_weight[yang[i]])
            val = pow(additional_point, 1.5)
            ying_sub += val
        already_1[yang[i]] = True

    common_chars = 0
    cur = 0
    cnt = 1
    already_2 = {}
    for i, ying_ch in enumerate(ying):
        if already_2.get(ying_ch, None) == True:
            continue
        low = i - search_range if i > search_range else 0
        hi = i + search_range if i + search_range < yang_len else yang_len - 1
        for j in range(low, hi+1):
            if yang[j] == ying_ch:
                if voca_weight.get(ying_ch, None) != None:
                    additional_point = int(voca_weight[ying_ch])
                    val = pow(additional_point, 1.5)
                    common_chars += val
                break
        already_2[ying_ch] = True
    # short circuit if no characters match
    if not common_chars:
        return 0.0

    # adjust for similarities in nonmatched characters
    common_chars = float(common_chars)
    weight = common_chars / ying_sub

    return weight
