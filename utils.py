def invmap(original):
    invor = {}
    for k,v in original.items():
        for x in v:
            invor.setdefault(x,[]).append(k)
    return invor