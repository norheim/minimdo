from polycasebuilder import generate_random_polynomials

eqv = {0: (6, 12, 13, 7), 1: (6, 10, 8), 2: (9, 13, 14, 5), 3: (6, 9, 13), 4: (12,)}
outset = {0: 2, 1: 3, 2: 0, 3: 8, 4: 7}

generate_random_polynomials(eqv, outset, 5)