from collections import OrderedDict
Ein = {
    1: (7,),
    2: (3,6),
    3: (1,),
    4: (2,4),
    5: (3,2)
}
# IMPORTANT: Need to enforce that each output variable 
# can only be connected to one function
Eout = {
    1: (6,),
    2: (1,),
    3: (2,), 
    4: (None,), # this gives us the size of the outputs
    5: (5,), # Use None to indicate the size of the output
}

Rin = {
    #3: (2,) # in original model its 3: (2,), so we have x_2 = f3(x_1)
}

Stree = {
    2:1
}
Ftree = OrderedDict((
    (1,1),
    (2,2),
    (3,2),
    (4,2),
    (5,1)
))
Vtree = {
    3:2
}