import numpy as np

a = np.zeros(shape=[3, 4, 5])
a[2, 3, 4] = 1
a[1, 2, :] = 1
print(a)

b = True
for i in range(5):
    b = np.logical_and(b, a[:, :, i] == 1)
print(b)
print(b.shape)

c = np.argwhere(b == 1)
print(c)
print('c.shape', c.shape)
print(b[c[:, 0], c[:, 1]])
print(a[c[:, 0], c[:, 1]])

# import numpy
# import cProfile

# def cross_diff(A, B):
#     return A[:,None] - B[None,:]

# def crossdiff2 (a,b):
#     ap = numpy.tile (a, (numpy.shape(b)[0],1))
#     bp = numpy.tile (b, (numpy.shape(a)[0],1))

#     return ap - bp.transpose()

# def crossdiff(a,b):
#     c = []
#     for a1 in range(len(a)):
#         for b1 in range(len(b)):
#             c.append (a[a1]-b[b1])
#     x = numpy.array(c)
#     x.reshape(len(a),len(b))
#     return x

# a = numpy.array(range(10000))
# b = numpy.array(range(10000))

# cProfile.run('crossdiff (a,b)')
# cProfile.run('crossdiff2 (a,b)')
# cProfile.run('cross_diff (a,b)')
# cProfile.run('numpy.subtract.outer (a,b)')

"""
100010009 function calls in 69.869 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    2.227    2.227   69.869   69.869 <string>:1(<module>)
        1   46.204   46.204   67.642   67.642 test2.py:13(crossdiff)
        1    0.000    0.000   69.869   69.869 {built-in method builtins.exec}
    10003    0.003    0.000    0.003    0.000 {built-in method builtins.len}
        1   15.077   15.077   15.077   15.077 {built-in method numpy.core.multiarray.array}
100000000    6.358    0.000    6.358    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.000    0.000 {method 'reshape' of 'numpy.ndarray' objects}


         31 function calls in 2.690 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.117    0.117    2.689    2.689 <string>:1(<module>)
        2    0.000    0.000    0.000    0.000 fromnumeric.py:1565(shape)
        2    0.000    0.000    0.891    0.445 shape_base.py:826(tile)
        4    0.000    0.000    0.000    0.000 shape_base.py:897(<genexpr>)
        6    0.000    0.000    0.000    0.000 shape_base.py:907(<genexpr>)
        1    1.682    1.682    2.573    2.573 test2.py:7(crossdiff2)
        2    0.000    0.000    0.000    0.000 {built-in method builtins.all}
        1    0.000    0.000    2.690    2.690 {built-in method builtins.exec}
        2    0.000    0.000    0.000    0.000 {built-in method builtins.len}
        2    0.000    0.000    0.000    0.000 {built-in method numpy.core.multiarray.array}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        2    0.891    0.445    0.891    0.445 {method 'repeat' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 {method 'reshape' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'transpose' of 'numpy.ndarray' objects}


         4 function calls in 0.456 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.035    0.035    0.456    0.456 <string>:1(<module>)
        1    0.421    0.421    0.421    0.421 test2.py:4(cross_diff)
        1    0.000    0.000    0.456    0.456 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}


         4 function calls in 0.459 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.036    0.036    0.459    0.459 <string>:1(<module>)
        1    0.000    0.000    0.459    0.459 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.423    0.423    0.423    0.423 {method 'outer' of 'numpy.ufunc' objects}
"""
