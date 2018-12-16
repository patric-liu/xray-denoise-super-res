a = [1,2 ,3 ,4 ,1 ,1 , 1, 2, 2, 2, 6, 7, 3, 3, 2, 4,5,6,1,2,3,4,1,6,3,2]

import collections

seen = {}
for n,x in enumerate(a):
	n = n + 4000
	if x not in seen:
		seen[x] = [n]
	else:
		seen[x].append(n)

a = 1
b = 1
c = 1

del a, del b