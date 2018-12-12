a = [1,52,34,5,1,23,1,5,23,4,10]

def partition(x):
	pivot = x[len(x)-1]
	pivpos = 0
	for i in range(len(x)):
		print(pivpos,i,x)
		if x[i] < pivot:
			pivpos += 1
			if i > pivpos:
				temp = x[pivpos-1]
				x[pivpos-1] = x[i]
				x[i] = temp

	temp = x[pivpos] 
	x[pivpos] = x[-1]
	x[-1] = temp
	return x


print(partition(a))

























'''import random

a = [1,2,3,4,5,6,7,8,9,10]
random.shuffle(a)



def merge(a,b):
	total = len(a) + len(b)
	idx1, idx2 = 0, 0
	output = []
	for _ in range(total):
		if (idx1 < len(a) and idx2 < len(b)):
			if a[idx1] < b[idx2]:
				output.append(a[idx1])
				idx1 += 1

			else:
				output.append(b[idx2])
				idx2 += 1

		elif idx1 == len(a):
			output.append(b[idx2])
			idx2 += 1

		elif idx2 == len(b):
			output.append(a[idx1])
			idx1 += 1

	return output

def sort(x):
	if len(x) == 1:
		return x
	mid = int(len(x)/2)
	arr1 = list(x[0:mid])
	arr2 = list(x[mid:len(x)])
	sorted1 = sort(arr1)
	sorted2 = sort(arr2)
	merged = merge(sorted1,sorted2)
	return merged


	
print(a)
print(sort(a))

'''