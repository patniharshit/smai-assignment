f = "breast-cancer-wisconsin.data"

fp = open(f,'r')

data = fp.read()

lines = data.split('\n')
for i,l in enumerate(lines):
	var = l.split(',')
	y = 0
	flag = True
	for j,x in enumerate(var):
		try:
			y = int(x)
		except:
			flag = False
	if flag:
		print l
