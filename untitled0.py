for i in range(9):
	b=False
	for j in range(5):
		if i==6:
			b=True
			break
	if b:
		break
	print(i,j)