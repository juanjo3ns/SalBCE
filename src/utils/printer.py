
table = [
	["PATH OUT", "", "", ""],
	["DEPTH", 0, "DATA AUGMENTATION", 1],
	["COORDCONV", 0, "OPTICAL FLOW", 1],
	["BATCH SIZE", 0, "LEARNING RATE", 1],
	["# EPOCHS", 0, "TRAIN. PARAM.", 1],
]
def print_table(table):
	longest_cols = [(max([len(str(row[i])) for row in table[1:3]]) + 3) for i in range(len(table[0]))]
	row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
	for row in table:
		print(row_format.format(*row))


def param_print(params):
	for i,(p,q) in enumerate(zip(params,[1,3]*5)):
		table[int(i/2)][q] = p
	print_table(table)
