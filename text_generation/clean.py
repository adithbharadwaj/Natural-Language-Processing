
c = open('english_clean.txt', "w")

with open("english.txt", "r") as f:
    for line in f:
    	line = list(line)

    	if('-' in line):
    		line.remove('-')
    	if('-' in line):	
    		line.remove('-')
    	
    	if(',' in line):
    		line.remove(',')
    		line.remove(',')
    	
    	if('.' in line):
    		line.remove('.')
    	
    	if("'" in line):
    		line.remove("'")  
        
    	str1 = ''.join(line)	
    	
    	c.write(str1)

