# Question 1 
# write me a function that will find the number of repeats for a string x 
# and return the number of repeats

def findRepeats(x):
    for i in range(len(x)):
        if x[i] == x[i+1]:
            return 1 + findRepeats(x[i+1:])
    return 0

# Question 2 

