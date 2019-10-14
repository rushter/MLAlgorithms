
# Python3 implementation to count 
# ways to sum up to a given value N 

# Function to count the total 
# number of ways to sum up to 'N' 
def countWays(arr, m, N): 

	count = [0 for i in range(N + 1)] 
	
	# base case 
	count[0] = 1
	
	# Count ways for all values up 
	# to 'N' and store the result 
	for i in range(1, N + 1): 
		for j in range(m): 

			# if i >= arr[j] then 
			# accumulate count for value 'i' as 
			# ways to form value 'i-arr[j]' 
			if (i >= arr[j]): 
				count[i] += count[i - arr[j]] 
	
	# required number of ways 
	return count[N] 
	
# Driver Code 
arr = [1, 5, 6] 
m = len(arr) 
N = 7
print("Total number of ways = ", 
		countWays(arr, m, N)) 
			
 
