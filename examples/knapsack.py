# Python program of a space optimized DP solution for 
# 0-1 knapsack problem. 

# val[] is for storing maximum profit for each weight 
# wt[] is for storing weights 
# n number of item 
# W maximum capacity of bag 
# dp[W+1] to store final result 
def KnapSack(val, wt, n, W): 
	
	# array to store final result 
	# dp[i] stores the profit with KnapSack capacity "i" 
	dp = [0]*(W+1); 

	# iterate through all items 
	for i in range(n): 
		
		#traverse dp array from right to left 
		for j in range(W,wt[i],-1): 
			dp[j] = max(dp[j] , val[i] + dp[j-wt[i]]); 
			
	'''above line finds out maximum of dp[j](excluding ith element value) 
	and val[i] + dp[j-wt[i]] (including ith element value and the 
	profit with "KnapSack capacity - ith element weight") *'''
	return dp[W]; 


# Driver program to test the cases 
val = [7, 8, 4]; 
wt = [3, 8, 6]; 
W = 10; n = 3; 
print(KnapSack(val, wt, n, W)); 


