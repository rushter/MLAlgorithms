# coding:utf-8
import numpy as np
import math

def __nint(val):
    return (int) (val+0.5)

def euclidean_distance(a, b):
    if isinstance(a, list) and isinstance(b, list):
        a = np.array(a)
        b = np.array(b)

    return math.sqrt(sum((a - b) ** 2))


def l2_distance(X):
    sum_X = np.sum(X * X, axis=1)
    return (-2 * np.dot(X, X.T) + sum_X).T + sum_X

def att_distance(a, b):
	if isinstance(a, list) and isinstance(b, list):
		a = np.array(a)
		b = np.array(b)
		c = math.sqrt(sum((a - b) ** 2))
		c2 = __nint(c)

		return c2+1 if c2<c else c2

def geo_distance(t1, t2):
    latitude = [0,0]
    longitude = [0,0]
    PI = 3.141592
    RRR = 6378.388

    d = __nint( t1[1] )
    m = t1[1] - d
    latitude[0] = PI * (d + 5.0 * m / 3.0 ) / 180.0
    d = __nint( t1[2] )
    m = t1[2] - d
    longitude[0] = PI * (d + 5.0 * m / 3.0 ) / 180.0
    
    d = __nint( t2[1] )
    m = t2[1] - d
    latitude[1] = PI * (d + 5.0 * m / 3.0 ) / 180.0
    d = __nint( t2[2] )
    m = t2[2] - d
    longitude[1] = PI * (d + 5.0 * m / 3.0 ) / 180.0

    q1 = math.cos( longitude[0] - longitude[1] )
    q2 = math.cos( latitude[0] - latitude[1] )
    q3 = math.cos( latitude[0] + latitude[1] )
    val = (int) ( RRR * math.acos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0)
    return val

