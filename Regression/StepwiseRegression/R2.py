import numpy as np

def R2(y,yp):

	# Enter your code here
    y = np.array(y).ravel()
    yp = np.array(yp).ravel()
    m = len(y)

    y_bar = np.mean(y)

    SSR = np.sqrt(np.sum(np.power((y-yp),2)))
    SST = np.sqrt(np.sum(np.power((yp-y_bar),2)))

    print(SSR)
    print(SST)
    
    R2 = 1-(SSR/SST)

    return R2
    

    
    
    
