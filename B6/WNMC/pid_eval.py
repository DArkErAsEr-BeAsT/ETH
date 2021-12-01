
import matplotlib.pyplot as plt 

result = []
error = []
derivative = []
integral = []
prop = []
prev_error=[]
output =[]
for n in range(0,40):
    result.append(3)
    error.append(0)
    derivative.append(0)
    integral.append(0)
    prev_error.append(0)
    prop.append(0)
inn = [0,0,0,0,10, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0];

for n in range(0,40):
    prop[n] = result[n]+ inn[n]
    error[n] = result[n] - inn[n]
    integral[n]=integral[n] + error[n] * 0.1
    derivative[n] = (error[n]- prev_error[n]) /0.1
    output = error + integral + derivative
    prev_error[n] = error[n]


# print("output " +str(output))
print("prev_error "+str(prev_error))
print("error "+str(error))
print("integral "+str(integral))
print("derivative "+str(derivative))
print("prop "+str(prop))
