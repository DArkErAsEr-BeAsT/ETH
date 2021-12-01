

result = [3]*40
error = []*40
derivative = []*40
integral = []*40
inn = [0,0,0,0,10, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0];
prev_error = 0
for n in range(1,40):
    
    error[n] = result[n] - inn[n]
    integral[n]=integral[n] + error[n] * 0.1;
    derivative[n] = (error [n]- prev_error[n]) /0.1;
    output = error + integral + derivative;
    prev_error = error;

for n in range (1,40):
    print(output[n])
    print(result[n])
    print(error[n])
    print(integral[n])
    print(derivative[n])