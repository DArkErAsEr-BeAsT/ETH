import matplotlib.pyplot as plt 
import csv


x = []
y = []

with open('/Users/darkeraser/Documents/WORK/ETH/B6/WNMC/jemula802/results_Assignment04/total_thrp.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots)
    for row in plots:
        if (row[6]=="NaN"): row[6] = 0
        y.append(float(row[6]))
        x.append(float(row[0]))

plt.plot(x,y)
plt.xlabel('Time in ms')
plt.ylabel('Throughput in Mb')
plt.title('Overall Throughput in Mbps')
plt.legend()
plt.show()
x = []
y = []

with open('/Users/darkeraser/Documents/WORK/ETH/B6/WNMC/jemula802/results_Assignment04/total_thrp.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots)
    for row in plots:
        if (row[5]=="NaN"): row[5] = 0
        y.append(float(row[5]))
        x.append(float(row[0]))

plt.plot(x,y)
plt.xlabel('Time in ms')
plt.ylabel('Throughput in Mb')
plt.title('Overall Throughput in Mbps')
plt.legend()
plt.show()