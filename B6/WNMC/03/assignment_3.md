#### David Colonna - 18944827



# Step 2

## Simulation Results

### 2 Stations

<img src="/Users/darkeraser/Pictures/Screenshot 2021-11-05 at 15.01.58.png" style="zoom:50%;" />

<img src="/Users/darkeraser/Pictures/Screenshot 2021-11-17 at 11.53.22.png" alt="Screenshot 2021-11-17 at 11.53.22" style="zoom:50%;" />

Here we can see that is alittle spiky at the beginning of the transmission window but it stabilizes after 1350 ms.
Regarding the average packet size, it is very irregular, since it spans from 660 bytes to 490 bytes.

---

### 5 Stations

### ![Screenshot 2021-11-17 at 11.28.39](/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-17 at 11.28.39.png)

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117153028212.png" alt="image-20211117153028212" style="zoom:50%;" /> 

We can see that fluctuates a lot a the beginning, starting at way over 4Mbps, but then going under 3.9 and staying relatively constant at a little over 3.9Mbps.

  <img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117153102148.png" alt="image-20211117153102148" style="zoom: 50%;" />

We can also see that the averag packet size stays around 580 bytes, but it goes constantly down by a few bytes every time interval.

![image-20211117121407431](/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117121407431.png)

From this graph we can see that the average dela stays the same at around 6ms, while the overall maximum delaypeaks at 80 at the beginning and then stays rather constant at around 70 ms.

I also tried to simulate the same scenario but instead of using “data” type, I set it too “saturation”, however this had not drastically different results.

---

### 10 Stations

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-17 at 12.08.41.png" alt="Screenshot 2021-11-17 at 12.08.41" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-17 at 13.43.05.png" alt="Screenshot 2021-11-17 at 13.43.05" style="zoom:50%;" />

We can see that the overall throughput stays relatively constant, apart from the big spkie at the start of the transmission window.

Regarding the average packet size, we can recognize a pattern : the packet size grows of about 10 bytes hits a threshhold and drops down drastically. This pattern repeats over and over. Furthermore the packet size is the same as the previous scenario as is the throughput.

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-17 at 13.54.13.png" alt="Screenshot 2021-11-17 at 13.54.13" style="zoom:50%;" />

We can see that the overall max delay is much bigger in the case of 10 stations since we jump from around 70ms for the 5 stations version to about 375 ms on maximum . The same spikes can be observed in both cases.

---

### 15 Stations

> I didn’t add a screenshot of the gui since it was very ressemblant to the previous one and I could not show all the stations at once.

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-17 at 14.04.45.png" alt="Screenshot 2021-11-17 at 14.04.45" style="zoom:50%;" />

We observe that the throughput has again diminished when comparing to the 10 station - scenario, going from the 3.8 Mbps average to a 3.6 Mbps average. The packet size seem to be quiet similar from one scenario to the other.

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117140848847.png" alt="image-20211117140848847" style="zoom:50%;" />

The maximum delay is now way bigger than the previous simulation, peaking at 890ms while before it was around 430ms.

#### Conclusion

We can already see a tendency that these bigger and bigger number of stations might not be beneficial to the overal network when looking at metrics like throughput and delay. What we will try now is a jumping a few steps and setting up a scenario with 50 stations.

### 50 Stations

### <img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-17 at 14.35.17.png" alt="Screenshot 2021-11-17 at 14.35.17" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-17 at 14.35.00.png" alt="Screenshot 2021-11-17 at 14.35.00" style="zoom:50%;" />

As we can see here the throughput has gone down significantly again , with a value way under the 3 Mbps line. Additionaly we can see that average packet size has gone down also from 650 bytes on average for the last scenario to now 560 bytes.

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117143928885.png" alt="image-20211117143928885" style="zoom: 50%;" />

Regarding the average max delay we reach now a whapping 5000 ms delay at the peak. We also observe that the max delay keeps getting bigger and bigger as the simulation advances.



### Final Conclusion

We have seen that the optimal scenarion regarding throughput seems to be the one with 4 to 5 stations, when going above that we can see that the delays are getting ridiculous, making the time to complete the simulation around 5 minutes for 50 stations. Also to be observed is that adding more and more stations to the network means that the throughput is getting less and less, mainly because of the packet size which changes irregularly over the time span of the simulation. It is not due to the number of packets sent every interval since this number always stays around 400 packets very interval, with variations between 390 and 420 packets/interval.

# Step 3

## Simulation results - 5 stations

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117165404422.png" alt="image-20211117165404422" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117153141691.png" alt="image-20211117153141691" style="zoom:50%;" />



<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117121407431.png" alt="image-20211117121407431" style="zoom:50%;" />

We use the same results as before, this is our base model for the following experiment.

#### Contention Window minimum at 100 instead of 15

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117165440762.png" alt="image-20211117165440762" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117171009942.png" alt="image-20211117171009942" style="zoom:50%;" />



We see immediately that by increasing the contention window size by a factor of appr. 7 we get a throughput of almost 4 Mbps on average. But we can also see a downward trend. Hence this is positive. Looking at the delay we can see that we have a lower delay than with the standard approach on maximum, but on average it is a little bit higher.

Let us try a smaller approach.

#### Contention window size = 30

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117170246783.png" alt="image-20211117170246783" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117170354226.png" alt="image-20211117170354226" style="zoom:50%;" />

Here we can see that the throughput is pretty constant just under the 4Mbps bar, similar to the previous setting. Remarquably the  delay is lower than the scenario where we didn’t change the contention window value, hence this would be around the optimal value for the 5 station-scenario, since on average as well as on maximum the system performs better delay-wise.

Now let us try a minimal value for the contention window and see how the system performs.

#### Contention window size = 5

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117171525863.png" alt="image-20211117171525863" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117171518536.png" alt="image-20211117171518536" style="zoom:50%;" />

We can see that with a lower contention window size the throughput is slightly lower than before, reaching only 3.8 Mbps. Hence no real conclusion can be built using this information. However we can clearly see that the average delay is higher than any of the other 5-station-scenarios, and that the maximum delay is of the charts.

##### Conclusion

We can see that the best setting for the 5 station-scenario is to set the CW to around the double of the default value.

## SImulation results - 10 stations

### Default Contention Window Size

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-17 at 13.43.05.png" alt="Screenshot 2021-11-17 at 13.43.05" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-17 at 13.54.13.png" alt="Screenshot 2021-11-17 at 13.54.13" style="zoom:50%;" />

### CW Size = 100

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117172455535.png" alt="image-20211117172455535" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117172601298.png" alt="image-20211117172601298" style="zoom:50%;" />

We can immediately see that throughput is way higher than the when using the default setting, giving us on average 4.4Mbps vs 3.75Mbps before.
Regarding the delay we can see that the maximum delay is 3 times as low than when using the default setting, which means that  this CW size is good on all sides for this particular number of stations.

Let us try an even bigger CW size.

### CW Size = 150

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117173029397.png" alt="image-20211117173029397" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117173105159.png" alt="image-20211117173105159" style="zoom:50%;" />

As we see here the effect of setting the CW size to 10 times as much as default gives us an equivalent performance (- 0.05Mbps) to the CW size=100 experiment, while reducing even more our maximum delay , down from 70 to 52 on average, which is still a positive outcome. On average we get the same delay as before, it is only the maximum delay that changes.
Let us try to see when we reach a performance threshold. 

### CW Size = 250

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117173605208.png" alt="image-20211117173605208" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117173612348.png" alt="image-20211117173612348" style="zoom:50%;" />

We observe that the throughput keeps regressing at the same rate as before, but it still acceptable. However what is remarkable with this scenario is that the delay stays quite the same as the CW=150 setting, hence we can say with certitude that the optimal CW size lies somewhere between 150 and 250 for the 10-station-scenario.

## Simulation Results - 15 Stations

### Default CW Size

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-17 at 14.04.45.png" alt="Screenshot 2021-11-17 at 14.04.45" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117140848847.png" alt="image-20211117140848847" style="zoom:50%;" />

### CW Size = 100

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117175347238.png" alt="image-20211117175347238" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117175407158.png" alt="image-20211117175407158" style="zoom:50%;" />

We can observe approximately the same results as in the previous scenario, since the throughput goes from 3.6 Mbps on average to 4.5 Mbps with this setting. Similarly the delay is much lower with this setting, since it goes from 750 ms to 120 on maximum. 

Lets us directly jump to a scenario where the CW is 10 times the default one.

### CW Size = 150

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117180748776.png" alt="image-20211117180748776" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117180831933.png" alt="image-20211117180831933" style="zoom:50%;" />

We can immediately see that the throughput is now at nearly 4.5 Mbps from the middle to the end of the simulation, hence an improvement.
However we can also see that the delay is loweer than before, peaking at 140ms, with a maximal average of 100 ms. Both of these parameters indicate an improvement over the previous scenario, hence we should keep pushing!

### CW Size = 450

we will now try to set the CW size to 30 times the original setting.

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117181254744.png" alt="image-20211117181254744" style="zoom:50%;" />

We can see that now there is no improvement in the throughput area since we go down by 0.2 Mbps compared to the previous setting. Hence this is not optimal.

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117181343528.png" alt="image-20211117181343528" style="zoom:50%;" />

When we look at the delay we can that we have divided by two the maximum delay, and reduced by some ms the average delay in the whole transmission. Hence we try to find a middle ground by trying a last scenario with CW size = 350

### CW Size = 350

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117181646401.png" alt="image-20211117181646401" style="zoom:50%;" />

Here, we nearly reach 4.3 Mbps, while keeping a relatively low maximum dela and the same avergae delay over the whole transmission.

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117181719635.png" alt="image-20211117181719635" style="zoom:50%;" />

### Conclusion

We observed the following trend : The lower the CW is set, the more throughput we have : between 100 and 150 ( it should also not be too low) gets us the best throughput. Also it is important to notice that as we augment the CW, the delay gets smaller and smaller, hence we need to reach a tradeoff such that the delay and the throoughput are satisfactory. This may depend on the use case. 

## Final Conclusion regarding CW Size

###### 	We have see that the CW size is highly dependent on the number of stations we are using in our network. As we use x stations  in the network, it seems to be a good rule of thumb to use a CW size of about 10x to 15x, and then to fine tune depending on the use case. 

# Simulation results Regarding CAAiFSn

## 15 Stations

### CAAIFSN = 2

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117183012081.png" alt="image-20211117183012081" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117183034972.png" alt="image-20211117183034972" style="zoom:50%;" />

### CAAIFSN = 4

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117182739203.png" alt="image-20211117182739203" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117182826474.png" alt="image-20211117182826474" style="zoom:50%;" />

### CAAIFSN = 1

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117183422701.png" alt="image-20211117183422701" style="zoom:50%;" />

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211117183409600.png" alt="image-20211117183409600" style="zoom:50%;" />

From these 3 differently set scenarios we can see that the CAAIFSN paramter doesnt change that much the behavior of a network with 15 stations. The only notable difference is that the maximum delay seems to be lower when using 1 as value rather than the default 2 value. The difference is not that flagrant.

Since we did not see any notable impact of changing the ````dot11EDCAAIFSN```` value, I will skip trying the same procedure with networks with different numbers of stations. I will rather invest the remaining time by adding one scenario regarding the CW size.



